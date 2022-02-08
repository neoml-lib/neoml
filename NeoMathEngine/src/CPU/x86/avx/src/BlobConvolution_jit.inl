/* Copyright © 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

namespace NeoML {

using reg64_t = Xbyak::Reg64;

template<int FltCnt>
CBlobConvolution<FltCnt>::CJitConvolution::CJitConvolution( CBlobConvolution<FltCnt>& bc, int yStepIndex )
    : Xbyak::CodeGenerator( 256 << 10 )
{
    using namespace Xbyak::util;
    using namespace Xbyak;

    // If class doesn't has narrow processing, both height and width are equal to INT_MAX
    const bool hasNarrowProcessing = bc.NarrowBatchProcessSize.Height != INT_MAX;

    // Process one rxSize in addRowProcessing
    auto addStepProcessing = [&]( bool useNarrowProcessing, size_t numSteps, size_t stepSize, size_t windowIndex ) {
        Label labelProcessingStart, labelProcessingEnd;

        // Skip using loop if we have only one batch step
        if( numSteps > 1 ) {
            // Iterate through all variants of src and flt intersection by axis X
            mov( regNumSteps, numSteps );

            // for( ; numSteps > 0; numSteps-- ) {
            L( labelProcessingStart );
            dec( regNumSteps );
            js( labelProcessingEnd, T_NEAR );
        }

        // Do we have any batch step at all?
        if( numSteps > 0 ) {
            if( stepSize == 1 ) {
                fillSingleProcessingKernel( bc, useNarrowProcessing, windowIndex );
            } else {
                fillBatchProcessingKernel( bc, useNarrowProcessing, windowIndex );
            }

            add( regSrcPtr, static_cast<uint32_t>( stepSize * bc.SrcXStep * sizeof( float ) ) );
            add( regResPtr, static_cast< uint32_t >( stepSize * FltCnt * sizeof( float ) ) );
        }

        if( numSteps > 1 ) {
            // } // for( ; numSteps > 0; numSteps-- )
            jmp( labelProcessingStart, T_NEAR );
            L( labelProcessingEnd );
        }
    };

    // Process whole row
    auto addRowProcessing = [&]( bool useNarrowProcessing ) {
        size_t windowIndex = yStepIndex * bc.PixelOffsetResStepsWidthX.size();
        for( auto& rxSize : bc.PixelOffsetResStepsWidthX ) {
            const int batchStep = useNarrowProcessing ? bc.NarrowBatchProcessSize.Width : bc.WideBatchProcessSize.Width;
            const int numBatchProcessing = rxSize / batchStep;
            const int numSingleProcessing = rxSize % batchStep;

            // Add batch processing (if required)
            addStepProcessing( useNarrowProcessing, numBatchProcessing, batchStep, windowIndex );

            // Add single processing (if required)
            addStepProcessing( useNarrowProcessing, numSingleProcessing, 1, windowIndex );

            windowIndex++;
        }

        epilogue();

        // !!! This instruction should always be called at the end of AVX code.
        // Intel® 64 and IA - 32 Architectures Optimization Reference Manual, item 11.3.1
        // Assembly / Compiler Coding Rule 72. ( H impact, H generality ) Add VZEROUPPER instruction after
        // 256 - bit AVX instructions are executed and before any function call that might execute SSE code.Add
        // VZEROUPPER at the end of any function that uses 256 - bit AVX instructions.
        vzeroupper();
        ret();
    };

    // Start code filling
    prologue();
#ifdef _WIN32
    // Parameters are in reverse order in stack
    // First two values are 'rip' and 'rbp'
    const int StackOffset = 6 * sizeof( void* );
    mov( regResPtr, ptr[rbp + StackOffset] );
#endif

    Label labelNarrow;
    if( hasNarrowProcessing ) {
        // Add selector narrow/wide
        test( regUseNarrowProcessing.cvt8(), regUseNarrowProcessing.cvt8() );
        jnz( labelNarrow, T_NEAR );
    }

    // Fill wide processing
    addRowProcessing( false );

    // Fill narrow processing (if applied)
    if( hasNarrowProcessing ) {
        L( labelNarrow );
        addRowProcessing( true );
    }
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::CJitConvolution::Run( bool useNarrowProcessing, const float* srcPtr, const float* fltPtr, const float* freeTermPtr, float* resPtr )
{
    getCode<void( * )( bool, const float*, const float*, const float*, float* )>()( useNarrowProcessing, srcPtr, fltPtr, freeTermPtr, resPtr );
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::CJitConvolution::prologue()
{
    using namespace Xbyak::util;
    using namespace Xbyak;
    push( rbp );
    mov( rbp, rsp );

    // stack should be alinged to 16 byte because we use vmovaps instruction
    int stackAlignment = 0;
#ifdef _WIN32
    stackAlignment = 8;
    push( regResPtr );
#endif
    push( regNumSteps );
    push( retTemp );
    for( int i = 6; i <= 15; i++ ) {
        // '-16' - place for first xmm
        vmovaps( ptr[rsp - stackAlignment - 16 - ( i - 6 ) * 16], Xmm( i ) );
    }
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::CJitConvolution::epilogue()
{
    using namespace Xbyak::util;
    using namespace Xbyak;
    // stack should be alinged to 16 byte because we use vmovaps instruction
    int stackAlignment = 0;
#ifdef _WIN32
    stackAlignment = 8;
#endif

    for( int i = 15; i >= 6; i-- ) {
        vmovaps( Xmm( i ), ptr[rsp - stackAlignment - 16  - ( i - 6 ) * 16] );
    }
    pop( retTemp );
    pop( regNumSteps );

#ifdef _WIN32
    pop( regResPtr );
#endif
    leave();
}

// Implementation for cases when FltCnt == FltCntM8
template<int FltCnt>
inline void CBlobConvolution<FltCnt>::CJitConvolution::initResRegs( size_t stepCount, size_t stepSize )
{
    using namespace Xbyak;

    Ymm resRegs[] = { ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
        ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15 };
    // tempRegs are always in reverse order in order to not overlap with resRegs.
    Ymm tempRegs[] = { ymm15, ymm14, ymm13, ymm12 };

    Label labelFillWithZeroes, labelEnd;
    test( regFreeTermPtr, regFreeTermPtr );
    jz( labelFillWithZeroes );
   
    // For single processing we haven't any shifts because stepSize equals to 1.
    // In another cases NumberOfShifts means number of shifts we should 
    // perform in order  before we come back to initial state.
    // Example for FC == 3:
    //        r0                r1                r2                r3
    // 0 1 2 0 1 2 0 1 | 2 0 1 2 0 1 2 0 | 1 2 0 1 2 0 1 2 | 0 1 2 0 1 2 0 1
    // We see that r3 is the same as r0, therefore number of shift is 3 (r0, r1, r2)
    const int NumberOfShifts = static_cast<int>( min( stepSize, FltCntM8 == FltCnt ? 1 : stepSize ) );
    // If we shift registers we will do that every ShiftStep regs.
    constexpr int ChunkSize = FltCntM8 / 8;
    // Number of identical chunks
    const size_t NumberOfCopies = stepSize * stepCount / ( NumberOfShifts * ChunkSize );

    // Init first batch of registers
    if( FltCnt <= 4 && NumberOfShifts == 1 ) {
        // High half of register should be zerroed because we might use it for result accumulation.
        vmovups( resRegs[0].copyAndSetKind( Operand::XMM ), ptr[regFreeTermPtr] );
    } else {
        for( int i = 0; i < ChunkSize; i++ ) {
            vmovups( resRegs[i], ptr[regFreeTermPtr + SizeOfYmm * i] );
        }
    }
   
    for( int shiftIdx = 0; shiftIdx < NumberOfShifts; shiftIdx++ ) {
        // Duplicate to the identical chunk
        for( int i = 1; i < NumberOfCopies; i++ ) {
            for( int j = 0; j < ChunkSize; j++ ) {
                vmovaps( resRegs[i * stepSize + j + shiftIdx], resRegs[j + shiftIdx] );
            }
        }

        if( shiftIdx < ( NumberOfShifts - 1 ) ) {
            circularShift( &resRegs[( shiftIdx + 1 ) * ChunkSize], &resRegs[shiftIdx * ChunkSize], tempRegs );
        }
    }

    jmp( labelEnd, T_NEAR );

    L( labelFillWithZeroes );
    // Init with zeroes
    for( int i = 0; i < stepCount * stepSize; i++ ) {
        vxorps( resRegs[i], resRegs[i], resRegs[i] );
    }
    L( labelEnd );
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::CJitConvolution::flushResRegs( CBlobConvolution<FltCnt>& bc, size_t stepCount, size_t stepSize, bool useNarrowProcessing )
{
    using namespace Xbyak;

    const size_t resNarrowStep = useNarrowProcessing ? bc.ResLineStride : stepSize * NumFloatInYmm;

    Label labelPartialStore, labelPartialStoreEnd;
    Ymm resRegs[] = { ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
        ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15 };
    // Last register is always unused at the end of the processing
    Ymm regMask = ymm15;

    // "narrow" kernel processes several rows per time
    const bool HasSeveralRows = useNarrowProcessing;
    const size_t RowCount = HasSeveralRows ? stepCount : 1;
    const size_t ColCount = HasSeveralRows ? stepSize : stepCount * stepSize;
    // If length of data for flushing doesn't multiple of size of ymm we should perform storring tail of data by mask.
    const bool HasPartitialFlush = ColCount * NumFloatInYmm % FltCnt != 0;
    // Number of ymm registers stored fully.
    const size_t FullFlushCount = HasPartitialFlush ? ColCount - 1 : ColCount;

    if( HasPartitialFlush ) {
        vmovdqa( regMask, ptr[rip + labelPartialStore] );
    }

    size_t offsetDisp = 0;
    size_t resNarrowStepDisp = 0;
    for( int i = 0; i < RowCount; i++ ) {
        int j = 0;
        for( ; j < FullFlushCount; j++ ) {
            vmovups( ptr[regResPtr + ( resNarrowStepDisp  + offsetDisp ) * sizeof( float )], resRegs[i * ColCount + j] );
            offsetDisp += NumFloatInYmm;
        }

        if( HasPartitialFlush ) {
            // Mask store ( only elements which is mentioned in regMask )
            vmaskmovps( ptr[regResPtr + ( resNarrowStepDisp  + offsetDisp ) * sizeof( float )], regMask, resRegs[i * ColCount + j] );
        }

        if( useNarrowProcessing ) {
            resNarrowStepDisp += resNarrowStep;
            offsetDisp = 0;
        }
    }
    if( HasPartitialFlush ) {
        jmp( labelPartialStoreEnd, T_NEAR );
        align( 32 );
        L( labelPartialStore );
        for( int i = 0; i < FltCnt % 8; i++ ){
            dd( -1 );
        }
        for( int i = 0; i < 8 - FltCnt % 8; i++ ){
            dd( 0 );
        }
        L( labelPartialStoreEnd );
    }
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::CJitConvolution::initProcessingMainLoop(
        CBlobConvolution<FltCnt>& bc,
        size_t stepCount, size_t stepSize, int batchChannelSize, const std::function<void(int)>& fillKernel,
        size_t windowIndex, bool useNarrowProcessing, const std::function<void()>* callBeforeFlush )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;

    // Initialize result registers with freeTerm
    initResRegs( stepCount, stepSize );

    // Process convolution
    auto srcIt = bc.SrcPixelsOffset[windowIndex].cbegin();
    auto fltIt = bc.FltPixelsOffset[windowIndex].cbegin();

    for( ; srcIt != bc.SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
        lea( regTempSrcPtr, ptr[regSrcPtr + *srcIt * sizeof( float )] );
        lea( regTempFltPtr, ptr[regFltPtr + *fltIt * sizeof( float )] );
        call( labelProcessingKernel );
    }

    if( callBeforeFlush ) {
        ( *callBeforeFlush )();
    }

    // Flush result registers
    flushResRegs( bc, stepCount, stepSize, useNarrowProcessing );

    // return from function
    jmp( labelFillProcessingKernelEnd, T_NEAR );

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Fill kernels
    int batchStepCount = bc.ChCnt / batchChannelSize;
    int remainedStepCount = bc.ChCnt % batchChannelSize;

    L( labelProcessingKernel );
    // Process kernels in group
    if( batchStepCount ) {
        if( batchStepCount > 1 ) {
            // for( regChCnt i = 0; regChCnt < batchStepCount; regChCnt++ ) {
            // If we need loop
            xor_( regChCnt, regChCnt );
            L( labelProcessingKernelStart );
            cmp( regChCnt, batchStepCount );
            je( labelProcessingKernelEnd, T_NEAR );
        }

        fillKernel( batchChannelSize );
        // Go to next channel in filter and source
        add( regTempFltPtr, batchChannelSize * FltCntM8 * sizeof( float ) );
        add( regTempSrcPtr, batchChannelSize * sizeof( float ) );

        if( batchStepCount > 1 ) {
            // } // for( regChCnt i = 0; regChCnt < batchStepCount; regChCnt++ ){}
            inc( regChCnt );
            jmp( labelProcessingKernelStart, T_NEAR );
        }

        L( labelProcessingKernelEnd );
    }

    if( remainedStepCount > 0 ) {
        fillKernel( remainedStepCount );
    }

    ret();
    // End of code
    L( labelFillProcessingKernelEnd );
}

} // namespace NeoML
