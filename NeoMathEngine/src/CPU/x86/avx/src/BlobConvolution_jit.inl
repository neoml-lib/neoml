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
        ret();
    };

    // Start code filling
    prologue();
#ifdef _WIN32
    // Parameters are in reverse order in stack
    // First two values are 'rip' and 'rbp', then two values pushed in prologue
    const int StackOffset = 8 * sizeof( void* );
    mov( regResPtr, ptr[rsp + StackOffset] );
#endif

    Label labelNarrow;
    if( hasNarrowProcessing ) {
        // Add selector narrow/wide
        test( regUseNarrowProcessing, regUseNarrowProcessing );
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
    push( rbp );
    mov( rbp, rsp );
#ifdef _WIN32
    push( regResPtr );
#endif
    push( regNumSteps );
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::CJitConvolution::epilogue()
{
    pop( regNumSteps );
#ifdef _WIN32
    pop( regResPtr );
#endif
    leave();
}

// Implementation for cases when FltCnt == FltCntM8
template<int FltCnt>
inline void CBlobConvolution<FltCnt>::CJitConvolution::initResRegs( Xbyak::Ymm* res, Xbyak::Ymm* tempRes, size_t stepCount, size_t stepSize )
{
    using namespace Xbyak;

    Label labelFillWithZeroes, labelEnd;
    test( regFreeTermPtr, regFreeTermPtr );
    jz( labelFillWithZeroes );
    // Init first batch of registers
    for( int i = 0; i < stepSize; i++ ) {
        vmovups( res[i], ptr[regFreeTermPtr + SizeOfYmm * i ] );
    }
    // Duplicate first batch into another ones
    size_t destIdx = stepSize;
    for( int j = 1; j < stepCount; j++ ) {
        for( int i = 0; i < stepSize; i++ ) {
            vmovups( res[destIdx++], res[i] );
        }
    }
    jmp( labelEnd, T_NEAR );

    L( labelFillWithZeroes );
    // Init with zeroes
    for( int i = 0; i < stepCount * stepSize; i++ ) {
        vxorps( *res, *res, *res );
        res++;
    }
    L( labelEnd );
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::CJitConvolution::flushResRegs( Xbyak::Ymm* res, size_t stepCount, size_t stepSize, bool useNarrowProcessing, size_t resNarrowStep )
{
    using namespace Xbyak;

    Label labelPartialStore, labelPartialStoreEnd;
    // Last register is always unused at the end of the processing
    Ymm regMask = ymm15;

    // "narrow" kernel processes several rows per time
    const bool HasSeveralRows = resNarrowStep != 0;
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
            vmovups( ptr[regResPtr + ( resNarrowStepDisp  + offsetDisp ) * sizeof( float )], res[i * ColCount + j] );
            offsetDisp += NumFloatInYmm;
        }

        if( HasPartitialFlush ) {
            // Mask store ( only elements which is mentioned in regMask )
            vmaskmovps( ptr[regResPtr + ( resNarrowStepDisp  + offsetDisp ) * sizeof( float )], regMask, res[i * ColCount + j] );
        }

        if( resNarrowStep != 0 ) {
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
        CBlobConvolution<FltCnt>& bc, Xbyak::Ymm* res, Xbyak::Ymm* tempRes,
        size_t stepCount, size_t stepSize,
        Xbyak::Label& labelKernel, Xbyak::Label& labelEndOfProcessingFunction,
        size_t windowIndex,
        bool useNarrowProcessing, size_t resNarrowStep, std::function<void()>* callBeforeFlush )
{
    // Initialize result registers with freeTerm
    initResRegs( res, tempRes, stepCount, stepSize );

    // Process convolution
    auto srcIt = bc.SrcPixelsOffset[windowIndex].cbegin();
    auto fltIt = bc.FltPixelsOffset[windowIndex].cbegin();

    for( ; srcIt != bc.SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
        lea( regTempSrcPtr, ptr[regSrcPtr + *srcIt * sizeof( float )] );
        lea( regTempFltPtr, ptr[regFltPtr + *fltIt * sizeof( float )] );
        call( labelKernel );
    }

    if( callBeforeFlush ) {
        ( *callBeforeFlush )();
    }

    // Flush result registers
    flushResRegs( res, stepCount, stepSize, useNarrowProcessing, resNarrowStep );

    // return from function
    jmp( labelEndOfProcessingFunction, T_NEAR );
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::CJitConvolution::rotateLeft6(
        Xbyak::Ymm& y0, Xbyak::Ymm& y1, Xbyak::Ymm& y2,
        Xbyak::Ymm& yt0, Xbyak::Ymm& yt1, Xbyak::Ymm& yt2 )
{   //   y0        y1        y2
    // 0 1 2 3 - 4 5 6 7 - 8 0 1 2
    // 3 4 5 6 - 7 8 0 1 - 2 3 4 5
    // 6 7 8 0 - 1 2 3 4 - 5 6 7 8

    // before: 0 1 2 3
    // after:  2 3 0 1
    vperm2f128( yt0, y0, y0, _MM_SHUFFLE( 0, 0, 0, 1 ) );
    // before: 4 5 6 7
    // after:  6 7 4 5
    vperm2f128( yt1, y1, y1, _MM_SHUFFLE( 0, 0, 0, 1 ) );
    // before: 6 7 4 5|8 0 1 2
    // after:      7 8 5 1
    vshufps( yt2, yt1, y2, _MM_SHUFFLE( 1, 0, 3, 2 ) );
    // before: 2 3 0 1|6 7 4 5
    // after:      2 3 4 5
    vblendps( y2, yt0, yt1, 0xf0 );
    // before: 2 3 4 5|4 5 6 7
    // after:      3 4 5 6
    vshufps( y0, y2, y1, _MM_SHUFFLE( 1, 0, 3, 2 ) );
    // before: 7 8 5 1|2 3 0 1
    // after:      7 8 0 1
    vblendps( y1, yt2, yt0, 0xf0 );
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::CJitConvolution::rotateLeft2(
        Xbyak::Ymm& y, Xbyak::Ymm& yt )
{
    // 0 1 2 0
    // 1 2 0 1
    // 2 0 1 2
    // before: 0 1 2 0
    // after:  2 0 0 1
    vperm2f128( yt, y, y, _MM_SHUFFLE( 0, 0, 0, 1 ) );
    // before: 0 1 2 0|2 0 0 1
    // after:      1 2 0 0
    vshufps( y, y, yt, _MM_SHUFFLE( 1, 0, 3, 2 ) );
    // before:  1 2 0 0|2 0 0 1
    // after:      1 2 0 1
    vblendps( y, y, yt, 0xf0 );
}

} // namespace NeoML
