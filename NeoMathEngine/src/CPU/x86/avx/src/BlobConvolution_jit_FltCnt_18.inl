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

// CBlobConvolution class specializations

namespace NeoML {

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Channel count: 18

template<>
inline void CBlobConvolution<18>::CJitConvolution::initResRegs( Xbyak::Ymm* res, Xbyak::Ymm* tempRes, size_t stepCount, size_t stepSize )
{
    using namespace Xbyak;

    Label labelFillWithZeroes, labelEnd;
    test( regFreeTermPtr, regFreeTermPtr );
    jz( labelFillWithZeroes );

    // Init first register
    for( int i = 0; i < stepSize; i++ ) {
        vmovups( res[i], ptr[regFreeTermPtr + i * SizeOfYmm] );
    }

    for( int step = 1; step < stepCount; step++ ) {
        Xbyak::Ymm* from = &res[( step - 1 ) * stepSize];
        Xbyak::Ymm* to = &res[step * stepSize];
        for( int i = 0; i < stepSize; i++ ) {
            vmovaps( to[i], from[i] );
        }
        rotateLeft6( to[0], to[1], to[2],
                tempRes[0], tempRes[1], tempRes[2] );
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

template<>
inline void CBlobConvolution<18>::CJitConvolution::fillBatchProcessingKernel( CBlobConvolution<18>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = 3;
    const int StepSize = 3;

    Ymm res[3][3] = { { ymm0, ymm1, ymm2 }, { ymm3, ymm4, ymm5 }, { ymm6, ymm7, ymm8 } };
    Ymm f[3] = { ymm9, ymm10, ymm11 };
    Ymm s[3] = { ymm12, ymm13, ymm14 };
    Ymm temp[4] = { ymm12, ymm13, ymm14, ymm15 };

    initProcessingMainLoop( bc, &res[0][0], &temp[0], StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex );

    ////////////////////////////////////////////////////////////////////////////////////////////
    // Batch process kernell function
    L( labelProcessingKernel );
    // for( int c = 0; c < ChCnt; c++ ) {
    if( bc.ChCnt > 1 ) {
        xor_( regChCnt, regChCnt );
        L( labelProcessingKernelStart );
        cmp( regChCnt, bc.ChCnt );
        je( labelProcessingKernelEnd, T_NEAR );
    }

    // We will process four pixels of source in three steps ( merge each two floats for clarity )
    // After each step we will "rotate" filter's values.
    // Step 1:
    // S: 0 0 0 0 - 0 0 0 0 - 0 1 1 1
    // F: 0 1 2 3 - 4 5 6 7 - 8 0 1 2
    vmovups( f[0], ptr[regTempFltPtr] );
    vmovups( f[1], ptr[regTempFltPtr + SizeOfYmm] );
    vmovups( f[2], ptr[regTempFltPtr + 2 * SizeOfYmm] );
    vbroadcastss( s[0], ptr[regTempSrcPtr] );
    vbroadcastss( s[2], ptr[regTempSrcPtr + bc.SrcXStep * sizeof( float )] );
    vfmadd231ps( res[0][0], f[0], s[0] );
    vblendps( s[2], s[2], s[0], 0x03 );
    vfmadd231ps( res[0][1], f[1], s[0] );
    vfmadd231ps( res[0][2], f[2], s[2] );

    // Step 2:
    // S: 1 1 1 1 - 1 1 2 2 - 2 2 2 2
    // F: 3 4 5 6 - 7 8 0 1 - 2 3 4 5
    // Don't use temp2 because it corresponds to s[2] which is used in next step
    rotateLeft6( f[0], f[1], f[2], temp[0], temp[1], temp[3] );
    vunpckhps( s[0], s[2], s[2] );
    vbroadcastss( s[2], ptr[regTempSrcPtr + 2 * bc.SrcXStep * sizeof( float )] );
    vblendps( s[1], s[0], s[2], 0xf0 );
    vfmadd231ps( res[1][0], f[0], s[0] );
    vfmadd231ps( res[1][1], f[1], s[1] );
    vfmadd231ps( res[1][2], f[2], s[2] );

    // Step 3:
    // S: 2 2 2 3 - 3 3 3 3 - 3 3 3 3
    // F: 6 7 8 0 - 1 2 3 4 - 5 6 7 8
    // Don't use temp[2] because it corresponds to s[2] which is used in next step
    rotateLeft6( f[0], f[1], f[2], temp[0], temp[1], temp[3] );
    vbroadcastss( s[1], ptr[regTempSrcPtr + 3 * bc.SrcXStep * sizeof( float )] );
    vblendps( s[0], s[2], s[1], 0xc0 );
    vfmadd231ps( res[2][0], f[0], s[0] );
    vfmadd231ps( res[2][1], f[1], s[1] );
    vfmadd231ps( res[2][2], f[2], s[1] );

    // Go to next channel in filter and source
    add( regTempFltPtr, FltCntM8 * sizeof( float ) );
    add( regTempSrcPtr, sizeof( float ) );

    if( bc.ChCnt > 1 ) {
        inc( regChCnt );
        jmp( labelProcessingKernelStart, T_NEAR );
        // }
        L( labelProcessingKernelEnd );
    }
    ret();
    // End of code
    L( labelFillProcessingKernelEnd );
}

template<>
inline void CBlobConvolution<18>::CJitConvolution::fillSingleProcessingKernel( CBlobConvolution<18>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = 1;
    const int StepSize = 3;

    Ymm res[3] = { ymm0, ymm1, ymm2 };
    Xmm s = xmm3;
    Ymm st0 = ymm4;
    Ymm f[3] = { ymm5, ymm6, ymm7 };

    initProcessingMainLoop( bc, &res[0], 0, StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex );

    ////////////////////////////////////////////////////////////////////////////////////////////

    const int BatchStepSize = 4;
    // channelCount - number of channels for frocessing
    // isLast - true if it is last of channel chank (we can skip src and flt pointers incrementing )
    auto fillKernel = [&]( int channelCount, bool isLast ) {
        // stepCount <= 4
        vmovups( s, ptr[regTempSrcPtr] );
        for( int i = 0; i < channelCount; i++ ) {
            unsigned int mask = i * 0x55;
            vpermilps( st0.copyAndSetKind( Operand::XMM ), s, mask );
            vinsertf128( st0, st0, st0.copyAndSetKind( Operand::XMM ), 1);

            vmovups( f[0], ptr[regTempFltPtr + ( StepSize * i + 0 ) * SizeOfYmm] );
            vmovups( f[1], ptr[regTempFltPtr + ( StepSize * i + 1 ) * SizeOfYmm] );
            vmovups( f[2], ptr[regTempFltPtr + ( StepSize * i + 2 ) * SizeOfYmm] );

            vfmadd231ps( res[0], f[0], st0 );
            vfmadd231ps( res[1], f[1], st0 );
            vfmadd231ps( res[2], f[2], st0 );

        }
        if( !isLast ) {
            add( regTempFltPtr, StepSize * channelCount * SizeOfYmm );
        }
    };

    // Single process kernell function
    int singleStepCount = bc.ChCnt % BatchStepSize;
    int batchStepCount = bc.ChCnt / BatchStepSize;

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
        // Only one batch step
        bool isLast = ( singleStepCount == 0 ) && ( batchStepCount == 1 );
        fillKernel( 4, isLast );

        if( !isLast ) {
            add( regTempSrcPtr, BatchStepSize * sizeof( float ) );
        }

        if( batchStepCount > 1 ) {
            // } // for( regChCnt i = 0; regChCnt < batchStepCount; regChCnt++ ){}
            inc( regChCnt );
            jmp( labelProcessingKernelStart, T_NEAR );
        }

        L( labelProcessingKernelEnd );
    }

    // Process remained kernels one by one
    if( singleStepCount  ) {
        fillKernel( singleStepCount, true );
    }
    ret();
    // End of code
    L( labelFillProcessingKernelEnd );
}

} // namespace NeoML
