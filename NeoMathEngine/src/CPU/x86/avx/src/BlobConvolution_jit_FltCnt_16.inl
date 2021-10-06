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
// Channel count: 16

template<>
const int CBlobConvolution<16>::WideBatchKernelHeight = 1;

template<>
const int CBlobConvolution<16>::WideBatchKernelWidth = 5;

template<>
inline void CBlobConvolution<16>::CJitConvolution::fillBatchProcessingKernel( CBlobConvolution<16>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = 5;
    const int StepSize = 2;

    Ymm res[5][2] = { { ymm0, ymm1 }, { ymm2, ymm3 }, { ymm4, ymm5 }, { ymm6, ymm7 }, { ymm8, ymm9 } };
    // We will first load 4 pixels, store them. Remained pixel will be loaded between storring of 3-d and 4-th result pixels
    Ymm st[4] = { ymm10, ymm11, ymm12, ymm13 };
    Ymm f[2] = { ymm14, ymm15 };

    initProcessingMainLoop( bc, &res[0][0], 0, StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex );

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

    // Load one channel for the same pixel as in source for all filters.
    vmovups( f[0], ptr[regTempFltPtr] );
    vmovups( f[1], ptr[regTempFltPtr + SizeOfYmm] );

    // Load one channel from one pixels in sequenced windows and fill one ymm register with its value.
    for( int i = 0; i < 4; i++ ) {
        vbroadcastss( st[i], ptr[regTempSrcPtr + i * bc.SrcXStep * sizeof( float )] );
    }
    // Multiply and accumulate result pixels [0..2]
    for( int i = 0; i < 3; i++ ) {
        vfmadd231ps( res[i][0], f[0], st[i] );
        vfmadd231ps( res[i][1], f[1], st[i] );
    }
    // Load remained pixel
    vbroadcastss( st[0], ptr[regTempSrcPtr + 4 * bc.SrcXStep * sizeof( float )] );
    vfmadd231ps( res[3][0], f[0], st[3] );
    vfmadd231ps( res[3][1], f[1], st[3] );
    vfmadd231ps( res[4][0], f[0], st[0] );
    vfmadd231ps( res[4][1], f[1], st[0] );

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
inline void CBlobConvolution<16>::CJitConvolution::fillSingleProcessingKernel( CBlobConvolution<16>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = 1;
    const int StepSize = 2;

    Ymm res[2] = { ymm0, ymm1 };
    // We will accumulate intermediate result in temp registers and then we will flush them to the 'res'.
    Ymm tempRes[2][2] = { { ymm0, ymm1 }, { ymm2, ymm3 } };
    Ymm s[4] = { ymm4, ymm5, ymm6, ymm7 };
    Ymm f[4][2] = { { ymm8, ymm9 }, { ymm10, ymm11 }, { ymm12, ymm13 }, { ymm14, ymm15 } };

    // Clear temp reg[1][..]
    vxorps( tempRes[1][0], tempRes[1][0], tempRes[1][0] );
    vxorps( tempRes[1][1], tempRes[1][1], tempRes[1][1] );
    std::function<void()> mergeResRegs( [&]() {
        vaddps( res[0], tempRes[0][0], tempRes[1][0] );
        vaddps( res[1], tempRes[0][1], tempRes[1][1] );
    } );

    initProcessingMainLoop( bc, &res[0], 0, StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex,
            false, 0, &mergeResRegs );

    ////////////////////////////////////////////////////////////////////////////////////////////

    // channelCount - number of channels for processing
    // isLast - true if it is last of channel chank (we can skip src and flt pointers incrementing )
    auto fillKernel = [&]( int channelCount, bool isLast ) {
        // Load channels
        for( int i = 0; i < channelCount; i++ ) {
            vbroadcastss( s[i], ptr[regTempSrcPtr + i * sizeof( float )] );
        }

        // Load filters
        for( int i = 0; i < channelCount; i++ ) {
            vmovups( f[i][0], ptr[regTempFltPtr + i * FltCntM8 * sizeof( float ) ] );
            vmovups( f[i][1], ptr[regTempFltPtr + ( i * FltCntM8 + NumFloatInYmm ) * sizeof( float ) ] );
        }

        if( !isLast ) {
            add( regTempFltPtr, channelCount * FltCntM8 * sizeof( float ) );
            add( regTempSrcPtr, channelCount * sizeof( float ) );
        }

        // Store data into temporary regs
        for( int i = 0; i < channelCount; i++ ) {
            vfmadd231ps( tempRes[i % 2][0], f[i][0], s[i] );
            vfmadd231ps( tempRes[i % 2][1], f[i][1], s[i] );
        }
    };

    // Steps are 4, 2, 1
    const int BatchStepSize = 4;
    int batchStepCount = bc.ChCnt / BatchStepSize;
    int remainedStepCount = bc.ChCnt % BatchStepSize;

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
        bool isLast = ( remainedStepCount == 0 ) && ( batchStepCount == 1 );
        fillKernel( 4, isLast );

        if( batchStepCount > 1 ) {
            // } // for( regChCnt i = 0; regChCnt < batchStepCount; regChCnt++ ){}
            inc( regChCnt );
            jmp( labelProcessingKernelStart, T_NEAR );
        }

        L( labelProcessingKernelEnd );
    }

    // Process remained kernels one by one
    if( remainedStepCount >= 2 ) {
        fillKernel( 2, remainedStepCount == 2 );
        remainedStepCount -= 2;
    }
    if( remainedStepCount > 0 ) {
        fillKernel( remainedStepCount, true );
    }
    ret();
    // End of code
    L( labelFillProcessingKernelEnd );
}

} // namespace NeoML
