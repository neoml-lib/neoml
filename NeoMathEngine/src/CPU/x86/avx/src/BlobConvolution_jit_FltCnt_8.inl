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
// Channel count: 8

template<>
const int CBlobConvolution<8>::WideBatchKernelHeight = 1;

template<>
const int CBlobConvolution<8>::WideBatchKernelWidth = 7;

template<>
inline void CBlobConvolution<8>::CJitConvolution::fillBatchProcessingKernel( CBlobConvolution<8>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = 7;
    const int StepSize = 1;

    Ymm res[7] = { ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6 };
    Ymm st[7] = { ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13 };
    Ymm f[1] = { ymm14 };

    initProcessingMainLoop( bc, &res[0], 0, StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex );

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

    // Load one channel from one pixels in sequenced windows and fill one ymm register with its value.
    for( int i = 0; i < 7; i++ ) {
        vbroadcastss( st[i], ptr[regTempSrcPtr + i * bc.SrcXStep * sizeof( float )] );
    }

    // Load one channel for the same pixel as in source for all filters.
    vmovups( f[0], ptr[regTempFltPtr] );

    // Take result for current pixels in three sequenced windows.
    // Multiply one channel for all filters ( for ONE src/flt pixels )
    for( int i = 0; i < 7; i++ ) {
        vfmadd231ps( res[i], f[0], st[i] );
    }

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
inline void CBlobConvolution<8>::CJitConvolution::fillSingleProcessingKernel( CBlobConvolution<8>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = 1;
    const int StepSize = 1;

    Ymm res[1] = { ymm0 };
    // We will accumulate intermediate result in temp registers and then we will flush them to the 'res'.
    Ymm tempRes[4] = { ymm0, ymm1, ymm2, ymm3 };
    Ymm s[6] = { ymm4, ymm5, ymm6, ymm7, ymm8, ymm9 };
    Ymm f[6] = { ymm10, ymm11, ymm12, ymm13, ymm14, ymm15 };

    // Clear temp regs[1..3]
    vxorps( tempRes[1], tempRes[1], tempRes[1] );
    vxorps( tempRes[2], tempRes[2], tempRes[2] );
    vxorps( tempRes[3], tempRes[3], tempRes[3] );
    initProcessingMainLoop( bc, &res[0], 0, StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex );

    ////////////////////////////////////////////////////////////////////////////////////////////

    // channelCount - number of channels for processing
    // isLast - true if it is last of channel chank (we can skip src and flt pointers incrementing )
    auto fillKernel = [&]( int channelCount, bool isLast ) {
        // channelCount are 6, 4, 2 or 1
        // Load channels
        for( int i = 0; i < channelCount; i++ ) {
            vbroadcastss( s[i], ptr[regTempSrcPtr + i * sizeof( float )] );
        }

        // Load filters
        for( int i = 0; i < channelCount; i++ ) {
            vmovups( f[i], ptr[regTempFltPtr + i * FltCntM8 * sizeof( float ) ] );
        }

        if( !isLast ) {
            add( regTempFltPtr, channelCount * SizeOfYmm );
            add( regTempSrcPtr, channelCount * sizeof( float ) );
        }

        // Store data into temporary regs
        if( channelCount <= 4 ) {
            for( int i = 0; i < channelCount; i++ ) {
                vfmadd231ps( res[i], f[i], s[i] );
            }
        } else {
            vfmadd231ps( res[0], f[0], s[0] );
            vmulps( s[1], f[1], s[1] );
            vfmadd231ps( res[1], f[2], s[2] );
            vmulps( s[3], f[3], s[3] );
            vfmadd231ps( res[2], f[4], s[4] );
            vfmadd231ps( res[3], f[5], s[5] );
            vaddps( res[0], res[0], s[1] );
            vaddps( res[1], res[1], s[3] );
        }
    };

    // Steps are 6, 4, 2, 1
    const int BatchStepSize = 6;
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
        fillKernel( 6, isLast );

        if( batchStepCount > 1 ) {
            // } // for( regChCnt i = 0; regChCnt < batchStepCount; regChCnt++ ){}
            inc( regChCnt );
            jmp( labelProcessingKernelStart, T_NEAR );
        }

        L( labelProcessingKernelEnd );
    }

    // Process remained kernels one by one
    if( remainedStepCount >= 4 ) {
        fillKernel( 4, remainedStepCount == 4 );
        remainedStepCount -= 4;
    }
    if( remainedStepCount > 0 ) {
        fillKernel( remainedStepCount, true );
    }
    ret();
    // End of code
    L( labelFillProcessingKernelEnd );
}

} // namespace NeoML
