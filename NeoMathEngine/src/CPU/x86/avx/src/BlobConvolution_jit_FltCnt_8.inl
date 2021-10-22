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

    const int StepCount = 7;
    const int StepSize = 1;
    const int BatchChannelSize = 16;

    Ymm res[7] = { ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6 };
    Ymm st[7] = { ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13 };
    Ymm f[1] = { ymm14 };

    std::function<void( int )> fillKernel( [&]( int channelCount ) {
        for( int c = 0; c < channelCount; c++ ) {
            size_t fltOffset = c * FltCntM8 * sizeof( float );
            size_t srcOffset = c * sizeof( float );

            // Load one channel for the same pixel as in source for all filters.
            vmovups( f[0], ptr[regTempFltPtr + fltOffset] );

            // Load one channel from one pixels in sequenced windows and fill one ymm register with its value.
            for( int i = 0; i < 7; i++ ) {
                vbroadcastss( st[i], ptr[regTempSrcPtr + srcOffset + i * bc.SrcXStep * sizeof( float )] );
            }

            // Take result for current pixels in three sequenced windows.
            // Multiply one channel for all filters ( for ONE src/flt pixels )
            for( int i = 0; i < 7; i++ ) {
                vfmadd231ps( res[i], f[0], st[i] );
            }
        }
        } );

    initProcessingMainLoop( bc, StepCount, StepSize, BatchChannelSize, fillKernel, windowIndex );
}

template<>
inline void CBlobConvolution<8>::CJitConvolution::fillSingleProcessingKernel( CBlobConvolution<8>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    const int StepCount = 1;
    const int StepSize = 1;
    const int BatchChannelSize = 15;

    Ymm res[1] = { ymm0 };
    // We will accumulate intermediate result in temp registers and then we will flush them to the 'res'.
    Ymm tempRes[5] = { ymm0, ymm1, ymm2, ymm3, ymm4 };
    Ymm s[5] = { ymm5, ymm6, ymm7, ymm8, ymm9 };
    Ymm f[5] = { ymm10, ymm11, ymm12, ymm13, ymm14 };

    // Clear temp regs[1..3]
    vxorps( tempRes[1], tempRes[1], tempRes[1] );
    vxorps( tempRes[2], tempRes[2], tempRes[2] );
    vxorps( tempRes[3], tempRes[3], tempRes[3] );
    vxorps( tempRes[4], tempRes[4], tempRes[4] );

    std::function<void()> mergeResRegs( [&]() {
        vaddps( tempRes[1], tempRes[1], tempRes[2] );
        vaddps( tempRes[3], tempRes[3], tempRes[4] );
        vaddps( tempRes[1], tempRes[1], tempRes[3] );
        vaddps( res[0], tempRes[0], tempRes[1] );
        } );

    std::function<void( int )> fillKernel( [&]( int channelCount ) {
        const int InnerBatchStepSize = 5;
        int offset = 0;
        for( int batchStep = channelCount; batchStep > 0; batchStep -= InnerBatchStepSize ) {
            const int innerChannelCount = min( InnerBatchStepSize, batchStep );
            size_t fltOffset = offset * FltCntM8 * sizeof( float );
            size_t srcOffset = offset * sizeof( float );

            // Load channels
            for( int i = 0; i < innerChannelCount; i++ ) {
                vbroadcastss( s[i], ptr[regTempSrcPtr + srcOffset + i * sizeof( float )] );
            }

            // Load filters
            for( int i = 0; i < innerChannelCount; i++ ) {
                vmovups( f[i], ptr[regTempFltPtr + fltOffset + i * FltCntM8 * sizeof( float )] );
            }

            // Store data into temporary regs
            for( int i = 0; i < innerChannelCount; i++ ) {
                vfmadd231ps( tempRes[i], f[i], s[i] );
            }
            offset += innerChannelCount;
        }
        } );

    initProcessingMainLoop( bc, StepCount, StepSize, BatchChannelSize, fillKernel,
        windowIndex, false, &mergeResRegs );
}

} // namespace NeoML