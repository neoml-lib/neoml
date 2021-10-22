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
// Channel count: 32

template<>
const int CBlobConvolution<32>::WideBatchKernelHeight = 1;

template<>
const int CBlobConvolution<32>::WideBatchKernelWidth = 2;

template<>
inline void CBlobConvolution<32>::CJitConvolution::fillBatchProcessingKernel( CBlobConvolution<32>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = 2;
    const int StepSize = 4;

    Ymm res[2][4] = { { ymm0, ymm1, ymm2, ymm3 }, { ymm4, ymm5, ymm6, ymm7 } };
    Ymm st[2] = { ymm8, ymm9 };
    Ymm f[4] = { ymm10, ymm11, ymm12, ymm13 };

    initProcessingMainLoop( bc, &res[0][0], 0, StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex );

    ////////////////////////////////////////////////////////////////////////////////////////////
    auto fillKernel = [&]( int channelCount ) {
        for( int i = 0; i < channelCount; i++ ) {
            size_t fltOffset = i * FltCntM8 * sizeof( float );
            size_t srcOffset = i * sizeof( float );
            // Load one channel from one pixels in sequenced windows and fill one ymm register with its value.
            vbroadcastss( st[0], ptr[regTempSrcPtr + srcOffset] );
            vbroadcastss( st[1], ptr[regTempSrcPtr + srcOffset + bc.SrcXStep * sizeof( float )] );
            // Load one channel for the same pixel as in source for all filters.
            vmovups( f[0], ptr[regTempFltPtr + fltOffset] );
            vmovups( f[1], ptr[regTempFltPtr + fltOffset + SizeOfYmm] );
            vmovups( f[2], ptr[regTempFltPtr + fltOffset + 2 * SizeOfYmm] );
            vmovups( f[3], ptr[regTempFltPtr + fltOffset + 3 * SizeOfYmm] );
            // Take result for current pixels in three sequenced windows.
            // Multiply one channel for all filters ( for ONE src/flt pixels )
            vfmadd231ps( res[0][0], f[0], st[0] );
            vfmadd231ps( res[0][1], f[1], st[0] );
            vfmadd231ps( res[0][2], f[2], st[0] );
            vfmadd231ps( res[0][3], f[3], st[0] );
            vfmadd231ps( res[1][0], f[0], st[1] );
            vfmadd231ps( res[1][1], f[1], st[1] );
            vfmadd231ps( res[1][2], f[2], st[1] );
            vfmadd231ps( res[1][3], f[3], st[1] );
        }

        // Go to next channel in filter and source
        add( regTempFltPtr, channelCount * FltCntM8 * sizeof( float ) );
        add( regTempSrcPtr, channelCount * sizeof( float ) );
    };

    const int BatchStepSize = 16;
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

        fillKernel( BatchStepSize );

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

template<>
inline void CBlobConvolution<32>::CJitConvolution::fillSingleProcessingKernel( CBlobConvolution<32>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = 1;
    const int StepSize = 4;

    Ymm res[4] = { ymm0, ymm1, ymm2, ymm3 };
    Ymm s[2] = { ymm5, ymm6 };
    Ymm f[2][4] = { { ymm7, ymm8, ymm9, ymm10 }, { ymm11, ymm12, ymm13, ymm14 } };

    initProcessingMainLoop( bc, &res[0], 0, StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex );

    ////////////////////////////////////////////////////////////////////////////////////////////

    // channelCount - number of channels for processing
    auto fillKernel = [&]( int channelCount ) {
        const int InnerBatchStepSize = 2;
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
                vmovups( f[i][0], ptr[regTempFltPtr + fltOffset + i * FltCntM8 * sizeof( float ) + 0 * SizeOfYmm] );
                vmovups( f[i][1], ptr[regTempFltPtr + fltOffset + i * FltCntM8 * sizeof( float ) + 1 * SizeOfYmm] );
                vmovups( f[i][2], ptr[regTempFltPtr + fltOffset + i * FltCntM8 * sizeof( float ) + 2 * SizeOfYmm] );
                vmovups( f[i][3], ptr[regTempFltPtr + fltOffset + i * FltCntM8 * sizeof( float ) + 3 * SizeOfYmm] );
            }

            // Store data into temporary regs
            for( int i = 0; i < innerChannelCount; i++ ) {
                vfmadd231ps( res[0], f[i][0], s[i] );
                vfmadd231ps( res[1], f[i][1], s[i] );
                vfmadd231ps( res[2], f[i][2], s[i] );
                vfmadd231ps( res[3], f[i][3], s[i] );
            }
            offset += innerChannelCount;
        }

        // Go to next channel in filter and source
        add( regTempFltPtr, channelCount * FltCntM8 * sizeof( float ) );
        add( regTempSrcPtr, channelCount * sizeof( float ) );
    };

    const int BatchStepSize = 8;
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

        fillKernel( BatchStepSize );

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
