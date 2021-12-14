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
const int CBlobConvolution<18>::WideBatchKernelHeight = 1;

template<>
const int CBlobConvolution<18>::WideBatchKernelWidth = 4;

template<>
inline void CBlobConvolution<18>::CJitConvolution::circularShift( Xbyak::Ymm* dst, Xbyak::Ymm* src, Xbyak::Ymm* temp )
{
    // src[0]    src[1]    src[2]
    // 0 1 2 3 - 4 5 6 7 - 8 0 1 2
    // 3 4 5 6 - 7 8 0 1 - 2 3 4 5
    // 6 7 8 0 - 1 2 3 4 - 5 6 7 8

    // before: 0 1 2 3
    // after:  2 3 0 1
    vperm2f128( temp[0], src[0], src[0], _MM_SHUFFLE( 0, 0, 0, 1 ) );
    // before: 4 5 6 7
    // after:  6 7 4 5
    vperm2f128( temp[1], src[1], src[1], _MM_SHUFFLE( 0, 0, 0, 1 ) );
    // before: 6 7 4 5|8 0 1 2
    // after:      7 8 5 1
    vshufps( temp[2], temp[1], src[2], _MM_SHUFFLE( 1, 0, 3, 2 ) );
    // before: 2 3 0 1|6 7 4 5
    // after:      2 3 4 5
    vblendps( dst[2], temp[0], temp[1], 0xf0 );
    // before: 2 3 4 5|4 5 6 7
    // after:      3 4 5 6
    vshufps( dst[0], dst[2], src[1], _MM_SHUFFLE( 1, 0, 3, 2 ) );
    // before: 7 8 5 1|2 3 0 1
    // after:      7 8 0 1
    vblendps( dst[1], temp[2], temp[0], 0xf0 );
}

template<>
inline void CBlobConvolution<18>::CJitConvolution::fillBatchProcessingKernel( CBlobConvolution<18>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    const int StepCount = 3;
    const int StepSize = 3;
    const int BatchChannelSize = 1;

    Ymm res[3][3] = { { ymm0, ymm1, ymm2 }, { ymm3, ymm4, ymm5 }, { ymm6, ymm7, ymm8 } };
    Ymm f[3] = { ymm9, ymm10, ymm11 };
    Ymm s[3] = { ymm12, ymm13, ymm14 };
    Ymm temp[3] = { ymm12, ymm13, ymm15 };

    std::function<void( int )> fillKernel( [&]( int channelCount ) {
        PRESUME_EXPR( channelCount == 1 );

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
        circularShift( f, f, temp );
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
        circularShift( f, f, temp );
        vbroadcastss( s[1], ptr[regTempSrcPtr + 3 * bc.SrcXStep * sizeof( float )] );
        vblendps( s[0], s[2], s[1], 0xc0 );
        vfmadd231ps( res[2][0], f[0], s[0] );
        vfmadd231ps( res[2][1], f[1], s[1] );
        vfmadd231ps( res[2][2], f[2], s[1] );
        } );

    initProcessingMainLoop( bc, StepCount, StepSize, BatchChannelSize, fillKernel, windowIndex );
}

template<>
inline void CBlobConvolution<18>::CJitConvolution::fillSingleProcessingKernel( CBlobConvolution<18>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    const int StepCount = 1;
    const int StepSize = 3;
    const int BatchChannelSize = 4;

    Ymm res[3] = { ymm0, ymm1, ymm2 };
    Ymm s = ymm3;
    Ymm st0 = ymm4;
    Ymm f[3] = { ymm5, ymm6, ymm7 };
    Ymm st[2] = { ymm8, ymm9 };

    std::function<void( int )> fillKernel( [&]( int channelCount ) {
        PRESUME_EXPR( channelCount <= 4 );
        switch( channelCount ) {
        case 4:
            vmovups( s.copyAndSetKind( Operand::XMM ), ptr[regTempSrcPtr] );
            break;
        default:
            // Create bitmask
            vxorps( st[0], st[0], st[0] );
            vpcmpeqd( st[1], st[1], st[1] );
            vblendps( st[1], st[0], st[1], 0xff >> ( 8 - channelCount ) );
            vmaskmovps( s, st[1], ptr[regTempSrcPtr] );
        }
        for( int i = 0; i < channelCount; i++ ) {
            unsigned int mask = i * 0x55;
            vpermilps( st0.copyAndSetKind( Operand::XMM ), s, mask );
            vinsertf128( st0, st0, st0.copyAndSetKind( Operand::XMM ), 1 );

            vmovups( f[0], ptr[regTempFltPtr + ( StepSize * i + 0 ) * SizeOfYmm] );
            vmovups( f[1], ptr[regTempFltPtr + ( StepSize * i + 1 ) * SizeOfYmm] );
            vmovups( f[2], ptr[regTempFltPtr + ( StepSize * i + 2 ) * SizeOfYmm] );

            vfmadd231ps( res[0], f[0], st0 );
            vfmadd231ps( res[1], f[1], st0 );
            vfmadd231ps( res[2], f[2], st0 );

        }
        } );
    initProcessingMainLoop( bc, StepCount, StepSize, BatchChannelSize, fillKernel, windowIndex );
}

} // namespace NeoML
