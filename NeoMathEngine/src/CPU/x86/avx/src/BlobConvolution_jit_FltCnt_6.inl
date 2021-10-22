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
// Channel count: 6

template<>
const int CBlobConvolution<6>::NarrowBatchKernelHeight = 3;

template<>
const int CBlobConvolution<6>::NarrowBatchKernelWidth = 4;

template<>
const int CBlobConvolution<6>::WideBatchKernelHeight = 1;

template<>
const int CBlobConvolution<6>::WideBatchKernelWidth = 12;

template<>
inline void CBlobConvolution<6>::CJitConvolution::initResRegs( Xbyak::Ymm* res, Xbyak::Ymm* tempRes, size_t stepCount, size_t StepSize )
{
    using namespace Xbyak;

    Label labelFillWithZeroes, labelEnd;
    test( regFreeTermPtr, regFreeTermPtr );
    jz( labelFillWithZeroes );

    // Init first register
    vmovups( res[0], ptr[regFreeTermPtr] );

    for( int c = 0; c < StepSize; c++ ) {
        for( int r = 1; r < stepCount; r++ ) {
            vmovaps(res[r * StepSize + c], res[c]);
        }

        if( c != ( StepSize - 1 ) ) {
            vmovaps( res[c + 1], res[c] );
            rotateLeft2( res[c + 1], tempRes[0]  );
        }
    }

    jmp( labelEnd, T_NEAR );

    L( labelFillWithZeroes );
    // Init with zeroes
    for( int i = 0; i < stepCount * StepSize; i++ ) {
        vxorps( *res, *res, *res );
        res++;
    }
    L( labelEnd );
}

template<>
inline void CBlobConvolution<6>::CJitConvolution::fillBatchProcessingKernel( CBlobConvolution<6>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int KernelHeight = useNarrowProcessing ? NarrowBatchKernelHeight : WideBatchKernelHeight;
    const int KernelWidth = useNarrowProcessing ? NarrowBatchKernelWidth : WideBatchKernelWidth;
    const int StepCount = 3;
    const int StepSize = 3;

    Ymm res[3][3] = { { ymm0, ymm1, ymm2 }, { ymm3, ymm4, ymm5 }, { ymm6, ymm7, ymm8 } };
    Ymm f[1] = { ymm9 };
    Ymm s[3] = { ymm10, ymm11, ymm12 };
    Ymm st[3] = { ymm13, ymm14, ymm15 };
    Ymm temp[1] = { ymm15 };

    const size_t srcNarrowStep = useNarrowProcessing ? bc.SrcYStep : 4 * bc.SrcXStep;
    const size_t resNarrowStep = useNarrowProcessing ? bc.ResLineStride : 24;

    initProcessingMainLoop( bc, &res[0][0], &temp[0], StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex,
            useNarrowProcessing, resNarrowStep );

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

    // We will process pixels in three steps
    //           Step 1    Step 2      Step 3
    // Source: 0 0 0 1 - 1 1 2  2  - 2  3  3  3
    // Source: 4 4 4 5 - 5 5 6  6  - 6  7  7  7
    // Source: 8 8 8 9 - 9 9 10 10 - 10 11 11 11
    // Filter: 0 1 2 0 - 1 2 0  1  - 2  0  1  2
    // Three groups of source windows are sequenced in wide case or are placed one below another in narrow case.
    // load: 0 1 2 0
    vmovups( f[0], ptr[regTempFltPtr] );

    // load: 0 0 0 0
    vbroadcastss( s[0], ptr[regTempSrcPtr] );
    // load: 1 1 1 1
    vbroadcastss( st[0], ptr[regTempSrcPtr + bc.SrcXStep * sizeof( float )] );
    // load: 4 4 4 4
    vbroadcastss( s[1], ptr[regTempSrcPtr + srcNarrowStep * sizeof( float )] );
    // load: 5 5 5 5
    vbroadcastss( st[1], ptr[regTempSrcPtr + ( srcNarrowStep + bc.SrcXStep ) * sizeof( float )] );
    // load: 8 8 8 8
    vbroadcastss( s[2], ptr[regTempSrcPtr + 2 * srcNarrowStep * sizeof( float )] );
    // load: 9 9 9 9
    vbroadcastss( st[2], ptr[regTempSrcPtr + ( 2 * srcNarrowStep + bc.SrcXStep ) * sizeof( float )] );

    // blend( 0 0 0 0, 1 1 1 1 ) -> 0 0 0 1
    vblendps( s[0], s[0], st[0], 0xc0 );
    // blend( 4 4 4 4, 5 5 5 5 ) -> 4 4 4 5
    vblendps( s[1], s[1], st[1], 0xc0 );
    // blend( 8 8 8 8, 9 9 9 9 ) -> 8 8 8 9
    vblendps( s[2], s[2], st[2], 0xc0 );
    vfmadd231ps( res[0][0], f[0], s[0] );
    vfmadd231ps( res[1][0], f[0], s[1] );
    vfmadd231ps( res[2][0], f[0], s[2] );
    // 0 1 2 0 -> 1 2 0 1 ( use s[2] as temp register )
    rotateLeft2( f[0], s[2] );

    // load: 2 2 2 2
    vbroadcastss( s[0], ptr[regTempSrcPtr + 2 * bc.SrcXStep * sizeof( float )] );
    // load: 6 6 6 6
    vbroadcastss( s[1], ptr[regTempSrcPtr + ( srcNarrowStep + 2 * bc.SrcXStep ) * sizeof( float )] );
    // load: 10 10 10 10
    vbroadcastss( s[2], ptr[regTempSrcPtr + ( 2 * srcNarrowStep + 2 * bc.SrcXStep ) * sizeof( float )] );

    // blend( 1 1 1 1, 2 2 2 2 ) -> 1 1 2 2
    vblendps( st[0], st[0], s[0], 0xf0 );
    // blend( 5 5 5 5, 6 6 6 6 ) -> 5 5 6 6
    vblendps( st[1], st[1], s[1], 0xf0 );
    // blend( 9 9 9 9, 10 10 10 10 ) -> 9 9 10 10
    vblendps( st[2], st[2], s[2], 0xf0 );
    vfmadd231ps( res[0][1], f[0], st[0] );
    vfmadd231ps( res[1][1], f[0], st[1] );
    vfmadd231ps( res[2][1], f[0], st[2] );
    // 1 2 0 1 -> 2 0 1 2 ( use st[2] as a temp register )
    rotateLeft2( f[0], st[2] );

    // load: 3 3 3 3
    vbroadcastss( st[0], ptr[regTempSrcPtr + 3 * bc.SrcXStep * sizeof( float )] );
    // load: 7 7 7 7
    vbroadcastss( st[1], ptr[regTempSrcPtr + ( srcNarrowStep + 3 * bc.SrcXStep ) * sizeof( float )] );
    // load: 11 11 11 11
    vbroadcastss( st[2], ptr[regTempSrcPtr + ( 2 * srcNarrowStep + 3 * bc.SrcXStep ) * sizeof( float )] );

    // blend( 3 3 3 3, 2 2 2 2 ) -> 2 3 3 3
    vblendps( st[0], st[0], s[0], 0x03 );
    // blend( 7 7 7 7, 6 6 6 6 ) -> 6 7 7 7
    vblendps( st[1], st[1], s[1], 0x03 );
    // blend( 11 11 11 11, 10 10 10 10 ) -> 10 11 11 11
    vblendps( st[2], st[2], s[2], 0x03 );
    vfmadd231ps( res[0][2], f[0], st[0] );
    vfmadd231ps( res[1][2], f[0], st[1] );
    vfmadd231ps( res[2][2], f[0], st[2] );

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
inline void CBlobConvolution<6>::CJitConvolution::fillSingleProcessingKernel( CBlobConvolution<6>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = useNarrowProcessing ? 3 : 1;
    const int StepSize = 1;

    Ymm res[3] = { ymm0,  ymm1, ymm2 };
    Ymm s[3] = { ymm3, ymm4, ymm5 };
    Ymm st[3] = { ymm6, ymm7, ymm8 };
    Ymm f = ymm9;
    Ymm temp[1] = { ymm15 };

    const size_t srcNarrowStep = bc.SrcYStep;
    const size_t resNarrowStep = bc.ResLineStride;

    initProcessingMainLoop( bc, &res[0], &temp[0], StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex,
            useNarrowProcessing, resNarrowStep );

    ////////////////////////////////////////////////////////////////////////////////////////////

    const int BatchStepSize = 4;
    // channelCount - number of channels for frocessing
    // isLast - true if it is last of channel chank (we can skip src and flt pointers incrementing )
    auto fillKernel = [&]( int channelCount, bool isLast ) {
        // stepCount <= 4
        switch( channelCount ) {
        case 4:
            vmovups( s[0].copyAndSetKind( Operand::XMM ), ptr[regTempSrcPtr] );
            if( useNarrowProcessing ) {
                vmovups( s[1].copyAndSetKind( Operand::XMM ), ptr[regTempSrcPtr + srcNarrowStep * sizeof( float )] );
                vmovups( s[2].copyAndSetKind( Operand::XMM ), ptr[regTempSrcPtr + srcNarrowStep * ( 2 * sizeof( float ) )] );
            }
            break;
        default:
            // Create bitmask
            vxorps( st[0], st[0], st[0] );
            vpcmpeqd( st[1], st[1], st[1] );
            vblendps( st[1], st[0], st[1], 0xff >> ( 8 - channelCount ) );
            vmaskmovps( s[0], st[1], ptr[regTempSrcPtr] );
            if( useNarrowProcessing ) {
                vmaskmovps( s[1], st[1], ptr[regTempSrcPtr + srcNarrowStep * sizeof( float )] );
                vmaskmovps( s[2], st[1], ptr[regTempSrcPtr + srcNarrowStep * ( 2 * sizeof( float ) )] );
            }
        }
        for( int i = 0; i < channelCount; i++ ) {
            unsigned int mask = i * 0x55;
            vpermilps( st[0].copyAndSetKind( Operand::XMM ), s[0], mask );
            if( useNarrowProcessing ) {
                vpermilps( st[1].copyAndSetKind( Operand::XMM ), s[1], mask );
                vpermilps( st[2].copyAndSetKind( Operand::XMM ), s[2], mask );
            }
            vinsertf128( st[0], st[0], st[0].copyAndSetKind( Operand::XMM ), 1);
            if( useNarrowProcessing ) {
                vinsertf128( st[1], st[1], st[1].copyAndSetKind( Operand::XMM ), 1);
                vinsertf128( st[2], st[2], st[2].copyAndSetKind( Operand::XMM ), 1);
            }

            vmovups( f, ptr[regTempFltPtr + ( StepSize * i + 0 ) * SizeOfYmm] );

            vfmadd231ps( res[0], f, st[0] );
            if( useNarrowProcessing ) {
                vfmadd231ps( res[1], f, st[1] );
                vfmadd231ps( res[2], f, st[2] );
            }

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
