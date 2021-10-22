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
// Channel count: 24

template<>
const int CBlobConvolution<24>::WideBatchKernelHeight = 1;

template<>
const int CBlobConvolution<24>::WideBatchKernelWidth = 3;

template<>
inline void CBlobConvolution<24>::CJitConvolution::fillBatchProcessingKernel( CBlobConvolution<24>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = 3;
    const int StepSize = 3;

    Ymm res[3][3] = { { ymm0, ymm1, ymm2 }, { ymm3, ymm4, ymm5 }, { ymm6, ymm7, ymm8 } };
    Ymm st[3] = { ymm9, ymm10, ymm11 };
    Ymm f[3] = { ymm12, ymm13, ymm14 };

    initProcessingMainLoop( bc, &res[0][0], 0, StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex );

    ////////////////////////////////////////////////////////////////////////////////////////////
    auto fillKernel = [&]( int channelCount ) {
        for( int i = 0; i < channelCount; i++ ) {
            size_t fltOffset = i * FltCntM8 * sizeof( float );
            size_t srcOffset = i * sizeof( float );
            // Load one channel from one pixels in sequenced windows and fill one ymm register with its value.
            vbroadcastss( st[0], ptr[regTempSrcPtr + srcOffset] );
            vbroadcastss( st[1], ptr[regTempSrcPtr + srcOffset + bc.SrcXStep * sizeof( float )] );
            vbroadcastss( st[2], ptr[regTempSrcPtr + srcOffset + 2 * bc.SrcXStep * sizeof( float )] );
            // Load one channel for the same pixel as in source for all filters.
            vmovups( f[0], ptr[regTempFltPtr + fltOffset] );
            vmovups( f[1], ptr[regTempFltPtr + fltOffset  + SizeOfYmm] );
            vmovups( f[2], ptr[regTempFltPtr + fltOffset + 2 * SizeOfYmm] );
            // Take result for current pixels in three sequenced windows.
            // Multiply one channel for all filters ( for ONE src/flt pixels )
            vfmadd231ps( res[0][0], f[0], st[0] );
            vfmadd231ps( res[0][1], f[1], st[0] );
            vfmadd231ps( res[0][2], f[2], st[0] );
            vfmadd231ps( res[1][0], f[0], st[1] );
            vfmadd231ps( res[1][1], f[1], st[1] );
            vfmadd231ps( res[1][2], f[2], st[1] );
            vfmadd231ps( res[2][0], f[0], st[2] );
            vfmadd231ps( res[2][1], f[1], st[2] );
            vfmadd231ps( res[2][2], f[2], st[2] );
        }

        // Go to next channel in filter and source
        add( regTempFltPtr, channelCount * FltCntM8 * sizeof( float ) );
        add( regTempSrcPtr, channelCount * sizeof( float ) );
    };

    const int BatchStepSize = 24;
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
inline void CBlobConvolution<24>::CJitConvolution::fillSingleProcessingKernel( CBlobConvolution<24>& bc, bool useNarrowProcessing, size_t windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = 1;
    const int StepSize = 3;

    Ymm res[3] = { ymm0, ymm1, ymm2 };
    Ymm tempRes[2][3] = { { ymm0, ymm1, ymm2 }, { ymm3, ymm4, ymm5 } };
    Ymm st[2] = { ymm6, ymm7 };
    Ymm s = ymm8;
    Ymm f[2][3] = { { ymm9, ymm10, ymm11 }, { ymm12, ymm13, ymm14 } };
    Ymm regShiftIndex = ymm15;

    // Clear temp regs[1..3]
    vxorps( tempRes[1][0], tempRes[1][0], tempRes[1][0] );
    vxorps( tempRes[1][1], tempRes[1][1], tempRes[1][1] );
    vxorps( tempRes[1][2], tempRes[1][2], tempRes[1][2] );

    std::function<void()> mergeResRegs( [&]() {
        vaddps( res[0], tempRes[0][0], tempRes[1][0] );
        vaddps( res[1], tempRes[0][1], tempRes[1][1] );
        vaddps( res[2], tempRes[0][2], tempRes[1][2] );
        } );

    Label labelShiftMask;
    vmovdqa( regShiftIndex, ptr[rip + labelShiftMask] );

    initProcessingMainLoop( bc, &res[0], 0, StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex,
        false, 0, &mergeResRegs );

    ////////////////////////////////////////////////////////////////////////////////////////////

    // channelCount - number of channels for frocessing
    // isLast - true if it is last of channel chank (we can skip src and flt pointers incrementing )
    auto fillKernel = [&]( int channelCount ) {
		// Load source to st[0]
		switch( channelCount ) {
		case 8:
			vmovups( s, ptr[regTempSrcPtr] );
			break;
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

        int ch = 0;

		auto fillKernelInternal = [&]( int channelCountInternal, bool isLast ) {
			// First source and filter
			vbroadcastss( st[0], s.copyAndSetKind( Operand::XMM ) );
			if( channelCountInternal == 2 ) {
				// shift right by one float
				vpermps( s, regShiftIndex, s );
			}
			vmovups( f[0][0], ptr[regTempFltPtr + ( StepSize * ch + 0 ) * SizeOfYmm] );
			vmovups( f[0][1], ptr[regTempFltPtr + ( StepSize * ch + 1 ) * SizeOfYmm] );
			vmovups( f[0][2], ptr[regTempFltPtr + ( StepSize * ch + 2 ) * SizeOfYmm] );
			ch++;

			if( channelCountInternal == 2 ) {
				// Second source and filter
				vbroadcastss( st[1], s.copyAndSetKind( Operand::XMM ) );
				if( !isLast ) {
					// shift right by one float
					vpermps( s, regShiftIndex, s );
				}
				vmovups( f[1][0], ptr[regTempFltPtr + ( StepSize * ch + 0 ) * SizeOfYmm] );
				vmovups( f[1][1], ptr[regTempFltPtr + ( StepSize * ch + 1 ) * SizeOfYmm] );
				vmovups( f[1][2], ptr[regTempFltPtr + ( StepSize * ch + 2 ) * SizeOfYmm] );
				ch++;
			}

			vfmadd231ps( tempRes[0][0], f[0][0], st[0] );
			vfmadd231ps( tempRes[0][1], f[0][1], st[0] );
			vfmadd231ps( tempRes[0][2], f[0][2], st[0] );
			if( channelCountInternal == 2 ) {
				vfmadd231ps( tempRes[1][0], f[1][0], st[1] );
				vfmadd231ps( tempRes[1][1], f[1][1], st[1] );
				vfmadd231ps( tempRes[1][2], f[1][2], st[1] );
			}
		};

		int batchStepCount = channelCount / 2;
		int remainedStepCount = channelCount % 2;
		for( int s = 0; s < batchStepCount; s++ ) {
			bool isLast = remainedStepCount == 0 && s == ( batchStepCount - 1 );
			fillKernelInternal( 2, isLast );
		}
		if( remainedStepCount != 0 ) {
			fillKernelInternal( 1, true );
		}

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

    align( 32 );
    L( labelShiftMask );
    // Index [1, 2, 3, 4, 5, 6, 7, 0] will circulary shift Ymm to the right
    for( int i = 1; i < 8; i++ ) {
        dd( i );
    }
    dd( 0 );

    // End of code
    L( labelFillProcessingKernelEnd );
}

} // namespace NeoML
