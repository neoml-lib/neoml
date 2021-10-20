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
// Channel count: 3

template<>
const int CBlobConvolution<3>::NarrowBatchKernelHeight = 3;

template<>
const int CBlobConvolution<3>::NarrowBatchKernelWidth = 8;

template<>
const int CBlobConvolution<3>::WideBatchKernelHeight = 1;

template<>
const int CBlobConvolution<3>::WideBatchKernelWidth = 24;

template<>
inline void CBlobConvolution<3>::CJitConvolution::initResRegs( Xbyak::Ymm* res, Xbyak::Ymm* tempRes, size_t stepCount, size_t StepSize )
{
	using namespace Xbyak;

	Label labelFillWithZeroes, labelEnd;
	test( regFreeTermPtr, regFreeTermPtr );
	jz( labelFillWithZeroes );

	// Init first register
	if( StepSize == 1 ) {
		vmovups( res[0].copyAndSetKind( Operand::XMM ), ptr[regFreeTermPtr] );
	}  else {
		vmovups( res[0], ptr[regFreeTermPtr] );
	}

	for( int c = 0; c < StepSize; c++ ) {
		for( int r = 1; r < stepCount; r++ ) {
			vmovaps( res[r * StepSize + c], res[c] );
		}

		if( c != ( StepSize - 1 ) ) {
			rotateRight2( res[c + 1], res[c] );
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
inline void CBlobConvolution<3>::CJitConvolution::fillBatchProcessingKernel( CBlobConvolution<3>& bc, bool useNarrowProcessing, size_t windowIndex )
{
	using namespace Xbyak;

	Label labelFillProcessingKernelEnd;
	Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
	const int StepCount = 3;
	const int StepSize = 3;

	Ymm res[3][3] = { { ymm0, ymm1, ymm2 }, { ymm3, ymm4, ymm5 }, { ymm6, ymm7, ymm8 } };
	Ymm f = ymm9;
	Ymm t[3] = { ymm10, ymm11, ymm12 };
	Ymm s[3] = { ymm13, ymm14, ymm15 };

	const size_t srcNarrowStep = useNarrowProcessing ? bc.SrcYStep : 8 * bc.SrcXStep;
	const size_t resNarrowStep = useNarrowProcessing ? bc.ResLineStride : 24;

	initProcessingMainLoop( bc, &res[0][0], 0, StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd, windowIndex,
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

	vmovups( f, ptr[regTempFltPtr] );

	for( int step = 0; step < 3; step++ ) {
		size_t srcOffset = srcNarrowStep * step * sizeof( float );

		//               s[0]               s[1]              s[2]
		// source 0: 0 0 0 1 1 1 2 2   2 3 3 3 4 4 4 5   5 5 6 6 6 7 7 7
		// s[0]: 0 0 0 0 0 0 0 0
		vbroadcastss( s[0], ptr[regTempSrcPtr + srcOffset] );
		// t[0]: 1 1 1 1 1 1 1 1
		vbroadcastss( t[0], ptr[regTempSrcPtr + srcOffset + bc.SrcXStep * sizeof( float )] );
		// t[1]: 2 2 2 2 2 2 2 2
		vbroadcastss( t[1], ptr[regTempSrcPtr + srcOffset + 2 * bc.SrcXStep * sizeof( float )] );
		// s[1]: 3 3 3 3 3 3 3 3
		vbroadcastss( s[1], ptr[regTempSrcPtr + srcOffset + 3 * bc.SrcXStep * sizeof( float )] );
		// t[2]: 4 4 4 4 4 4 4 4
		vbroadcastss( t[2], ptr[regTempSrcPtr + srcOffset + 4 * bc.SrcXStep * sizeof( float )] );
		// s[2]: 5 5 5 5 5 5 5 5
		vbroadcastss( s[2], ptr[regTempSrcPtr + srcOffset + 5 * bc.SrcXStep * sizeof( float )] );

		// s[0] : 0 0 0 1 1 1 1 1
		vblendps( s[0], s[0], t[0], 0xf8 );
		// s[1] : 3 3 3 3 4 4 4 4
		vblendps( s[1], s[1], t[2], 0xf0 );
		// t[0]: 6 6 6 6 6 6 6 6
		vbroadcastss( t[0], ptr[regTempSrcPtr + srcOffset + 6 * bc.SrcXStep * sizeof( float )] );
		// t[2]: 7 7 7 7 7 7 7 7
		vbroadcastss( t[2], ptr[regTempSrcPtr + srcOffset + 7 * bc.SrcXStep * sizeof( float )] );
		// s[0] : 0 0 0 1 1 1 2 2 (READY)
		vblendps( s[0], s[0], t[1], 0xc0 );
		// s[1] : 2 3 3 3 4 4 4 4
		vblendps( s[1], s[1], t[1], 0x01 );
		// t[1]: shifted filter for s[1]
		rotateRight2( t[1], f );
		// s[1] : 2 3 3 3 4 4 4 5 (READY)
		vblendps( s[1], s[1], s[2], 0x80 );
		// s[2] " 5 5 6 6 6 6 6 6
		vblendps( s[2], s[2], t[0], 0xfc );
		// t[0]: shifted filter for s[2]
		rotateRight2( t[0], t[1] );
		// s[2] " 5 5 6 6 6 7 7 7
		vblendps( s[2], s[2], t[2], 0xe0 );

		vfmadd231ps( res[step][0], f, s[0] );
		vfmadd231ps( res[step][1], t[1], s[1] );
		vfmadd231ps( res[step][2], t[0], s[2] );

	}
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
inline void CBlobConvolution<3>::CJitConvolution::fillSingleProcessingKernel( CBlobConvolution<3>& bc, bool useNarrowProcessing, size_t windowIndex )
{
	using namespace Xbyak;

	Label labelFillProcessingKernelEnd;
	Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
	const int StepCount = useNarrowProcessing ? 3 : 1;
	const int StepSize = 1;

	Ymm res[3] = { ymm0, ymm1, ymm2 };
	// We haven't enough ymm registers, therefore we will use only two temporary ones.
	Ymm tempRes[2] = { ymm14, ymm15 };
	Ymm s[3][2] = { { ymm3, ymm4 }, { ymm5, ymm6 }, { ymm7, ymm8 } };
	Ymm f[2] = { ymm9, ymm10 };
	Ymm t[3] = { ymm11, ymm12, ymm13 };

	const size_t srcNarrowStep = bc.SrcYStep;
	const size_t resNarrowStep = bc.ResLineStride;

	// Clear temp regs
	vxorps( tempRes[0], tempRes[0], tempRes[0] );
	vxorps( tempRes[1], tempRes[1], tempRes[1] );

	std::function<void()> mergeResRegs( [&]() {
		if( useNarrowProcessing ) {
			vaddps( res[1], res[1], tempRes[0] );
			vaddps( res[2], res[2], tempRes[1] );
			// First register is already accumulated
			// Append high half od res to low one.
			vextractf128( t[0].copyAndSetKind( Operand::XMM ), res[0], 1 );
			vextractf128( t[1].copyAndSetKind( Operand::XMM ), res[1], 1 );
			vextractf128( t[2].copyAndSetKind( Operand::XMM ), res[2], 1 );
			vaddps( res[0], res[0], t[0] );
			vaddps( res[1], res[1], t[1] );
			vaddps( res[2], res[2], t[2] );
		} else {
			vaddps( res[0], res[0], tempRes[0] );
			// Append high half od res to low one.
			vextractf128( t[0].copyAndSetKind( Operand::XMM ), res[0], 1 );
			vaddps( res[0], res[0], t[0] );
		}
		} );

	initProcessingMainLoop( bc, &res[0], 0, StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd, windowIndex,
		useNarrowProcessing, resNarrowStep, &mergeResRegs );

	////////////////////////////////////////////////////////////////////////////////////////////

	// channelCount - number of channels for processing
	// isLast - true if it is last of channel chank (we can skip src and flt pointers incrementing )
	auto fillKernelCh4 = [&]() {
		const int channelCount = 4;
		// Load filter
		vmovups( f[0], ptr[regTempFltPtr] );
		vmovups( f[1], ptr[regTempFltPtr + FltCntM8 * sizeof( float )] );
		vmovups( t[0], ptr[regTempFltPtr + 2 * FltCntM8 * sizeof( float )] );
		vmovups( t[1], ptr[regTempFltPtr + 3 * FltCntM8 * sizeof( float )] );

		// Load Source
		vmovups( s[0][0].copyAndSetKind( Operand::XMM ), ptr[regTempSrcPtr] );
		if( useNarrowProcessing ) {
			vmovups( s[1][0].copyAndSetKind( Operand::XMM ), ptr[regTempSrcPtr + srcNarrowStep * sizeof( float )] );
			vmovups( s[2][0].copyAndSetKind( Operand::XMM ), ptr[regTempSrcPtr + 2 * srcNarrowStep * sizeof( float )] );
		}

		add( regTempFltPtr, channelCount * FltCntM8 * sizeof( float ) );
		add( regTempSrcPtr, channelCount * sizeof( float ) );

		// Prepare filter
		// f0 :     f00 f01 f02 f00 f01 f02 f00 f01
		// t0 :     f20 f21 f22 f20 f21 f22 f20 f21
		// f0 res : f00 f01 f02 f00 f20 f21 f22 f20
		vinsertf128( f[0], f[0], t[0].copyAndSetKind( Operand::XMM ), 1 );
		vinsertf128( f[1], f[1], t[1].copyAndSetKind( Operand::XMM ), 1 );

		// Prepare source
		// s0 :     s00 s01 s02 s03   0   0   0   0
		// s0 res : s00 s01 s00 s01 s02 s03 s02 s03
		vpermpd( s[0][0], s[0][0], _MM_SHUFFLE( 1, 1, 0, 0 ) );
		if( useNarrowProcessing ) {
			vpermpd( s[1][0], s[1][0], _MM_SHUFFLE( 1, 1, 0, 0 ) );
			vpermpd( s[2][0], s[2][0], _MM_SHUFFLE( 1, 1, 0, 0 ) );
		}
		// s[0][0] :     s00 s01 s00 s01 s02 s03 s02 s03
		// s[0][1] res : s01 s01 s01 s01 s03 s03 s03 s03
		vshufps( s[0][1], s[0][0], s[0][0], 0xff );
		// s[0][0] :     s00 s01 s00 s01 s02 s03 s02 s03
		// s[0][1] res : s00 s00 s00 s00 s01 s01 s01 s01
		vshufps( s[0][0], s[0][0], s[0][0], 0x00 );
		if( useNarrowProcessing ) {
			vshufps( s[1][1], s[1][0], s[1][0], 0xff );
			vshufps( s[1][0], s[1][0], s[1][0], 0x00 );
			vshufps( s[2][1], s[2][0], s[2][0], 0xff );
			vshufps( s[2][0], s[2][0], s[2][0], 0x00 );
		}

		// Calculate res
		if( useNarrowProcessing ) {
			vfmadd231ps( res[0], s[0][0], f[0] );
			vmulps( t[2], s[0][1], f[1] );
			vfmadd231ps( res[1], s[1][0], f[0] );
			vfmadd231ps( tempRes[0], s[1][1], f[1] );
			vfmadd231ps( res[2], s[2][0], f[0] );
			vfmadd231ps( tempRes[1], s[2][1], f[1] );
			// Accumulate register r[0]. Other registers will be accumulated at the end.
			vaddps( res[0], res[0], t[2] );
		} else {
			vfmadd231ps( res[0], s[0][0], f[0] );
			vfmadd231ps( tempRes[0], s[0][1], f[1] );
		}
	};

	auto fillKernelCh = [&]( int channelCount ) {
		// Load filter
		if( channelCount == 1 ) {
			// High hals is zerroed
			vmovups( f[0].copyAndSetKind( Operand::XMM ), ptr[regTempFltPtr] );
		} else {
			vmovups( f[0], ptr[regTempFltPtr] );
			vmovups( f[1], ptr[regTempFltPtr + FltCntM8 * sizeof( float )] );
		}

		// Load Source
		if( channelCount == 1 ) {
			// Load one float
			vbroadcastss( s[0][0], ptr[regTempSrcPtr] );
			if( useNarrowProcessing ) {
				vbroadcastss( s[1][0], ptr[regTempSrcPtr + srcNarrowStep * sizeof( float )] );
				vbroadcastss( s[2][0], ptr[regTempSrcPtr + 2 * srcNarrowStep * sizeof( float )] );
			}
		} else {
			// Load two floats
			vbroadcastsd( s[0][0], ptr[regTempSrcPtr] );
			if( useNarrowProcessing ) {
				vbroadcastsd( s[1][0], ptr[regTempSrcPtr + srcNarrowStep * sizeof( float )] );
				vbroadcastsd( s[2][0], ptr[regTempSrcPtr + 2 * srcNarrowStep * sizeof( float )] );
			}
		}

		add( regTempFltPtr, channelCount * FltCntM8 * sizeof( float ) );
		add( regTempSrcPtr, channelCount * sizeof( float ) );

		// Prepare filter
		// f0 :     f00 f01 f02 f00 f01 f02 f00 f01
		// t0 :     f20 f21 f22 f20 f21 f22 f20 f21
		// f0 res : f00 f01 f02 f00 f20 f21 f22 f20
		if( channelCount == 2 ) {
			vinsertf128( f[0], f[0], f[1].copyAndSetKind( Operand::XMM ), 1 );
		}

		// Prepare source
		if( channelCount == 2 ) {
			// For channelCount == 1 all are already prepared.
			// s[0][0] :     s00 s01 s00 s01 s00 s01 s00 s01
			// s[0][0] res : s00 s00 s01 s01 s00 s00 s01 s01
			vunpcklps( s[0][0], s[0][0], s[0][0] );
			// s[0][0] :     s00 s00 s01 s01 s00 s00 s01 s01
			// s[0][0] res : s00 s00 s00 s00 s01 s01 s01 s01
			vshufpd( s[0][0], s[0][0], s[0][0], 0x0c );
			if( useNarrowProcessing ) {
				vunpckhps( s[1][0], s[1][0], s[1][0] );
				vshufpd( s[1][0], s[1][0], s[1][0], 0x0c );
				vunpckhps( s[2][0], s[2][0], s[2][0] );
				vshufpd( s[2][0], s[2][0], s[2][0], 0x0c );
			}
		}

		vfmadd231ps( res[0], s[0][0], f[0] );
		// Calculate res
		if( useNarrowProcessing ) {
			vfmadd231ps( res[1], s[1][0], f[0] );
			vfmadd231ps( res[2], s[2][0], f[0] );
		}
	};

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

		fillKernelCh4();

		if( batchStepCount > 1 ) {
			// } // for( regChCnt i = 0; regChCnt < batchStepCount; regChCnt++ ){}
			inc( regChCnt );
			jmp( labelProcessingKernelStart, T_NEAR );
		}

		L( labelProcessingKernelEnd );
	}

	if( remainedStepCount >= 2 ) {
		fillKernelCh( 2 );
		remainedStepCount -= 2;
	}

	if( remainedStepCount > 0 ) {
		fillKernelCh( 1 );
	}

	ret();
	// End of code
	L( labelFillProcessingKernelEnd );
}

} // namespace NeoML
