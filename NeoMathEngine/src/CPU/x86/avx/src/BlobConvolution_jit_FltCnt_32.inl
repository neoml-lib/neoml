/* Copyright � 2017-2020 ABBYY Production LLC

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
inline void CBlobConvolution<32>::CCode::fillBatchProcessingKernel( CBlobConvolution<32>& bc, bool useNarrowProcessing, int windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = 2;
    const int StepSize = 4;

	Ymm res[StepCount][StepSize] = { { ymm0, ymm1, ymm2, ymm3 }, { ymm4, ymm5, ymm6, ymm7 } };
	Ymm st[2] = { ymm8, ymm9 };
	Ymm f[4] = { ymm10, ymm11, ymm12, ymm13 };

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
	// Load one channel from one pixels in sequenced windows and fill one ymm register with its value.
	vbroadcastss( st[0], ptr[regTempSrcPtr] );
	vbroadcastss( st[1], ptr[regTempSrcPtr + bc.SrcXStep * sizeof( float )] );
	// Load one channel for the same pixel as in source for all filters.
	vmovups( f[0], ptr[regTempFltPtr] );
	vmovups( f[1], ptr[regTempFltPtr + SizeOfYmm] );
	vmovups( f[2], ptr[regTempFltPtr + 2 * SizeOfYmm] );
	vmovups( f[3], ptr[regTempFltPtr + 3 * SizeOfYmm] );
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
inline void CBlobConvolution<32>::CCode::fillSingleProcessingKernel( CBlobConvolution<32>& bc, bool useNarrowProcessing, int windowIndex )
{
    using namespace Xbyak;

    Label labelFillProcessingKernelEnd;
    Label labelProcessingKernel, labelProcessingKernelStart, labelProcessingKernelEnd;
    const int StepCount = 1;
    const int StepSize = 4;

	Ymm res[StepSize] = { ymm0, ymm1, ymm2, ymm3 };
	Xmm s = xmm4;
	Ymm st0 = ymm5;
	Xmm st0_toXmm = xmm5;
	Ymm f[StepSize] = { ymm6, ymm7, ymm8, ymm9 };

	initProcessingMainLoop( bc, &res[0], 0, StepCount, StepSize, labelProcessingKernel, labelFillProcessingKernelEnd,  windowIndex );

	////////////////////////////////////////////////////////////////////////////////////////////

	const int BatchStepSize = 4;
	// channelCount - number of channels for frocessing
	// isLast - true if it is last of channel chank (we can skip src and flt pointers incrementing )
	auto fillKernel = [&]( int channelCount, bool isLast ) {
		// stepCount <= 4
		movups( s, ptr[regTempSrcPtr] );
		for( int i = 0; i < channelCount; i++ ) {
			unsigned int mask = i * 0x55;
			vpermilps( st0_toXmm, s, mask );
			vinsertf128( st0, st0, st0_toXmm, 1);

			vmovups( f[0], ptr[regTempFltPtr + ( StepSize * i + 0 ) * SizeOfYmm] );
			vmovups( f[1], ptr[regTempFltPtr + ( StepSize * i + 1 ) * SizeOfYmm] );
			vmovups( f[2], ptr[regTempFltPtr + ( StepSize * i + 2 ) * SizeOfYmm] );
			vmovups( f[3], ptr[regTempFltPtr + ( StepSize * i + 3 ) * SizeOfYmm] );

			vfmadd231ps( res[0], f[0], st0 );
			vfmadd231ps( res[1], f[1], st0 );
			vfmadd231ps( res[2], f[2], st0 );
			vfmadd231ps( res[3], f[3], st0 );

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
