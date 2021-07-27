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
inline CBlobConvolution<32>::CSize CBlobConvolution<32>::getWideBatchProcessSize()
{
	return { 1, 2 };
}

template<>
inline void CBlobConvolution<32>::CCode::fillBatchProcessingKernel( CBlobConvolution<32>& bc, bool useNarrowProcessing, int windowIndex,
                                                                    Xbyak::Reg64 regSrcPtr, Xbyak::Reg64 regFltPtr, Xbyak::Reg64 regResPtr )
{
    using namespace Xbyak;

    Reg64 regTempSrcPtr =  util::r10;
    Reg64 regTempFltPtr =  util::r11;
    Reg64 regChCnt =  util::rax;

    Label labelFillBatchProcessingKernelEnd;
    Label labelBatchProcessingKernel, labelBatchProcessingKernelStart, labelBatchProcessingKernelEnd;

	const int rowNum = 2;
	const int colNum = 4;
	Ymm res[2][4] = { { ymm0, ymm1, ymm2, ymm3 }, { ymm4, ymm5, ymm6, ymm7 } };
	Ymm st[2] = { ymm8, ymm9 };
	Ymm f[4] = { ymm10, ymm11, ymm12, ymm13 };

	initResRegs( &res[0][0], bc.freeTerm, rowNum, colNum );

	auto srcIt = bc.SrcPixelsOffset[windowIndex].cbegin();
	auto fltIt = bc.FltPixelsOffset[windowIndex].cbegin();

	for( ; srcIt != bc.SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
		lea( regTempSrcPtr, ptr[regSrcPtr + *srcIt * sizeof( float )] );
		lea( regTempFltPtr, ptr[regFltPtr + *fltIt * sizeof( float )] );
		call( labelBatchProcessingKernel );
	}

	flushResRegs( &res[0][0], regResPtr, rowNum, colNum );


	// return from function
	jmp( labelFillBatchProcessingKernelEnd, T_NEAR );

	////////////////////////////////////////////////////////////////////////////////////////////
	// Batch process kernell function
	L( labelBatchProcessingKernel );
	// for( int c = 0; c < ChCnt; c++ ) {
	if( bc.ChCnt > 1 ) {
		xor_( regChCnt, regChCnt );
		L( labelBatchProcessingKernelStart );
		cmp( regChCnt, bc.ChCnt );
		je( labelBatchProcessingKernelEnd, T_NEAR );
	}
	// Load one channel from one pixels in sequenced windows and fill one ymm register with its value.
	vbroadcastss( st[0], ptr[regTempSrcPtr] );
	vbroadcastss( st[1], ptr[regTempSrcPtr + bc.SrcXStep * sizeof( float )] );
	// Load one channel for the same pixel as in source for all filters.
	vmovups( f[0], ptr[regTempFltPtr] );
	vmovups( f[1], ptr[regTempFltPtr + 8* sizeof( float )] );
	vmovups( f[2], ptr[regTempFltPtr + 16* sizeof( float )] );
	vmovups( f[3], ptr[regTempFltPtr + 24* sizeof( float )] );
	// Take result for current pixels in three sequenced windows.
	// Multiply one chennel for all filters ( for ONE src/flt pixels )
	vfmadd231ps( res[0][0], f[0], st[0] );
	vfmadd231ps( res[0][1], f[1], st[0] );
	vfmadd231ps( res[0][2], f[2], st[0] );
	vfmadd231ps( res[0][3], f[3], st[0] );
	vfmadd231ps( res[1][0], f[0], st[1] );
	vfmadd231ps( res[1][1], f[1], st[1] );
	vfmadd231ps( res[1][2], f[2], st[1] );
	vfmadd231ps( res[1][3], f[3], st[1] );

	// Go to next chennel in filter and source
	add( regTempFltPtr, FltCntM8 * sizeof( float ) );
	add( regTempSrcPtr, sizeof( float ) );
	inc( regChCnt );

	if( bc.ChCnt > 1 ) {
		jmp( labelBatchProcessingKernelStart, T_NEAR );
		// }
		L( labelBatchProcessingKernelEnd );
	}
	ret();
	// End of code
	L( labelFillBatchProcessingKernelEnd );
}

template<>
inline void CBlobConvolution<32>::CCode::fillSingleProcessingKernel( CBlobConvolution<32>& bc, bool useNarrowProcessing, int windowIndex,
																	 Xbyak::Reg64 regSrcPtr, Xbyak::Reg64 regFltPtr, Xbyak::Reg64 regResPtr )
{
	using namespace Xbyak;

    Reg64 regTempSrcPtr =  util::r10;
    Reg64 regTempFltPtr =  util::r11;
    Reg64 regChCnt =  util::rax;

    Label labelFillSingleProcessingKernelEnd;
    Label labelSingleProcessingKernel, labelSingleProcessingKernelStart, labelSingleProcessingKernelEnd;

	const int rowNum = 1;
	const int colNum = 4;
	Ymm res[4] = { ymm0, ymm1, ymm2, ymm3 };
	Xmm s = xmm4;
	Ymm st0 = ymm5;
	Xmm st0_toXmm = xmm5;
	Ymm f[4] = { ymm6, ymm7, ymm8, ymm9 };

	initResRegs( &res[0], bc.freeTerm, rowNum, colNum );

	auto srcIt = bc.SrcPixelsOffset[windowIndex].cbegin();
	auto fltIt = bc.FltPixelsOffset[windowIndex].cbegin();

	for( ; srcIt != bc.SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
		lea( regTempSrcPtr, ptr[regSrcPtr + *srcIt * sizeof( float )] );
		lea( regTempFltPtr, ptr[regFltPtr + *fltIt * sizeof( float )] );
		call( labelSingleProcessingKernel );
	}

	flushResRegs( &res[0], regResPtr, rowNum, colNum );


	// return from function
	jmp( labelFillSingleProcessingKernelEnd, T_NEAR );

	////////////////////////////////////////////////////////////////////////////////////////////
	// Single process kernell function
	L( labelSingleProcessingKernel );
	// for( ; c >= 4; c -= 4 ) {
	if( bc.ChCnt >= 4 ) {
		xor_( regChCnt, regChCnt );
		L( labelSingleProcessingKernelStart );
		cmp( regChCnt, bc.ChCnt / 4 );
		je( labelSingleProcessingKernelEnd, T_NEAR );

		movups( s, ptr[regTempSrcPtr] );
		for( unsigned int mask = 0x00; mask <= 0xff; mask += 0x55 ) {
			vpermilps( st0_toXmm, s, mask );
			vinsertf128( st0, st0, st0_toXmm, 1);

			vmovups( f[0], ptr[regTempFltPtr] );
			vmovups( f[1], ptr[regTempFltPtr + 8* sizeof( float )] );
			vmovups( f[2], ptr[regTempFltPtr + 16* sizeof( float )] );
			vmovups( f[3], ptr[regTempFltPtr + 24* sizeof( float )] );

			vfmadd231ps( res[0], f[0], st0 );
			vfmadd231ps( res[1], f[1], st0 );
			vfmadd231ps( res[2], f[2], st0 );
			vfmadd231ps( res[3], f[3], st0 );

			add( regTempFltPtr, FltCntM8 * sizeof( float ) );
		}

		add( regTempSrcPtr, 4 * sizeof( float ) );
		inc( regChCnt );

		jmp( labelSingleProcessingKernelStart, T_NEAR );
		// }
		L( labelSingleProcessingKernelEnd );
	}

	int chCntRemained = bc.ChCnt % 4;
	if( chCntRemained  ) {
		movups( s, ptr[regTempSrcPtr] );
		for( unsigned int mask = 0x00; mask < chCntRemained * 0x55; mask += 0x55 ) {
			vpermilps( st0_toXmm, s, mask );
			vinsertf128( st0, st0, st0_toXmm, 1);

			vmovups( f[0], ptr[regTempFltPtr] );
			vmovups( f[1], ptr[regTempFltPtr + 8* sizeof( float )] );
			vmovups( f[2], ptr[regTempFltPtr + 16* sizeof( float )] );
			vmovups( f[3], ptr[regTempFltPtr + 24* sizeof( float )] );

			vfmadd231ps( res[0], f[0], st0 );
			vfmadd231ps( res[1], f[1], st0 );
			vfmadd231ps( res[2], f[2], st0 );
			vfmadd231ps( res[3], f[3], st0 );

			add( regTempFltPtr, FltCntM8 * sizeof( float ) );
		}
	}
	ret();
	// End of code
	L( labelFillSingleProcessingKernelEnd );
}

template<>
inline void CBlobConvolution<32>::batchProcessChannels( const float* srcPtr, const float* fltPtr,
	__m256& r00, __m256& r01, __m256& r02, __m256& r03,
	__m256& r10, __m256& r11, __m256& r12, __m256& r13 )
{
	// Process three result pixels per time.
	auto processNext = [&]()
	{
		// Load one channel from one pixels in sequenced windows and fill one ymm register with its value.
		__m256 st0 = _mm256_broadcast_ss( srcPtr );
		__m256 st1 = _mm256_broadcast_ss( srcPtr + SrcXStep );
		// Load one channel for the same pixel as in source for all filters.
		__m256 f0 = _mm256_loadu_ps( fltPtr );
		__m256 f1 = _mm256_loadu_ps( fltPtr + 8 );
		__m256 f2 = _mm256_loadu_ps( fltPtr + 16 );
		__m256 f3 = _mm256_loadu_ps( fltPtr + 24 );
		// Take result for current pixels in three sequenced windows.
		// Multiply one chennel for all filters ( for ONE src/flt pixels )
		r00 = _mm256_fmadd_ps( f0, st0, r00 );
		r01 = _mm256_fmadd_ps( f1, st0, r01 );
		r02 = _mm256_fmadd_ps( f2, st0, r02 );
		r03 = _mm256_fmadd_ps( f3, st0, r03 );
		r10 = _mm256_fmadd_ps( f0, st1, r10 );
		r11 = _mm256_fmadd_ps( f1, st1, r11 );
		r12 = _mm256_fmadd_ps( f2, st1, r12 );
		r13 = _mm256_fmadd_ps( f3, st1, r13 );

		// Go to next chennel in filter and source
		fltPtr += FltCntM8;
		srcPtr++;
	};

	for( int c = 0; c < ChCnt; c++ ) {
		processNext();
	}
}

template<>
inline void CBlobConvolution<32>::singleProcessChannels( const float* srcPtr, const float* fltPtr,
	__m256& r0, __m256& r1, __m256& r2, __m256& r3 )
{
	// Process one result pixels per time.
	auto processNext = [&]( __m128 st0_m128 )
	{
		__m256 st0 = _mm256_set_m128( st0_m128, st0_m128 );

		__m256 f0 = _mm256_loadu_ps( fltPtr );
		__m256 f1 = _mm256_loadu_ps( fltPtr + 8 );
		__m256 f2 = _mm256_loadu_ps( fltPtr + 16 );
		__m256 f3 = _mm256_loadu_ps( fltPtr + 24 );

		r0 = _mm256_fmadd_ps( f0, st0, r0 );
		r1 = _mm256_fmadd_ps( f1, st0, r1 );
		r2 = _mm256_fmadd_ps( f2, st0, r2 );
		r3 = _mm256_fmadd_ps( f3, st0, r3 );

		fltPtr += FltCntM8;
	};

	int c = ChCnt;
	for( ; c >= 4; c -= 4 ) {
		__m128 s = _mm_loadu_ps( srcPtr );
		// Load four channels and fill __m256 register four times with each of channel.
		processNext( _mm_permute_ps( s, 0x00 ) );
		processNext( _mm_permute_ps( s, 0x55 ) );
		processNext( _mm_permute_ps( s, 0xaa ) );
		processNext( _mm_permute_ps( s, 0xff ) );
		srcPtr += 4;
	}
	if( c != 0 ) {
		// Process remaining channels
		__m128 s = _mm_loadu_ps( srcPtr );
		processNext( _mm_permute_ps( s, 0x00 ) );
		if( c > 1 ) {
			processNext( _mm_permute_ps( s, 0x55 ) );
		}
		if( c > 2 ) {
			processNext( _mm_permute_ps( s, 0xaa ) );
		}
	}
}

template<>
inline void CBlobConvolution<32>::batchProcess( const float* srcPtr, float* resPtr, size_t windowIndex, bool /*useNarrowProcessing*/ )
{
	const __m256 ft0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	const __m256 ft1 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 8 ) : _mm256_setzero_ps();
	const __m256 ft2 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 16 ) : _mm256_setzero_ps();
	const __m256 ft3 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 24 ) : _mm256_setzero_ps();

	// Initialize result pixels with freeterm.
	__m256 r00 = ft0;
	__m256 r01 = ft1;
	__m256 r02 = ft2;
	__m256 r03 = ft3;
	__m256 r10 = ft0;
	__m256 r11 = ft1;
	__m256 r12 = ft2;
	__m256 r13 = ft3;

	auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
	auto fltIt = FltPixelsOffset[windowIndex].cbegin();
	for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
		batchProcessChannels( srcPtr + *srcIt, flt + *fltIt, r00, r01, r02, r03, r10, r11, r12, r13 );
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_storeu_ps( resPtr, r00 );
	_mm256_storeu_ps( resPtr + 8, r01 );
	_mm256_storeu_ps( resPtr + 16, r02 );
	_mm256_storeu_ps( resPtr + 24, r03 );
	_mm256_storeu_ps( resPtr + 32, r10 );
	_mm256_storeu_ps( resPtr + 40, r11 );
	_mm256_storeu_ps( resPtr + 48, r12 );
	_mm256_storeu_ps( resPtr + 56, r13 );
}

template<>
inline void CBlobConvolution<32>::singleProcess( const float* srcPtr, float* resPtr, size_t windowIndex )
{
	// Initialize result pixels with freeterm.
	__m256 r0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	__m256 r1 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 8 ) : _mm256_setzero_ps();
	__m256 r2 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 16 ) : _mm256_setzero_ps();
	__m256 r3 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 24 ) : _mm256_setzero_ps();

	auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
	auto fltIt = FltPixelsOffset[windowIndex].cbegin();
	for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
		singleProcessChannels( srcPtr + *srcIt, flt + *fltIt, r0, r1, r2, r3 );
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_storeu_ps( resPtr, r0 );
	_mm256_storeu_ps( resPtr + 8, r1 );
	_mm256_storeu_ps( resPtr + 16, r2 );
	_mm256_storeu_ps( resPtr + 24, r3 );
}

} // namespace NeoML
