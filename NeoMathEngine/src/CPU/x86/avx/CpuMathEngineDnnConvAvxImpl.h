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

#if !defined( _mm256_set_m128 )
// This instruction is defined since 8 gcc in avxintrin.h
#define _mm256_set_m128( hi, lo) _mm256_insertf128_ps( _mm256_castps128_ps256( lo ), ( hi ), 0x1 )
#endif

namespace NeoML {

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Channel count: 24

template<>
inline CBlobConvolution<24>::CSize CBlobConvolution<24>::getWideBatchProcessSize()
{
	return { 1, 3 };
}

template<>
inline void CBlobConvolution<24>::batchProcessChannels( const float* srcPtr, const float* fltPtr,
	__m256& r00, __m256& r01, __m256& r02,
	__m256& r10, __m256& r11, __m256& r12,
	__m256& r20, __m256& r21, __m256& r22 )
{
	// Process three result pixels per time.
	auto ProcessNext = [&]()
	{
		// Load one channel from one pixels in sequenced windows and fill one ymm register with its value.
		__m256 st0 = _mm256_broadcast_ss( srcPtr );
		__m256 st1 = _mm256_broadcast_ss( srcPtr + SrcXStep );
		__m256 st2 = _mm256_broadcast_ss( srcPtr + 2 * SrcXStep );
		// Load one channel for the same pixel as in source for all filters.
		__m256 f0 = _mm256_loadu_ps( fltPtr );
		__m256 f1 = _mm256_loadu_ps( fltPtr + 8 );
		__m256 f2 = _mm256_loadu_ps( fltPtr + 16 );
		// Take result for current pixels in three sequenced windows.
		// Multiply one chennel for all filters ( for ONE src/flt pixels )
		r00 = _mm256_fmadd_ps( f0, st0, r00 );
		r01 = _mm256_fmadd_ps( f1, st0, r01 );
		r02 = _mm256_fmadd_ps( f2, st0, r02 );
		r10 = _mm256_fmadd_ps( f0, st1, r10 );
		r11 = _mm256_fmadd_ps( f1, st1, r11 );
		r12 = _mm256_fmadd_ps( f2, st1, r12 );
		r20 = _mm256_fmadd_ps( f0, st2, r20 );
		r21 = _mm256_fmadd_ps( f1, st2, r21 );
		r22 = _mm256_fmadd_ps( f2, st2, r22 );

		// Go to next chennel in filter and source
		fltPtr += FCm8;
		srcPtr++;
	};

	for( int c = 0; c < C; c++ ) {
		ProcessNext();
	}
}

template<>
inline void CBlobConvolution<24>::singleProcessChannels( const float* srcPtr, const float* fltPtr,
	__m256& r0, __m256& r1, __m256& r2 )
{
	// Process one result pixels per time.
	auto ProcessNext = [&]( __m128 st0_m128 )
	{
		__m256 st0 = _mm256_set_m128( st0_m128, st0_m128 );

		__m256 f0 = _mm256_loadu_ps( fltPtr );
		__m256 f1 = _mm256_loadu_ps( fltPtr + 8 );
		__m256 f2 = _mm256_loadu_ps( fltPtr + 16 );

		r0 = _mm256_fmadd_ps( f0, st0, r0 );
		r1 = _mm256_fmadd_ps( f1, st0, r1 );
		r2 = _mm256_fmadd_ps( f2, st0, r2 );

		fltPtr += FCm8;
	};

	int c = C;
	for( ; c >= 4; c -= 4 ) {
		__m128 s = _mm_loadu_ps( srcPtr );
		// Load four channels and fill __m256 register four times with each of channel.
		ProcessNext( _mm_permute_ps( s, 0x00 ) );
		ProcessNext( _mm_permute_ps( s, 0x55 ) );
		ProcessNext( _mm_permute_ps( s, 0xaa ) );
		ProcessNext( _mm_permute_ps( s, 0xff ) );
		srcPtr += 4;
	}
	if( c != 0 ) {
		// Process remaining channels
		__m128 s = _mm_loadu_ps( srcPtr );
		ProcessNext( _mm_permute_ps( s, 0x00 ) );
		if( c > 1 ) {
			ProcessNext( _mm_permute_ps( s, 0x55 ) );
		}
		if( c > 2 ) {
			ProcessNext( _mm_permute_ps( s, 0xaa ) );
		}
	}
}

template<>
inline void CBlobConvolution<24>::batchProcess( const float* srcPtr, float* dstPtr, int windowIndex, bool /*useNarrowProcessing*/ )
{
	const __m256 ft0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	const __m256 ft1 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 8 ) : _mm256_setzero_ps();
	const __m256 ft2 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 16 ) : _mm256_setzero_ps();

	// Initialize result pixels with freeterm.
	__m256 r00 = ft0;
	__m256 r01 = ft1;
	__m256 r02 = ft2;
	__m256 r10 = ft0;
	__m256 r11 = ft1;
	__m256 r12 = ft2;
	__m256 r20 = ft0;
	__m256 r21 = ft1;
	__m256 r22 = ft2;

	if( windowIndex == -1 ) {
		const float* fltPtr = flt;
		for( int fy = 0; fy < FH; fy++ ) {
			for( int fx = 0; fx < FW; fx++ ) {
				batchProcessChannels( srcPtr, fltPtr, r00, r01, r02, r10, r11, r12, r20, r21, r22 );
				// Move to next pixel in source image on the SAME line
				srcPtr += SrcXDilation;
				fltPtr += C * FCm8;
			}
			// Move to next pixel in source image on the NEXT line
			srcPtr += SrcYDilation - SrcXWindowSize;
		}
	} else {
		auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
		auto fltIt = FltPixelsOffset[windowIndex].cbegin();
		for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
			batchProcessChannels( srcPtr + *srcIt, flt + *fltIt, r00, r01, r02, r10, r11, r12, r20, r21, r22 );
		}
	}
	

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_storeu_ps( dstPtr, r00 );
	_mm256_storeu_ps( dstPtr + 8, r01 );
	_mm256_storeu_ps( dstPtr + 16, r02 );
	_mm256_storeu_ps( dstPtr + 24, r10 );
	_mm256_storeu_ps( dstPtr + 32, r11 );
	_mm256_storeu_ps( dstPtr + 40, r12 );
	_mm256_storeu_ps( dstPtr + 48, r20 );
	_mm256_storeu_ps( dstPtr + 56, r21 );
	_mm256_storeu_ps( dstPtr + 64, r22 );
}

template<>
inline void CBlobConvolution<24>::singleProcess( const float* srcPtr, float* dstPtr, int windowIndex )
{
	// Initialize result pixels with freeterm.
	__m256 r0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	__m256 r1 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 8 ) : _mm256_setzero_ps();
	__m256 r2 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 16 ) : _mm256_setzero_ps();

	if( windowIndex == -1 ) {
		const float* fltPtr = flt;
		for( int fy = 0; fy < FH; fy++ ) {
			for( int fx = 0; fx < FW; fx++ ) {
				singleProcessChannels( srcPtr, fltPtr, r0, r1, r2 );
				// Move to next pixel in source image on the SAME line
				srcPtr += SrcXDilation;
				fltPtr += C * FCm8;
			}
			// Move to next pixel in source image on the NEXT line
			srcPtr += SrcYDilation - SrcXWindowSize;
		}
	} else {
		auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
		auto fltIt = FltPixelsOffset[windowIndex].cbegin();
		for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
			singleProcessChannels( srcPtr + *srcIt, flt + *fltIt, r0, r1, r2 );
		}
	}
	
	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_storeu_ps( dstPtr, r0 );
	_mm256_storeu_ps( dstPtr + 8, r1 );
	_mm256_storeu_ps( dstPtr + 16, r2 );
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Channel count: 18

template<>
inline CBlobConvolution<18>::CSize CBlobConvolution<18>::getWideBatchProcessSize()
{
	return { 1, 4 };
}

template<>
inline void CBlobConvolution<18>::batchProcessChannels( const float* srcPtr, const float* fltPtr,
	__m256& r00, __m256& r01, __m256& r02,
	__m256& r10, __m256& r11, __m256& r12,
	__m256& r20, __m256& r21, __m256& r22 )
{
	auto ProcessNext = [&]()
	{
		// We will process four pixels of source in three steps ( merge each two floats for clarity )
		// After each step we will "rotate" filter's values.
		// Step 1:
		// S: 0 0 0 0 - 0 0 0 0 - 0 1 1 1
		// F: 0 1 2 3 - 4 5 6 7 - 8 0 1 2
		__m256 f0 = _mm256_loadu_ps( fltPtr );
		__m256 f1 = _mm256_loadu_ps( fltPtr + 8 );
		__m256 f2 = _mm256_loadu_ps( fltPtr + 16 );
		__m256 s0 = _mm256_broadcast_ss( srcPtr );
		__m256 s2 = _mm256_broadcast_ss( srcPtr + SrcXStep );
		r00 = _mm256_fmadd_ps( f0, s0, r00 );
		s2 = _mm256_blend_ps( s2, s0, 0x03 );
		r01 = _mm256_fmadd_ps( f1, s0, r01 );
		r02 = _mm256_fmadd_ps( f2, s2, r02 );

		// Step 2:
		// S: 1 1 1 1 - 1 1 2 2 - 2 2 2 2
		// F: 3 4 5 6 - 7 8 0 1 - 2 3 4 5
		RotateLeft6( f0, f1, f2 );
		s0 = _mm256_unpackhi_ps( s2, s2 );
		s2 = _mm256_broadcast_ss( srcPtr + 2 * SrcXStep );
		__m256 s1 = _mm256_blend_ps( s0, s2, 0xf0 );
		r10 = _mm256_fmadd_ps( f0, s0, r10 );
		r11 = _mm256_fmadd_ps( f1, s1, r11 );
		r12 = _mm256_fmadd_ps( f2, s2, r12 );

		// Step 3:
		// S: 2 2 2 3 - 3 3 3 3 - 3 3 3 3
		// F: 6 7 8 0 - 1 2 3 4 - 5 6 7 8
		RotateLeft6( f0, f1, f2 );
		s1 = _mm256_broadcast_ss( srcPtr + 3 * SrcXStep );
		s0 = _mm256_blend_ps( s2, s1, 0xc0 );
		r20 = _mm256_fmadd_ps( f0, s0, r20 );
		r21 = _mm256_fmadd_ps( f1, s1, r21 );
		r22 = _mm256_fmadd_ps( f2, s1, r22 );

		fltPtr += FCm8;
		srcPtr++;
	};

	for( int c = 0; c < C; c++ ) {
		ProcessNext();
	}
}

template<>
inline void CBlobConvolution<18>::singleProcessChannels( const float* srcPtr, const float* fltPtr,
	__m256& r0, __m256& r1, __m256& r2 )
{
	auto ProcessNext = [&]( __m128 st0_m128 )
	{
		__m256 st0 = _mm256_set_m128( st0_m128, st0_m128 );

		__m256 f0 = _mm256_loadu_ps( fltPtr );
		__m256 f1 = _mm256_loadu_ps( fltPtr + 8 );
		__m256 f2 = _mm256_loadu_ps( fltPtr + 16 );

		r0 = _mm256_fmadd_ps( f0, st0, r0 );
		r1 = _mm256_fmadd_ps( f1, st0, r1 );
		r2 = _mm256_fmadd_ps( f2, st0, r2 );

		fltPtr += FCm8;
	};

	int c = C;
	for( ; c >= 4; c -= 4 ) {
		__m128 s = _mm_loadu_ps( srcPtr );
		ProcessNext( _mm_permute_ps( s, 0x00 ) );
		ProcessNext( _mm_permute_ps( s, 0x55 ) );
		ProcessNext( _mm_permute_ps( s, 0xaa ) );
		ProcessNext( _mm_permute_ps( s, 0xff ) );
		srcPtr += 4;
	}
	if( c != 0 ) {
		// Process remaining channels
		__m128 s = _mm_loadu_ps( srcPtr );
				ProcessNext( _mm_permute_ps( s, 0x00 ) );
		if( c > 1 ) {
			ProcessNext( _mm_permute_ps( s, 0x55 ) );
		}
		if( c > 2 ) {
			ProcessNext( _mm_permute_ps( s, 0xaa ) );
		}
	}
}

template<>
inline void CBlobConvolution<18>::batchProcess( const float* srcPtr, float* dstPtr, int windowIndex, bool /*useNarrowProcessing*/ )
{
	__m256 ft0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	__m256 ft1 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 8 ) : _mm256_setzero_ps();
	__m256 ft2 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 16 ) : _mm256_setzero_ps();

	// Initialize result pixels with freeterm.
	__m256 r00 = ft0;
	__m256 r01 = ft1;
	__m256 r02 = ft2;
	RotateLeft6( ft0, ft1, ft2 );
	__m256 r10 = ft0;
	__m256 r11 = ft1;
	__m256 r12 = ft2;
	RotateLeft6( ft0, ft1, ft2 );
	__m256 r20 = ft0;
	__m256 r21 = ft1;
	__m256 r22 = ft2;

	if( windowIndex == -1 ) {
		const float* fltPtr = flt;
		for( int fy = 0; fy < FH; fy++ ) {
			for( int fx = 0; fx < FW; fx++ ) {
				batchProcessChannels( srcPtr, fltPtr, r00, r01, r02, r10, r11, r12, r20, r21, r22 );
				// Move to next pixel in source image on the SAME line
				srcPtr += SrcXDilation;
				fltPtr += C * FCm8;
			}
			// Move to next pixel in source image on the NEXT line
			srcPtr += SrcYDilation - SrcXWindowSize;
		}
	} else {
		auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
		auto fltIt = FltPixelsOffset[windowIndex].cbegin();
		for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
			batchProcessChannels( srcPtr + *srcIt, flt + *fltIt, r00, r01, r02, r10, r11, r12, r20, r21, r22 );
		}
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_storeu_ps( dstPtr, r00 );
	_mm256_storeu_ps( dstPtr + 8, r01 );
	_mm256_storeu_ps( dstPtr + 16, r02 );
	_mm256_storeu_ps( dstPtr + 24, r10 );
	_mm256_storeu_ps( dstPtr + 32, r11 );
	_mm256_storeu_ps( dstPtr + 40, r12 );
	_mm256_storeu_ps( dstPtr + 48, r20 );
	_mm256_storeu_ps( dstPtr + 56, r21 );
	_mm256_storeu_ps( dstPtr + 64, r22 );
}

template<>
inline void CBlobConvolution<18>::singleProcess( const float* srcPtr, float* dstPtr, int windowIndex )
{
	// Initialize result pixels with freeterm.
	__m256 r0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	__m256 r1 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 8 ) : _mm256_setzero_ps();
	__m256 r2 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 16 ) : _mm256_setzero_ps();

	if( windowIndex == -1 ) {
		const float* fltPtr = flt;
		for( int fy = 0; fy < FH; fy++ ) {
			for( int fx = 0; fx < FW; fx++ ) {
				singleProcessChannels( srcPtr, fltPtr, r0, r1, r2 );
				// Move to next pixel in source image on the SAME line
				srcPtr += SrcXDilation;
				fltPtr += C * FCm8;
			}
			// Move to next pixel in source image on the NEXT line
			srcPtr += SrcYDilation - SrcXWindowSize;
		}
	} else {
		auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
		auto fltIt = FltPixelsOffset[windowIndex].cbegin();
		for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
			singleProcessChannels( srcPtr + *srcIt, flt + *fltIt, r0, r1, r2 );
		}
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_storeu_ps( dstPtr, r0 );
	_mm256_storeu_ps( dstPtr + 8, r1 );
	_mm256_maskstore_ps( dstPtr + 16, _mm256_set_epi32( 0, 0, 0, 0, 0, 0, -1, -1 ), r2 );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Channel count: 6

template<>
inline CBlobConvolution<6>::CSize CBlobConvolution<6>::getNarrowBatchProcessSize()
{
	// Height, Width
	return { 3, 4 };
}

template<>
inline CBlobConvolution<6>::CSize CBlobConvolution<6>::getWideBatchProcessSize()
{
	// Height, Width
	return { 1, 12 };
}

template<>
inline void CBlobConvolution<6>::batchProcessChannels( const float* srcPtr, const float* fltPtr,  int srcNarrowStep,
	__m256& r0, __m256& r1, __m256& r2,
	__m256& r3, __m256& r4, __m256& r5,
	__m256& r6, __m256& r7, __m256& r8 )
{
	// Process group of 12 source windows. ( 1x12 for wide case or 3x4 for narrow one )
	const float* srcPtr0 = srcPtr;
	const float* srcPtr1 = srcPtr + srcNarrowStep;
	const float* srcPtr2 = srcPtr + 2 * srcNarrowStep;
	auto ProcessNext = [&]()
	{
		// We will process pixels in three steps
		//           Step 1    Step 2      Step 3
		// Source: 0 0 0 1 - 1 1 2  2  - 2  3  3  3 
		// Source: 4 4 4 5 - 5 5 6  6  - 6  7  7  7 
		// Source: 8 8 8 9 - 9 9 10 10 - 10 11 11 11 
		// Filter: 0 1 2 0 - 1 2 0  1  - 2  0  1  2
		// Three groups of source windows are sequenced in wide case or are placed one below another in narrow case.
		// load: 0 1 2 0
		__m256 f0 = _mm256_loadu_ps( fltPtr );

		// load: 0 0 0 0
		__m256 s0 = _mm256_broadcast_ss( srcPtr0 );
		// load: 1 1 1 1
		__m256 st0 = _mm256_broadcast_ss( srcPtr0 + SrcXStep );
		// load: 4 4 4 4
		__m256 s1 = _mm256_broadcast_ss( srcPtr1 );
		// load: 5 5 5 5
		__m256 st1 = _mm256_broadcast_ss( srcPtr1 + SrcXStep );
		// load: 8 8 8 8
		__m256 s2 = _mm256_broadcast_ss( srcPtr2 );
		// load: 9 9 9 9
		__m256 st2 = _mm256_broadcast_ss( srcPtr2 + SrcXStep );

		// blend( 0 0 0 0, 1 1 1 1 ) -> 0 0 0 1
		s0 = _mm256_blend_ps( s0, st0, 0xc0 );
		// blend( 4 4 4 4, 5 5 5 5 ) -> 4 4 4 5
		s1 = _mm256_blend_ps( s1, st1, 0xc0 );
		// blend( 8 8 8 8, 9 9 9 9 ) -> 8 8 8 9
		s2 = _mm256_blend_ps( s2, st2, 0xc0 );
		r0 = _mm256_fmadd_ps( f0, s0, r0 );
		r3 = _mm256_fmadd_ps( f0, s1, r3 );
		r6 = _mm256_fmadd_ps( f0, s2, r6 );
		// 0 1 2 0 -> 1 2 0 1
		RotateLeft2( f0 );

		// load: 2 2 2 2
		s0 = _mm256_broadcast_ss( srcPtr0 + 2 *SrcXStep );
		// load: 6 6 6 6
		s1 = _mm256_broadcast_ss( srcPtr1 + 2 * SrcXStep );
		// load: 10 10 10 10
		s2 = _mm256_broadcast_ss( srcPtr2 + 2 * SrcXStep );

		// blend( 1 1 1 1, 2 2 2 2 ) -> 1 1 2 2
		st0 = _mm256_blend_ps( st0, s0, 0xf0 );
		// blend( 5 5 5 5, 6 6 6 6 ) -> 5 5 6 6
		st1 = _mm256_blend_ps( st1, s1, 0xf0 );
		// blend( 9 9 9 9, 10 10 10 10 ) -> 9 9 10 10
		st2 = _mm256_blend_ps( st2, s2, 0xf0 );
		r1 = _mm256_fmadd_ps( f0, st0, r1 );
		r4 = _mm256_fmadd_ps( f0, st1, r4 );
		r7 = _mm256_fmadd_ps( f0, st2, r7 );
		// 1 2 0 1 -> 2 0 1 2
		RotateLeft2( f0 );

		// load: 3 3 3 3
		st0 = _mm256_broadcast_ss( srcPtr0 + 3 * SrcXStep );
		// load: 7 7 7 7
		st1 = _mm256_broadcast_ss( srcPtr1 + 3 * SrcXStep );
		// load: 11 11 11 11
		st2 = _mm256_broadcast_ss( srcPtr2 + 3 * SrcXStep );

		// blend( 3 3 3 3, 2 2 2 2 ) -> 2 3 3 3
		st0 = _mm256_blend_ps( st0, s0, 0x03 );
		// blend( 7 7 7 7, 6 6 6 6 ) -> 6 7 7 7
		st1 = _mm256_blend_ps( st1, s1, 0x03 );
		// blend( 11 11 11 11, 10 10 10 10 ) -> 10 11 11 11
		st2 = _mm256_blend_ps( st2, s2, 0x03 );
		r2 = _mm256_fmadd_ps( f0, st0, r2 );
		r5 = _mm256_fmadd_ps( f0, st1, r5 );
		r8 = _mm256_fmadd_ps( f0, st2, r8 );

		srcPtr0++;
		srcPtr1++;
		srcPtr2++;
		fltPtr += FCm8;
	};

	for( int c = 0; c < C; c++ ) {
		ProcessNext();
	}
}

template<>
inline void CBlobConvolution<6>::singleProcessChannels( const float* srcPtr, const float* fltPtr,
	__m256& r0 )
{
	auto ProcessNext = [&]( __m128 st0_m128 )
	{
		__m256 st0 = _mm256_set_m128( st0_m128, st0_m128 );

		__m256 f0 = _mm256_loadu_ps( fltPtr );

		r0 = _mm256_fmadd_ps( f0, st0, r0 );

		fltPtr += FCm8;
	};

	int c = C;
	for( ; c >= 4; c -= 4 ) {
		__m128 s = _mm_loadu_ps( srcPtr );
		ProcessNext( _mm_permute_ps( s, 0x00 ) );
		ProcessNext( _mm_permute_ps( s, 0x55 ) );
		ProcessNext( _mm_permute_ps( s, 0xaa ) );
		ProcessNext( _mm_permute_ps( s, 0xff ) );
		srcPtr += 4;
	}
	if( c != 0 ) {
		__m128 s = _mm_loadu_ps( srcPtr );
				ProcessNext( _mm_permute_ps( s, 0x00 ) );
		if( c > 1 ) ProcessNext( _mm_permute_ps( s, 0x55 ) );
		if( c > 2 )	ProcessNext( _mm_permute_ps( s, 0xaa ) );
	}
}

template<>
inline void CBlobConvolution<6>::singleProcessChannelsNarrow( const float* srcPtr, const float* fltPtr,
	__m256& r0, __m256& r1, __m256& r2 )
{
	// Process thee source windows one below another.
	auto ProcessNext = [&]( __m128 st0_m128, __m128 st1_m128, __m128 st2_m128 )
	{
		__m256 st0 = _mm256_set_m128( st0_m128, st0_m128 );
		__m256 st1 = _mm256_set_m128( st1_m128, st1_m128 );
		__m256 st2 = _mm256_set_m128( st2_m128, st2_m128 );

		__m256 f0 = _mm256_loadu_ps( fltPtr );

		r0 = _mm256_fmadd_ps( f0, st0, r0 );
		r1 = _mm256_fmadd_ps( f0, st1, r1 );
		r2 = _mm256_fmadd_ps( f0, st2, r2 );

		fltPtr += FCm8;
	};

	int c = C;
	for( ; c >= 4; c -= 4 ) {
		__m128 s0 = _mm_loadu_ps( srcPtr );
		__m128 s1 = _mm_loadu_ps( srcPtr + SrcYStep );
		__m128 s2 = _mm_loadu_ps( srcPtr + 2 * SrcYStep );
		srcPtr += 4;
		ProcessNext( _mm_permute_ps( s0, 0x00 ), _mm_permute_ps( s1, 0x00 ), _mm_permute_ps( s2, 0x00 ) );
		ProcessNext( _mm_permute_ps( s0, 0x55 ), _mm_permute_ps( s1, 0x55 ), _mm_permute_ps( s2, 0x55 ) );
		ProcessNext( _mm_permute_ps( s0, 0xaa ), _mm_permute_ps( s1, 0xaa ), _mm_permute_ps( s2, 0xaa ) );
		ProcessNext( _mm_permute_ps( s0, 0xff ), _mm_permute_ps( s1, 0xff ), _mm_permute_ps( s2, 0xff ) );
	}
	if( c != 0 ) {
		__m128 s0 = _mm_loadu_ps( srcPtr );
		__m128 s1 = _mm_loadu_ps( srcPtr + SrcYStep );
		__m128 s2 = _mm_loadu_ps( srcPtr + 2 * SrcYStep );
				ProcessNext( _mm_permute_ps( s0, 0x00 ), _mm_permute_ps( s1, 0x00 ), _mm_permute_ps( s2, 0x00 ) );
		if( c > 1 ) {
				ProcessNext( _mm_permute_ps( s0, 0x55 ), _mm_permute_ps( s1, 0x55 ), _mm_permute_ps( s2, 0x55 ) );
		}
		if( c > 2 ) {
				ProcessNext( _mm_permute_ps( s0, 0xaa ), _mm_permute_ps( s1, 0xaa ), _mm_permute_ps( s2, 0xaa ) );
		}
	}
}

template<>
inline void CBlobConvolution<6>::batchProcess( const float* srcPtr, float* dstPtr, int windowIndex, bool useNarrowProcessing )
{
	// We will set this member for narrow batch processing in order to step between neighbor source windows.
	const int srcNarrowStep = useNarrowProcessing ? SrcYStep : 4 * SrcXStep;
	const int dstNarraowStep = useNarrowProcessing ? DstLineStride : 24;
	__m256 ft0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();

	// Initialize result pixels with freeterm.
	__m256 r0 = ft0;
	__m256 r3 = ft0;
	__m256 r6 = ft0;
	RotateLeft2( ft0 );
	__m256 r1 = ft0;
	__m256 r4 = ft0;
	__m256 r7 = ft0;
	RotateLeft2( ft0 );
	__m256 r2 = ft0;
	__m256 r5 = ft0;
	__m256 r8 = ft0;

	if( windowIndex == -1 ) {
		const float* fltPtr = flt;
		for( int fy = 0; fy < FH; fy++ ) {
			for( int fx = 0; fx < FW; fx++ ) {
				batchProcessChannels( srcPtr, fltPtr, srcNarrowStep, r0, r1, r2, r3, r4, r5, r6, r7, r8 );
				// Move to next pixel in source image on the SAME line
				srcPtr += SrcXDilation;
				fltPtr += C * FCm8;
			}
			// Move to next pixel in source image on the NEXT line
			srcPtr += SrcYDilation - SrcXWindowSize;
		}
	} else {
		auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
		auto fltIt = FltPixelsOffset[windowIndex].cbegin();
		for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
			batchProcessChannels( srcPtr + *srcIt, flt + *fltIt, srcNarrowStep, r0, r1, r2, r3, r4, r5, r6, r7, r8 );
		}
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_storeu_ps( dstPtr, r0 );
	_mm256_storeu_ps( dstPtr + 8, r1 );
	_mm256_storeu_ps( dstPtr + 16, r2 );
	_mm256_storeu_ps( dstPtr + dstNarraowStep, r3 );
	_mm256_storeu_ps( dstPtr + dstNarraowStep + 8, r4 );
	_mm256_storeu_ps( dstPtr + dstNarraowStep + 16, r5 );
	_mm256_storeu_ps( dstPtr + 2 * dstNarraowStep, r6 );
	_mm256_storeu_ps( dstPtr + 2 * dstNarraowStep + 8, r7 );
	_mm256_storeu_ps( dstPtr + 2 * dstNarraowStep + 16, r8 );
}

template<>
inline void CBlobConvolution<6>::singleProcess( const float* srcPtr, float* dstPtr, int windowIndex )
{
	// Initialize result pixels with freeterm.
	__m256 r0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();

	if( windowIndex == -1 ) {
		const float* fltPtr = flt;
		for( int fy = 0; fy < FH; fy++ ) {
			for( int fx = 0; fx < FW; fx++ ) {
				singleProcessChannels( srcPtr, fltPtr, r0 );
				// Move to next pixel in source image on the SAME line
				srcPtr += SrcXDilation;
				fltPtr += C * FCm8;
			}
			// Move to next pixel in source image on the NEXT line
			srcPtr += SrcYDilation - SrcXWindowSize;
		}
	} else {
		auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
		auto fltIt = FltPixelsOffset[windowIndex].cbegin();
		for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
			singleProcessChannels( srcPtr + *srcIt, flt + *fltIt, r0 );
		}
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_maskstore_ps( dstPtr, _mm256_set_epi32( 0, 0, -1, -1, -1, -1, -1, -1 ), r0 );
}

template<>
inline void CBlobConvolution<6>::singleProcessNarrow( const float* srcPtr, float* dstPtr, int windowIndex )
{
	// Initialize result pixels with freeterm.
	__m256 f0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	__m256 r0 = f0;
	__m256 r1 = f0;
	__m256 r2 = f0;

	if( windowIndex == -1 ) {
		const float* fltPtr = flt;
		for( int fy = 0; fy < FH; fy++ ) {
			for( int fx = 0; fx < FW; fx++ ) {
				singleProcessChannelsNarrow( srcPtr, fltPtr, r0, r1, r2 );
				// Move to next pixel in source image on the SAME line
				srcPtr += SrcXDilation;
				fltPtr += C * FCm8;
			}
			// Move to next pixel in source image on the NEXT line
			srcPtr += SrcYDilation - SrcXWindowSize;
		}
	} else {
		auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
		auto fltIt = FltPixelsOffset[windowIndex].cbegin();
		for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
			singleProcessChannelsNarrow( srcPtr + *srcIt, flt + *fltIt, r0, r1, r2 );
		}
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_maskstore_ps( dstPtr, _mm256_set_epi32( 0, 0, -1, -1, -1, -1, -1, -1 ), r0 );
	_mm256_maskstore_ps( dstPtr + DstLineStride, _mm256_set_epi32( 0, 0, -1, -1, -1, -1, -1, -1 ), r1 );
	_mm256_maskstore_ps( dstPtr + 2 * DstLineStride, _mm256_set_epi32( 0, 0, -1, -1, -1, -1, -1, -1 ), r2 );
}

} // namespace NeoML
