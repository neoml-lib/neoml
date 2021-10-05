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
inline void CBlobConvolution<6>::batchProcessChannels( const float* srcPtr, const float* fltPtr, size_t srcNarrowStep,
	__m256& r0, __m256& r1, __m256& r2,
	__m256& r3, __m256& r4, __m256& r5,
	__m256& r6, __m256& r7, __m256& r8 )
{
	// Process group of 12 source windows. ( 1x12 for wide case or 3x4 for narrow one )
	const float* srcPtr0 = srcPtr;
	const float* srcPtr1 = srcPtr + srcNarrowStep;
	const float* srcPtr2 = srcPtr + 2 * srcNarrowStep;
	auto processNext = [&]()
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
		rotateLeft2( f0 );

		// load: 2 2 2 2
		s0 = _mm256_broadcast_ss( srcPtr0 + 2 * SrcXStep );
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
		rotateLeft2( f0 );

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
		fltPtr += FltCntM8;
	};

	for( int c = 0; c < ChCnt; c++ ) {
		processNext();
	}
}

template<>
inline void CBlobConvolution<6>::singleProcessChannels( const float* srcPtr, const float* fltPtr,
	__m256& r0 )
{
	auto processNext = [&]( __m128 st0_m128 )
	{
		__m256 st0 = _mm256_set_m128( st0_m128, st0_m128 );

		__m256 f0 = _mm256_loadu_ps( fltPtr );

		r0 = _mm256_fmadd_ps( f0, st0, r0 );

		fltPtr += FltCntM8;
	};

	int c = ChCnt;
	for( ; c >= 4; c -= 4 ) {
		__m128 s = _mm_loadu_ps( srcPtr );
		processNext( _mm_permute_ps( s, 0x00 ) );
		processNext( _mm_permute_ps( s, 0x55 ) );
		processNext( _mm_permute_ps( s, 0xaa ) );
		processNext( _mm_permute_ps( s, 0xff ) );
		srcPtr += 4;
	}
	if( c != 0 ) {
		__m128 s = _mm_loadu_ps( srcPtr );
		processNext( _mm_permute_ps( s, 0x00 ) );
		if( c > 1 ) processNext( _mm_permute_ps( s, 0x55 ) );
		if( c > 2 )	processNext( _mm_permute_ps( s, 0xaa ) );
	}
}

template<>
inline void CBlobConvolution<6>::singleProcessChannelsNarrow( const float* srcPtr, const float* fltPtr,
	__m256& r0, __m256& r1, __m256& r2 )
{
	// Process thee source windows one below another.
	auto processNext = [&]( __m128 st0_m128, __m128 st1_m128, __m128 st2_m128 )
	{
		__m256 st0 = _mm256_set_m128( st0_m128, st0_m128 );
		__m256 st1 = _mm256_set_m128( st1_m128, st1_m128 );
		__m256 st2 = _mm256_set_m128( st2_m128, st2_m128 );

		__m256 f0 = _mm256_loadu_ps( fltPtr );

		r0 = _mm256_fmadd_ps( f0, st0, r0 );
		r1 = _mm256_fmadd_ps( f0, st1, r1 );
		r2 = _mm256_fmadd_ps( f0, st2, r2 );

		fltPtr += FltCntM8;
	};

	int c = ChCnt;
	for( ; c >= 4; c -= 4 ) {
		__m128 s0 = _mm_loadu_ps( srcPtr );
		__m128 s1 = _mm_loadu_ps( srcPtr + SrcYStep );
		__m128 s2 = _mm_loadu_ps( srcPtr + 2 * SrcYStep );
		srcPtr += 4;
		processNext( _mm_permute_ps( s0, 0x00 ), _mm_permute_ps( s1, 0x00 ), _mm_permute_ps( s2, 0x00 ) );
		processNext( _mm_permute_ps( s0, 0x55 ), _mm_permute_ps( s1, 0x55 ), _mm_permute_ps( s2, 0x55 ) );
		processNext( _mm_permute_ps( s0, 0xaa ), _mm_permute_ps( s1, 0xaa ), _mm_permute_ps( s2, 0xaa ) );
		processNext( _mm_permute_ps( s0, 0xff ), _mm_permute_ps( s1, 0xff ), _mm_permute_ps( s2, 0xff ) );
	}
	if( c != 0 ) {
		__m128 s0 = _mm_loadu_ps( srcPtr );
		__m128 s1 = _mm_loadu_ps( srcPtr + SrcYStep );
		__m128 s2 = _mm_loadu_ps( srcPtr + 2 * SrcYStep );
		processNext( _mm_permute_ps( s0, 0x00 ), _mm_permute_ps( s1, 0x00 ), _mm_permute_ps( s2, 0x00 ) );
		if( c > 1 ) {
			processNext( _mm_permute_ps( s0, 0x55 ), _mm_permute_ps( s1, 0x55 ), _mm_permute_ps( s2, 0x55 ) );
		}
		if( c > 2 ) {
			processNext( _mm_permute_ps( s0, 0xaa ), _mm_permute_ps( s1, 0xaa ), _mm_permute_ps( s2, 0xaa ) );
		}
	}
}

template<>
inline void CBlobConvolution<6>::batchProcess( const float* srcPtr, float* resPtr, size_t windowIndex, bool useNarrowProcessing )
{
	// We will set this member for narrow batch processing in order to step between neighbor source windows.
	const size_t srcNarrowStep = useNarrowProcessing ? SrcYStep : 4 * SrcXStep;
	const size_t resNarrowStep = useNarrowProcessing ? ResLineStride : 24;
	__m256 ft0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();

	// Initialize result pixels with freeterm.
	__m256 r0 = ft0;
	__m256 r3 = ft0;
	__m256 r6 = ft0;
	rotateLeft2( ft0 );
	__m256 r1 = ft0;
	__m256 r4 = ft0;
	__m256 r7 = ft0;
	rotateLeft2( ft0 );
	__m256 r2 = ft0;
	__m256 r5 = ft0;
	__m256 r8 = ft0;

	auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
	auto fltIt = FltPixelsOffset[windowIndex].cbegin();
	for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
		batchProcessChannels( srcPtr + *srcIt, flt + *fltIt, srcNarrowStep, r0, r1, r2, r3, r4, r5, r6, r7, r8 );
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_storeu_ps( resPtr, r0 );
	_mm256_storeu_ps( resPtr + 8, r1 );
	_mm256_storeu_ps( resPtr + 16, r2 );
	_mm256_storeu_ps( resPtr + resNarrowStep, r3 );
	_mm256_storeu_ps( resPtr + resNarrowStep + 8, r4 );
	_mm256_storeu_ps( resPtr + resNarrowStep + 16, r5 );
	_mm256_storeu_ps( resPtr + 2 * resNarrowStep, r6 );
	_mm256_storeu_ps( resPtr + 2 * resNarrowStep + 8, r7 );
	_mm256_storeu_ps( resPtr + 2 * resNarrowStep + 16, r8 );
}

template<>
inline void CBlobConvolution<6>::singleProcess( const float* srcPtr, float* resPtr, size_t windowIndex )
{
	// Initialize result pixels with freeterm.
	__m256 r0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();

	auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
	auto fltIt = FltPixelsOffset[windowIndex].cbegin();
	for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
		singleProcessChannels( srcPtr + *srcIt, flt + *fltIt, r0 );
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_maskstore_ps( resPtr, _mm256_set_epi32( 0, 0, -1, -1, -1, -1, -1, -1 ), r0 );
}

template<>
inline void CBlobConvolution<6>::singleProcessNarrow( const float* srcPtr, float* resPtr, size_t windowIndex )
{
	// Initialize result pixels with freeterm.
	__m256 f0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	__m256 r0 = f0;
	__m256 r1 = f0;
	__m256 r2 = f0;

	auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
	auto fltIt = FltPixelsOffset[windowIndex].cbegin();
	for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
		singleProcessChannelsNarrow( srcPtr + *srcIt, flt + *fltIt, r0, r1, r2 );
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_maskstore_ps( resPtr, _mm256_set_epi32( 0, 0, -1, -1, -1, -1, -1, -1 ), r0 );
	_mm256_maskstore_ps( resPtr + ResLineStride, _mm256_set_epi32( 0, 0, -1, -1, -1, -1, -1, -1 ), r1 );
	_mm256_maskstore_ps( resPtr + 2 * ResLineStride, _mm256_set_epi32( 0, 0, -1, -1, -1, -1, -1, -1 ), r2 );
}

} // namespace NeoML
