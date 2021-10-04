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
inline void CBlobConvolution<24>::batchProcessChannels( const float* srcPtr, const float* fltPtr,
	__m256& r00, __m256& r01, __m256& r02,
	__m256& r10, __m256& r11, __m256& r12,
	__m256& r20, __m256& r21, __m256& r22 )
{
	// Process three result pixels per time.
	auto processNext = [&]()
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
		// Multiply one channel for all filters ( for ONE src/flt pixels )
		r00 = _mm256_fmadd_ps( f0, st0, r00 );
		r01 = _mm256_fmadd_ps( f1, st0, r01 );
		r02 = _mm256_fmadd_ps( f2, st0, r02 );
		r10 = _mm256_fmadd_ps( f0, st1, r10 );
		r11 = _mm256_fmadd_ps( f1, st1, r11 );
		r12 = _mm256_fmadd_ps( f2, st1, r12 );
		r20 = _mm256_fmadd_ps( f0, st2, r20 );
		r21 = _mm256_fmadd_ps( f1, st2, r21 );
		r22 = _mm256_fmadd_ps( f2, st2, r22 );

		// Go to next channel in filter and source
		fltPtr += FltCntM8;
		srcPtr++;
	};

	for( int c = 0; c < ChCnt; c++ ) {
		processNext();
	}
}

template<>
inline void CBlobConvolution<24>::singleProcessChannels( const float* srcPtr, const float* fltPtr,
	__m256& r0, __m256& r1, __m256& r2 )
{
	// Process one result pixels per time.
	auto processNext = [&]( __m128 st0_m128 )
	{
		__m256 st0 = _mm256_set_m128( st0_m128, st0_m128 );

		__m256 f0 = _mm256_loadu_ps( fltPtr );
		__m256 f1 = _mm256_loadu_ps( fltPtr + 8 );
		__m256 f2 = _mm256_loadu_ps( fltPtr + 16 );

		r0 = _mm256_fmadd_ps( f0, st0, r0 );
		r1 = _mm256_fmadd_ps( f1, st0, r1 );
		r2 = _mm256_fmadd_ps( f2, st0, r2 );

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
inline void CBlobConvolution<24>::batchProcess( const float* srcPtr, float* resPtr, size_t windowIndex, bool /*useNarrowProcessing*/ )
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

	auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
	auto fltIt = FltPixelsOffset[windowIndex].cbegin();
	for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
		batchProcessChannels( srcPtr + *srcIt, flt + *fltIt, r00, r01, r02, r10, r11, r12, r20, r21, r22 );
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_storeu_ps( resPtr, r00 );
	_mm256_storeu_ps( resPtr + 8, r01 );
	_mm256_storeu_ps( resPtr + 16, r02 );
	_mm256_storeu_ps( resPtr + 24, r10 );
	_mm256_storeu_ps( resPtr + 32, r11 );
	_mm256_storeu_ps( resPtr + 40, r12 );
	_mm256_storeu_ps( resPtr + 48, r20 );
	_mm256_storeu_ps( resPtr + 56, r21 );
	_mm256_storeu_ps( resPtr + 64, r22 );
}

template<>
inline void CBlobConvolution<24>::singleProcess( const float* srcPtr, float* resPtr, size_t windowIndex )
{
	// Initialize result pixels with freeterm.
	__m256 r0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	__m256 r1 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 8 ) : _mm256_setzero_ps();
	__m256 r2 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 16 ) : _mm256_setzero_ps();

	auto srcIt = SrcPixelsOffset[windowIndex].cbegin();
	auto fltIt = FltPixelsOffset[windowIndex].cbegin();
	for( ; srcIt != SrcPixelsOffset[windowIndex].cend(); srcIt++, fltIt++ ) {
		singleProcessChannels( srcPtr + *srcIt, flt + *fltIt, r0, r1, r2 );
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_storeu_ps( resPtr, r0 );
	_mm256_storeu_ps( resPtr + 8, r1 );
	_mm256_storeu_ps( resPtr + 16, r2 );
}

} // namespace NeoML
