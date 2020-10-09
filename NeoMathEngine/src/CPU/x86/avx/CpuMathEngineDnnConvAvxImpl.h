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
// Channels: 24
// Channel count: 24
// Filter height: 3
// Filter width: 3

template<>
constexpr int CBlobConvolution<24, 24, 3, 3>::getBatchProcessSize()
{
	return 3;
}

template<>
inline void CBlobConvolution<24, 24, 3, 3>::partialBatchProcess( const float* srcPtr, const std::vector<int>& srcPixelsOffset,
	const float* fltPtr, const vector<int>& fltPixelsOffset, float* dstPtr )
{
	const __m256 ft0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	const __m256 ft1 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 8 ) : _mm256_setzero_ps();
	const __m256 ft2 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 16 ) : _mm256_setzero_ps();

	__m256 r00 = ft0;
	__m256 r01 = ft1;
	__m256 r02 = ft2;
	__m256 r10 = ft0;
	__m256 r11 = ft1;
	__m256 r12 = ft2;
	__m256 r20 = ft0;
	__m256 r21 = ft1;
	__m256 r22 = ft2;

	auto srcIt = srcPixelsOffset.cbegin();
	auto fltIt = fltPixelsOffset.cbegin();
	for( ; srcIt != srcPixelsOffset.cend(); srcIt++, fltIt++ ) {
		batchProcessChannels( srcPtr + *srcIt, fltPtr + *fltIt, r00, r01, r02, r10, r11, r12, r20, r21, r22 );
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
inline void CBlobConvolution<24, 24, 3, 3>::partialSingleProcess( const float* srcPtr, const std::vector<int>& srcPixelsOffset,
	const float* fltPtr, const std::vector<int>& fltPixelsOffset, float* dstPtr )
{
	__m256 r0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	__m256 r1 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 8 ) : _mm256_setzero_ps();
	__m256 r2 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 16 ) : _mm256_setzero_ps();

	auto srcIt = srcPixelsOffset.cbegin();
	auto fltIt = fltPixelsOffset.cbegin();
	for( ; srcIt != srcPixelsOffset.cend(); srcIt++, fltIt++ ) {
		singleProcessChannels( srcPtr + *srcIt, fltPtr + *fltIt, r0, r1, r2 );
	}

	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_storeu_ps( dstPtr, r0 );
	_mm256_storeu_ps( dstPtr + 8, r1 );
	_mm256_storeu_ps( dstPtr + 16, r2 );
}

template<>
inline void CBlobConvolution<24, 24, 3, 3>::wholeBatchProcess( const float* srcPtr, const float* fltPtr, float* dstPtr )
{
	const __m256 ft0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	const __m256 ft1 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 8 ) : _mm256_setzero_ps();
	const __m256 ft2 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 16 ) : _mm256_setzero_ps();

	__m256 r00 = ft0;
	__m256 r01 = ft1;
	__m256 r02 = ft2;
	__m256 r10 = ft0;
	__m256 r11 = ft1;
	__m256 r12 = ft2;
	__m256 r20 = ft0;
	__m256 r21 = ft1;
	__m256 r22 = ft2;

	for( int fy = 0; fy < FH; fy++ ) {
		for( int fx = 0; fx < FW; fx++ ) {
			batchProcessChannels( srcPtr, fltPtr, r00, r01, r02, r10, r11, r12, r20, r21, r22 );
			// Move to next pixel in source image on the SAME line
			srcPtr += SrcXDilation;
			fltPtr += C * FC;
		}
		// Move to next pixel in source image on the NEXT line
		srcPtr += SrcYDilation - SrcXWindowSize;
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
inline void CBlobConvolution<24, 24, 3, 3>::wholeSingleProcess( const float* srcPtr, const float* fltPtr, float* dstPtr )
{
	__m256 r0 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm ) : _mm256_setzero_ps();
	__m256 r1 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 8 ) : _mm256_setzero_ps();
	__m256 r2 = freeTerm != nullptr ? _mm256_loadu_ps( freeTerm + 16 ) : _mm256_setzero_ps();

	for( int fy = 0; fy < FH; fy++ ) {
		for( int fx = 0; fx < FW; fx++ ) {
			singleProcessChannels( srcPtr, fltPtr, r0, r1, r2 );
			// Move to next pixel in source image on the SAME line
			srcPtr += SrcXDilation;
			fltPtr += C * FC;
		}
		// Move to next pixel in source image on the NEXT line
		srcPtr += SrcYDilation - SrcXWindowSize;
	}
	// Store result of convolution for (fx,fy) pixel of f-th channel
	_mm256_storeu_ps( dstPtr, r0 );
	_mm256_storeu_ps( dstPtr + 8, r1 );
	_mm256_storeu_ps( dstPtr + 16, r2 );
}

template<>
inline void CBlobConvolution<24, 24, 3, 3>::batchProcessChannels( const float* srcPtr, const float* fltPtr,
	__m256& r00, __m256& r01, __m256& r02,
	__m256& r10, __m256& r11, __m256& r12,
	__m256& r20, __m256& r21, __m256& r22 )
{
	auto ProcessNext = [&]()
	{
		__m256 st0 = _mm256_broadcast_ss( srcPtr );
		__m256 st1 = _mm256_broadcast_ss( srcPtr + SrcXStep );
		__m256 st2 = _mm256_broadcast_ss( srcPtr + 2 * SrcXStep );
		__m256 f0 = _mm256_loadu_ps( fltPtr );
		__m256 f1 = _mm256_loadu_ps( fltPtr + 8 );
		__m256 f2 = _mm256_loadu_ps( fltPtr + 16 );
		r00 = _mm256_fmadd_ps( f0, st0, r00 );
		r01 = _mm256_fmadd_ps( f1, st0, r01 );
		r02 = _mm256_fmadd_ps( f2, st0, r02 );
		r10 = _mm256_fmadd_ps( f0, st1, r10 );
		r11 = _mm256_fmadd_ps( f1, st1, r11 );
		r12 = _mm256_fmadd_ps( f2, st1, r12 );
		r20 = _mm256_fmadd_ps( f0, st2, r20 );
		r21 = _mm256_fmadd_ps( f1, st2, r21 );
		r22 = _mm256_fmadd_ps( f2, st2, r22 );

		fltPtr += FC;
		srcPtr++;
	};

	for( int c = 0; c < C; c++ ) {
		ProcessNext();
	}
}

template<>
inline void CBlobConvolution<24, 24, 3, 3>::singleProcessChannels( const float* srcPtr, const float* fltPtr,
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

		fltPtr += FC;
	};

	for( int c = 0; c < C; c += 4 ) {
		__m128 s = _mm_loadu_ps( srcPtr + c );
		ProcessNext( _mm_permute_ps( s, 0x00 ) );
		ProcessNext( _mm_permute_ps( s, 0x55 ) );
		ProcessNext( _mm_permute_ps( s, 0xaa ) );
		ProcessNext( _mm_permute_ps( s, 0xff ) );
	}
}

} // namespace NeoML