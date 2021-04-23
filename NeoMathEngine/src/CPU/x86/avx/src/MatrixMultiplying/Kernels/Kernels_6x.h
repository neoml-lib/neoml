/* Copyright © 2017-2021 ABBYY Production LLC

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

#include <AvxCommon.h>
#include <MicroKernels/MicroKernelBase.h>

// Prepare and transpose the matrix
template<>
struct CInterleaverBase<true, 4> {
	static void Prepare( float* out, const float* in, size_t stride, size_t width, size_t height )
	{
		const int Len = 4;

		const size_t iStep = stride * Len;
		const size_t oStep = width * Len;

		for( ; height >= Len; height -= Len ) {
			int tempWidth = width;
			const float* tempIn = in;
			float* tempOut = out;
			for( ; tempWidth >= 8; tempWidth -= 8 ) {
				__m256 a = _mm256_loadu_ps( tempIn + 0 * stride );
				__m256 b = _mm256_loadu_ps( tempIn + 1 * stride );
				__m256 c = _mm256_loadu_ps( tempIn + 2 * stride );
				__m256 d = _mm256_loadu_ps( tempIn + 3 * stride );

				// a:     a0 a1 a2 a3 a4 a5 a6 a7
				// b:     b0 b1 b2 b3 b4 b5 b6 b7
				// ab_lo: a0 b0 a1 b1 a4 b4 a5 b5
				__m256 ab_lo = _mm256_unpacklo_ps( a, b );
				// a:     a0 a1 a2 a3 a4 a5 a6 a7
				// b:     b0 b1 b2 b3 b4 b5 b6 b7
				// ab_hi: a2 b2 a3 b3 a6 b6 a7 b7
				__m256 ab_hi = _mm256_unpackhi_ps( a, b );
				// c:     c0 c1 c2 c3 c4 c5 c6 c7
				// d:     d0 d1 d2 d3 d4 d5 d6 d7
				// cd_lo: c0 d0 c1 d1 c4 d4 c5 d5
				__m256 cd_lo = _mm256_unpacklo_ps( c, d );
				// c:     c0 c1 c2 c3 c4 c5 c6 c7
				// d:     d0 d1 d2 d3 d4 d5 d6 d7
				// cd_hi: c2 d2 c3 d3 c6 d6 c7 d7
				__m256 cd_hi = _mm256_unpackhi_ps( c, d );

				// ab_lo:   a0 b0 a1 b1 a4 b4 a5 b5
				// cd_lo:   c0 d0 c1 d1 c4 d4 c5 d5
				// abcd_04: a0 b0 c0 d0 a4 b4 c4 d4
				__m256 abcd_04 = _mm256_shuffle_ps( ab_lo, cd_lo, _MM_SHUFFLE( 1, 0, 1, 0 ) );
				__m256 abcd_15 = _mm256_shuffle_ps( ab_lo, cd_lo, _MM_SHUFFLE( 3, 2, 3, 2 ) );
				__m256 abcd_26 = _mm256_shuffle_ps( ab_hi, cd_hi, _MM_SHUFFLE( 1, 0, 1, 0 ) );
				__m256 abcd_37 = _mm256_shuffle_ps( ab_hi, cd_hi, _MM_SHUFFLE( 3, 2, 3, 2 ) );

				// __m256 abcd_01 = _mm256_permute2f128_ps( abcd_04, abcd_15, 0 | 2 << 4 );
				_mm256_storeu_ps( tempOut + 0, _mm256_permute2f128_ps( abcd_04, abcd_15, 0 | 2 << 4 ) );
				// __m256 abcd_23 = _mm256_permute2f128_ps( abcd_26, abcd_37, 0 | 2 << 4 );
				_mm256_storeu_ps( tempOut + 8, _mm256_permute2f128_ps( abcd_26, abcd_37, 0 | 2 << 4 ) );
				// __m256 abcd_45 = _mm256_permute2f128_ps( abcd_04, abcd_15, 1 | 3 << 4 );
				_mm256_storeu_ps( tempOut + 16, _mm256_permute2f128_ps( abcd_04, abcd_15, 1 | 3 << 4 ) );
				// __m256 abcd_67 = _mm256_permute2f128_ps( abcd_26, abcd_37, 1 | 3 << 4 );
				_mm256_storeu_ps( tempOut + 24, _mm256_permute2f128_ps( abcd_26, abcd_37, 1 | 3 << 4 ) );
				tempIn += 8;
				tempOut += 32;
			}
			if( tempWidth != 0 ) {
				CInterleaverBase<false, 1>::Transpose( tempOut, Len, tempIn, stride, Len, tempWidth );
			}
			in += iStep;
			out += oStep;
		}
		height %= Len;
		if( height > 0 ) {
			CInterleaverBase<false, 1>::Transpose(out, Len, in, stride, height, width);
			out += height;
			const size_t len = (Len - height) * sizeof(float);
			for( ; width > 0; --width ) {
				memset(out, 0, len);
				out += Len;
			}
		}
	}
};


////////////////////////////


inline __m256 _mm256_loadu2_m128 ( float const *hiAddr, float const *loAddr )
{
  return _mm256_insertf128_ps( _mm256_castps128_ps256( _mm_loadu_ps ( loAddr ) ), _mm_loadu_ps( hiAddr ), 1 );
}
inline void _mm256_storeu2_m128( float *hiAddr, float *loAddr, __m256 data )
{
  _mm_storeu_ps ( loAddr, _mm256_castps256_ps128( data ) );
  _mm_storeu_ps ( hiAddr, _mm256_extractf128_ps( data, 1) );
}

struct CMicroKernel_6x16 : public CMicroKernelBase<6, 16> {
	static void Calculate( const float* aPtr, const float* bPtr, float* cPtr, size_t cRowSize, size_t k ) {
		_mm_prefetch( cPtr + 0 * cRowSize, _MM_HINT_T0 );
		_mm_prefetch( cPtr + 1 * cRowSize, _MM_HINT_T0 );
		_mm_prefetch( cPtr + 2 * cRowSize, _MM_HINT_T0 );
		_mm_prefetch( cPtr + 3 * cRowSize, _MM_HINT_T0 );
		_mm_prefetch( cPtr + 4 * cRowSize, _MM_HINT_T0 );
		_mm_prefetch( cPtr + 5 * cRowSize, _MM_HINT_T0 );
		__m256 c00 = _mm256_setzero_ps();
		__m256 c01 = _mm256_setzero_ps();
		__m256 c10 = _mm256_setzero_ps();
		__m256 c11 = _mm256_setzero_ps();
		__m256 c20 = _mm256_setzero_ps();
		__m256 c21 = _mm256_setzero_ps();
		__m256 c30 = _mm256_setzero_ps();
		__m256 c31 = _mm256_setzero_ps();
		__m256 c40 = _mm256_setzero_ps();
		__m256 c41 = _mm256_setzero_ps();
		__m256 c50 = _mm256_setzero_ps();
		__m256 c51 = _mm256_setzero_ps();

		__m256 b0, b1, a0, a1;

		for( ; k >= 4; k -= 4 ) {
			// Iteration 0
			_mm_prefetch( aPtr + 80, _MM_HINT_T0 );
			b0 = _mm256_loadu_ps( bPtr + 0 );
			b1 = _mm256_loadu_ps( bPtr + 8 );
			a0 = _mm256_broadcast_ss( aPtr + 0 );
			a1 = _mm256_broadcast_ss( aPtr + 1 );
			c00 = _mm256_fmadd_ps( a0, b0, c00 );
			c01 = _mm256_fmadd_ps( a0, b1, c01 );
			c10 = _mm256_fmadd_ps( a1, b0, c10 );
			c11 = _mm256_fmadd_ps( a1, b1, c11 );

			a0 = _mm256_broadcast_ss( aPtr + 2 );
			a1 = _mm256_broadcast_ss( aPtr + 3 );
			c20 = _mm256_fmadd_ps( a0, b0, c20 );
			c21 = _mm256_fmadd_ps( a0, b1, c21 );
			c30 = _mm256_fmadd_ps( a1, b0, c30 );
			c31 = _mm256_fmadd_ps( a1, b1, c31 );

			a0 = _mm256_broadcast_ss( aPtr + 4 );
			a1 = _mm256_broadcast_ss( aPtr + 5 );
			c40 = _mm256_fmadd_ps( a0, b0, c40 );
			c41 = _mm256_fmadd_ps( a0, b1, c41 );
			c50 = _mm256_fmadd_ps( a1, b0, c50 );
			c51 = _mm256_fmadd_ps( a1, b1, c51 );

			// Iteration 1
			b0 = _mm256_loadu_ps( bPtr + 16 );
			b1 = _mm256_loadu_ps( bPtr + 24 );
			a0 = _mm256_broadcast_ss( aPtr + 6 );
			a1 = _mm256_broadcast_ss( aPtr + 7 );
			c00 = _mm256_fmadd_ps( a0, b0, c00 );
			c01 = _mm256_fmadd_ps( a0, b1, c01 );
			c10 = _mm256_fmadd_ps( a1, b0, c10 );
			c11 = _mm256_fmadd_ps( a1, b1, c11 );

			a0 = _mm256_broadcast_ss( aPtr + 8 );
			a1 = _mm256_broadcast_ss( aPtr + 9 );
			c20 = _mm256_fmadd_ps( a0, b0, c20 );
			c21 = _mm256_fmadd_ps( a0, b1, c21 );
			c30 = _mm256_fmadd_ps( a1, b0, c30 );
			c31 = _mm256_fmadd_ps( a1, b1, c31 );

			a0 = _mm256_broadcast_ss( aPtr + 10 );
			a1 = _mm256_broadcast_ss( aPtr + 11 );
			c40 = _mm256_fmadd_ps( a0, b0, c40 );
			c41 = _mm256_fmadd_ps( a0, b1, c41 );
			c50 = _mm256_fmadd_ps( a1, b0, c50 );
			c51 = _mm256_fmadd_ps( a1, b1, c51 );

			// Iteration 2
			_mm_prefetch( aPtr + 76, _MM_HINT_T0 );
			b0 = _mm256_loadu_ps( bPtr + 32 );
			b1 = _mm256_loadu_ps( bPtr + 40 );
			a0 = _mm256_broadcast_ss( aPtr +12 );
			a1 = _mm256_broadcast_ss( aPtr + 13 );
			c00 = _mm256_fmadd_ps( a0, b0, c00 );
			c01 = _mm256_fmadd_ps( a0, b1, c01 );
			c10 = _mm256_fmadd_ps( a1, b0, c10 );
			c11 = _mm256_fmadd_ps( a1, b1, c11 );

			a0 = _mm256_broadcast_ss( aPtr + 14 );
			a1 = _mm256_broadcast_ss( aPtr + 15 );
			c20 = _mm256_fmadd_ps( a0, b0, c20 );
			c21 = _mm256_fmadd_ps( a0, b1, c21 );
			c30 = _mm256_fmadd_ps( a1, b0, c30 );
			c31 = _mm256_fmadd_ps( a1, b1, c31 );

			a0 = _mm256_broadcast_ss( aPtr + 16 );
			a1 = _mm256_broadcast_ss( aPtr + 17 );
			c40 = _mm256_fmadd_ps( a0, b0, c40 );
			c41 = _mm256_fmadd_ps( a0, b1, c41 );
			c50 = _mm256_fmadd_ps( a1, b0, c50 );
			c51 = _mm256_fmadd_ps( a1, b1, c51 );

			// Iteration 3
			b0 = _mm256_loadu_ps( bPtr + 48 );
			b1 = _mm256_loadu_ps( bPtr + 56 );
			a0 = _mm256_broadcast_ss( aPtr + 18 );
			a1 = _mm256_broadcast_ss( aPtr + 19 );
			c00 = _mm256_fmadd_ps( a0, b0, c00 );
			c01 = _mm256_fmadd_ps( a0, b1, c01 );
			c10 = _mm256_fmadd_ps( a1, b0, c10 );
			c11 = _mm256_fmadd_ps( a1, b1, c11 );

			a0 = _mm256_broadcast_ss( aPtr + 20 );
			a1 = _mm256_broadcast_ss( aPtr + 21 );
			c20 = _mm256_fmadd_ps( a0, b0, c20 );
			c21 = _mm256_fmadd_ps( a0, b1, c21 );
			c30 = _mm256_fmadd_ps( a1, b0, c30 );
			c31 = _mm256_fmadd_ps( a1, b1, c31 );

			a0 = _mm256_broadcast_ss( aPtr + 22 );
			a1 = _mm256_broadcast_ss( aPtr + 23 );
			c40 = _mm256_fmadd_ps( a0, b0, c40 );
			c41 = _mm256_fmadd_ps( a0, b1, c41 );
			c50 = _mm256_fmadd_ps( a1, b0, c50 );
			c51 = _mm256_fmadd_ps( a1, b1, c51 );


			bPtr += 64; aPtr += 24;
		}

		for( ; k > 0; k-- ) {
			b0 = _mm256_loadu_ps( bPtr + 0 );
			b1 = _mm256_loadu_ps( bPtr + 8 );
			a0 = _mm256_broadcast_ss( aPtr + 0 );
			a1 = _mm256_broadcast_ss( aPtr + 1 );
			c00 = _mm256_fmadd_ps( a0, b0, c00 );
			c01 = _mm256_fmadd_ps( a0, b1, c01 );
			c10 = _mm256_fmadd_ps( a1, b0, c10 );
			c11 = _mm256_fmadd_ps( a1, b1, c11 );

			a0 = _mm256_broadcast_ss( aPtr + 2 );
			a1 = _mm256_broadcast_ss( aPtr + 3 );
			c20 = _mm256_fmadd_ps( a0, b0, c20 );
			c21 = _mm256_fmadd_ps( a0, b1, c21 );
			c30 = _mm256_fmadd_ps( a1, b0, c30 );
			c31 = _mm256_fmadd_ps( a1, b1, c31 );

			a0 = _mm256_broadcast_ss( aPtr + 4 );
			a1 = _mm256_broadcast_ss( aPtr + 5 );
			c40 = _mm256_fmadd_ps( a0, b0, c40 );
			c41 = _mm256_fmadd_ps( a0, b1, c41 );
			c50 = _mm256_fmadd_ps( a1, b0, c50 );
			c51 = _mm256_fmadd_ps( a1, b1, c51 );

			bPtr += 16; aPtr += 6;
		}

		_mm256_storeu_ps( cPtr, _mm256_add_ps( c00, _mm256_loadu_ps( cPtr ) ) );
		_mm256_storeu_ps( cPtr + 8, _mm256_add_ps( c01, _mm256_loadu_ps( cPtr + 8 ) ) );
		cPtr += cRowSize;
		_mm256_storeu_ps( cPtr, _mm256_add_ps( c10, _mm256_loadu_ps( cPtr ) ) );
		_mm256_storeu_ps( cPtr + 8, _mm256_add_ps( c11, _mm256_loadu_ps( cPtr + 8 ) ) );
		cPtr += cRowSize;
		_mm256_storeu_ps( cPtr, _mm256_add_ps( c20, _mm256_loadu_ps( cPtr ) ) );
		_mm256_storeu_ps( cPtr + 8, _mm256_add_ps( c21, _mm256_loadu_ps( cPtr + 8 ) ) );
		cPtr += cRowSize;
		_mm256_storeu_ps( cPtr, _mm256_add_ps( c30, _mm256_loadu_ps( cPtr ) ) );
		_mm256_storeu_ps( cPtr + 8, _mm256_add_ps( c31, _mm256_loadu_ps( cPtr + 8 ) ) );
		cPtr += cRowSize;
		_mm256_storeu_ps( cPtr, _mm256_add_ps( c40, _mm256_loadu_ps( cPtr ) ) );
		_mm256_storeu_ps( cPtr + 8, _mm256_add_ps( c41, _mm256_loadu_ps( cPtr + 8 ) ) );
		cPtr += cRowSize;
		_mm256_storeu_ps( cPtr, _mm256_add_ps( c50, _mm256_loadu_ps( cPtr ) ) );
		_mm256_storeu_ps( cPtr + 8, _mm256_add_ps( c51, _mm256_loadu_ps( cPtr + 8 ) ) );
	}
};

struct CMicroKernel_96x1 : public CMicroKernelBase<96, 1> {
	static void Calculate( const float* aPtr, const float* bPtr, float* cPtr, size_t cRowSize, size_t k ) {
				__m256 c0 = _mm256_setzero_ps();
				__m256 c1 = _mm256_setzero_ps();
				__m256 c2 = _mm256_setzero_ps();
				__m256 c3 = _mm256_setzero_ps();
				__m256 c4 = _mm256_setzero_ps();
				__m256 c5 = _mm256_setzero_ps();
				__m256 c6 = _mm256_setzero_ps();
				__m256 c7 = _mm256_setzero_ps();
				__m256 c8 = _mm256_setzero_ps();
				__m256 c9 = _mm256_setzero_ps();
				__m256 c10 = _mm256_setzero_ps();
				__m256 c11 = _mm256_setzero_ps();
				__m256 b, a0, a1, a2;
				for( int l = 0; l < k; l++ ) {
					b = _mm256_broadcast_ss( bPtr );
					a0 = _mm256_loadu_ps( aPtr + 0 );
					a1 = _mm256_loadu_ps( aPtr + 8 );
					a2 = _mm256_loadu_ps( aPtr + 16 );
					c0 = _mm256_fmadd_ps( a0, b, c0 );
					c1 = _mm256_fmadd_ps( a1, b, c1 );
					c2 = _mm256_fmadd_ps( a2, b, c2 );

					a0 = _mm256_loadu_ps( aPtr + 24 );
					a1 = _mm256_loadu_ps( aPtr + 32 );
					a2 = _mm256_loadu_ps( aPtr + 40 );
					c3 = _mm256_fmadd_ps( a0, b, c3 );
					c4 = _mm256_fmadd_ps( a1, b, c4 );
					c5 = _mm256_fmadd_ps( a2, b, c5 );

					a0 = _mm256_loadu_ps( aPtr + 48 );
					a1 = _mm256_loadu_ps( aPtr + 56 );
					a2 = _mm256_loadu_ps( aPtr + 64 );
					c6 = _mm256_fmadd_ps( a0, b, c6 );
					c7 = _mm256_fmadd_ps( a1, b, c7 );
					c8 = _mm256_fmadd_ps( a2, b, c8 );

					a0 = _mm256_loadu_ps( aPtr + 72 );
					a1 = _mm256_loadu_ps( aPtr + 80 );
					a2 = _mm256_loadu_ps( aPtr + 88 );
					c9 = _mm256_fmadd_ps( a0, b, c9 );
					c10 = _mm256_fmadd_ps( a1, b, c10 );
					c11 = _mm256_fmadd_ps( a2, b, c11 );

					bPtr++; aPtr += 96;
				}

				_mm256_storeu_ps( cPtr + 0, _mm256_add_ps( c0, _mm256_loadu_ps( cPtr + 0 ) ) );
				_mm256_storeu_ps( cPtr + 8, _mm256_add_ps( c1, _mm256_loadu_ps( cPtr + 8 ) ) );
				_mm256_storeu_ps( cPtr + 16, _mm256_add_ps( c2, _mm256_loadu_ps( cPtr + 16 ) ) );
				_mm256_storeu_ps( cPtr + 24, _mm256_add_ps( c3, _mm256_loadu_ps( cPtr + 24 ) ) );
				_mm256_storeu_ps( cPtr + 32, _mm256_add_ps( c4, _mm256_loadu_ps( cPtr + 32 ) ) );
				_mm256_storeu_ps( cPtr + 40, _mm256_add_ps( c5, _mm256_loadu_ps( cPtr + 40 ) ) );
				_mm256_storeu_ps( cPtr + 48, _mm256_add_ps( c6, _mm256_loadu_ps( cPtr + 48 ) ) );
				_mm256_storeu_ps( cPtr + 56, _mm256_add_ps( c7, _mm256_loadu_ps( cPtr + 56 ) ) );
				_mm256_storeu_ps( cPtr + 64, _mm256_add_ps( c8, _mm256_loadu_ps( cPtr + 64 ) ) );
				_mm256_storeu_ps( cPtr + 72, _mm256_add_ps( c9, _mm256_loadu_ps( cPtr + 72 ) ) );
				_mm256_storeu_ps( cPtr + 80, _mm256_add_ps( c10, _mm256_loadu_ps( cPtr + 80 ) ) );
				_mm256_storeu_ps( cPtr + 88, _mm256_add_ps( c11, _mm256_loadu_ps( cPtr + 88) ) );
	}
};

struct CMicroKernel_6x8 : public CMicroKernelBase<6, 8> {
	static void Calculate( const float* aPtr, const float* bPtr, float* cPtr, size_t cRowSize, size_t k ) {
				__m256 c0 = _mm256_setzero_ps();
				__m256 c1 = _mm256_setzero_ps();
				__m256 c2 = _mm256_setzero_ps();
				__m256 c3 = _mm256_setzero_ps();
				__m256 c4 = _mm256_setzero_ps();
				__m256 c5 = _mm256_setzero_ps();

				__m256 b, a0, a1, a2, a3, a4, a5;

				for( ; k >= 4; k -= 4 ) {
					b = _mm256_loadu_ps( bPtr );
					a0 = _mm256_broadcast_ss( aPtr );
					a1 = _mm256_broadcast_ss( aPtr + 1 );
					a2 = _mm256_broadcast_ss( aPtr + 2 );
					a3 = _mm256_broadcast_ss( aPtr + 3 );
					a4 = _mm256_broadcast_ss( aPtr + 4 );
					a5 = _mm256_broadcast_ss( aPtr + 5 );

					c0 = _mm256_fmadd_ps( a0, b, c0 );
					c1 = _mm256_fmadd_ps( a1, b, c1 );
					c2 = _mm256_fmadd_ps( a2, b, c2 );
					c3 = _mm256_fmadd_ps( a3, b, c3 );
					c4 = _mm256_fmadd_ps( a4, b, c4 );
					c5 = _mm256_fmadd_ps( a5, b, c5 );

					b = _mm256_loadu_ps( bPtr + 8 );
					a0 = _mm256_broadcast_ss( aPtr + 6 );
					a1 = _mm256_broadcast_ss( aPtr + 7 );
					a2 = _mm256_broadcast_ss( aPtr + 8 );
					a3 = _mm256_broadcast_ss( aPtr + 9 );
					a4 = _mm256_broadcast_ss( aPtr + 10 );
					a5 = _mm256_broadcast_ss( aPtr + 11 );

					c0 = _mm256_fmadd_ps( a0, b, c0 );
					c1 = _mm256_fmadd_ps( a1, b, c1 );
					c2 = _mm256_fmadd_ps( a2, b, c2 );
					c3 = _mm256_fmadd_ps( a3, b, c3 );
					c4 = _mm256_fmadd_ps( a4, b, c4 );
					c5 = _mm256_fmadd_ps( a5, b, c5 );

					b = _mm256_loadu_ps( bPtr + 16 );
					a0 = _mm256_broadcast_ss( aPtr + 12 );
					a1 = _mm256_broadcast_ss( aPtr + 13 );
					a2 = _mm256_broadcast_ss( aPtr + 14 );
					a3 = _mm256_broadcast_ss( aPtr + 15 );
					a4 = _mm256_broadcast_ss( aPtr + 16 );
					a5 = _mm256_broadcast_ss( aPtr + 17 );

					c0 = _mm256_fmadd_ps( a0, b, c0 );
					c1 = _mm256_fmadd_ps( a1, b, c1 );
					c2 = _mm256_fmadd_ps( a2, b, c2 );
					c3 = _mm256_fmadd_ps( a3, b, c3 );
					c4 = _mm256_fmadd_ps( a4, b, c4 );
					c5 = _mm256_fmadd_ps( a5, b, c5 );

					b = _mm256_loadu_ps( bPtr + 24 );
					a0 = _mm256_broadcast_ss( aPtr + 18 );
					a1 = _mm256_broadcast_ss( aPtr + 19 );
					a2 = _mm256_broadcast_ss( aPtr + 20 );
					a3 = _mm256_broadcast_ss( aPtr + 21 );
					a4 = _mm256_broadcast_ss( aPtr + 22 );
					a5 = _mm256_broadcast_ss( aPtr + 23 );

					c0 = _mm256_fmadd_ps( a0, b, c0 );
					c1 = _mm256_fmadd_ps( a1, b, c1 );
					c2 = _mm256_fmadd_ps( a2, b, c2 );
					c3 = _mm256_fmadd_ps( a3, b, c3 );
					c4 = _mm256_fmadd_ps( a4, b, c4 );
					c5 = _mm256_fmadd_ps( a5, b, c5 );

					bPtr += 32; aPtr += 24;
				}

				for( ; k > 0; k-- ) {
					b = _mm256_loadu_ps( bPtr );
					a0 = _mm256_broadcast_ss( aPtr );
					a1 = _mm256_broadcast_ss( aPtr + 1 );
					a2 = _mm256_broadcast_ss( aPtr + 2 );
					a3 = _mm256_broadcast_ss( aPtr + 3 );
					a4 = _mm256_broadcast_ss( aPtr + 4 );
					a5 = _mm256_broadcast_ss( aPtr + 5 );

					c0 = _mm256_fmadd_ps( a0, b, c0 );
					c1 = _mm256_fmadd_ps( a1, b, c1 );
					c2 = _mm256_fmadd_ps( a2, b, c2 );
					c3 = _mm256_fmadd_ps( a3, b, c3 );
					c4 = _mm256_fmadd_ps( a4, b, c4 );
					c5 = _mm256_fmadd_ps( a5, b, c5 );

					bPtr += 8; aPtr += 6;
				}

				_mm256_storeu_ps( cPtr, _mm256_add_ps( c0, _mm256_loadu_ps( cPtr ) ) );
				cPtr += cRowSize;
				_mm256_storeu_ps( cPtr, _mm256_add_ps( c1, _mm256_loadu_ps( cPtr ) ) );
				cPtr += cRowSize;
				_mm256_storeu_ps( cPtr, _mm256_add_ps( c2, _mm256_loadu_ps( cPtr ) ) );
				cPtr += cRowSize;
				_mm256_storeu_ps( cPtr, _mm256_add_ps( c3, _mm256_loadu_ps( cPtr ) ) );
				cPtr += cRowSize;
				_mm256_storeu_ps( cPtr, _mm256_add_ps( c4, _mm256_loadu_ps( cPtr ) ) );
				cPtr += cRowSize;
				_mm256_storeu_ps( cPtr, _mm256_add_ps( c5, _mm256_loadu_ps( cPtr ) ) );
				cPtr += cRowSize;
	}
};

struct CMicroKernel_6x4 : public CMicroKernelBase<6, 4> {
	static void Calculate( const float* aPtr, const float* bPtr, float* cPtr, size_t cRowSize, size_t k ) {
				__m256 c0 = _mm256_setzero_ps();
				__m256 c1 = _mm256_setzero_ps();
				__m256 c2 = _mm256_setzero_ps();

				__m256 b, b0, b1;
				__m256 a0, a1, a2, a00, a01, a02, a10, a11, a12;
				
				__m128 const * aPtrVec = reinterpret_cast<__m128 const *>( aPtr );
				
				for( ; k >= 4; k -= 4 ) {
					// Iteration 0, 1:
					b = _mm256_loadu_ps( bPtr );
					a0 = _mm256_broadcast_ps( aPtrVec++ );
					a1 = _mm256_broadcast_ps( aPtrVec++ );
					a2 = _mm256_broadcast_ps( aPtrVec++ );
					b0 = _mm256_permute2f128_ps( b, b, PERMUTE2( 0, 0 ) );
					b1 = _mm256_permute2f128_ps( b, b, PERMUTE2( 1, 1 ) );
					bPtr += 8;
					
					a00 = _mm256_permutevar_ps( a0, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					a01 = _mm256_permutevar_ps( a0, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
					a02 = _mm256_permutevar_ps( a1, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					a10 = _mm256_permutevar_ps( a1, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
					a11 = _mm256_permutevar_ps( a2, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					a12 = _mm256_permutevar_ps( a2, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );

					b = _mm256_loadu_ps( bPtr );

					c0 = _mm256_fmadd_ps( a00, b0, c0 );
					c1 = _mm256_fmadd_ps( a01, b0, c1 );
					c2 = _mm256_fmadd_ps( a02, b0, c2 );

					a0 = _mm256_broadcast_ps( aPtrVec++ );
					a1 = _mm256_broadcast_ps( aPtrVec++ );
					a2 = _mm256_broadcast_ps( aPtrVec++ );

					c0 = _mm256_fmadd_ps( a10, b1, c0 );
					c1 = _mm256_fmadd_ps( a11, b1, c1 );
					c2 = _mm256_fmadd_ps( a12, b1, c2 );

					b0 = _mm256_permute2f128_ps( b, b, PERMUTE2( 0, 0 ) );
					b1 = _mm256_permute2f128_ps( b, b, PERMUTE2( 1, 1 ) );
					bPtr += 8;

					a00 = _mm256_permutevar_ps( a0, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					a01 = _mm256_permutevar_ps( a0, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
					a02 = _mm256_permutevar_ps( a1, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					a10 = _mm256_permutevar_ps( a1, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
					a11 = _mm256_permutevar_ps( a2, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					a12 = _mm256_permutevar_ps( a2, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );

					c0 = _mm256_fmadd_ps( a00, b0, c0 );
					c1 = _mm256_fmadd_ps( a01, b0, c1 );
					c2 = _mm256_fmadd_ps( a02, b0, c2 );
					c0 = _mm256_fmadd_ps( a10, b1, c0 );
					c1 = _mm256_fmadd_ps( a11, b1, c1 );
					c2 = _mm256_fmadd_ps( a12, b1, c2 );

				}

				if( k >= 2 ) {
					k -= 2;
					b = _mm256_loadu_ps( bPtr );
					a0 = _mm256_broadcast_ps( aPtrVec++ );
					a1 = _mm256_broadcast_ps( aPtrVec++ );
					a2 = _mm256_broadcast_ps( aPtrVec++ );
					b0 = _mm256_permute2f128_ps( b, b, PERMUTE2( 0, 0 ) );
					b1 = _mm256_permute2f128_ps( b, b, PERMUTE2( 1, 1 ) );
					bPtr += 8;

					a00 = _mm256_permutevar_ps( a0, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					a01 = _mm256_permutevar_ps( a0, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
					a02 = _mm256_permutevar_ps( a1, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					a10 = _mm256_permutevar_ps( a1, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
					a11 = _mm256_permutevar_ps( a2, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					a12 = _mm256_permutevar_ps( a2, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );

					c0 = _mm256_fmadd_ps( a00, b0, c0 );
					c1 = _mm256_fmadd_ps( a01, b0, c1 );
					c2 = _mm256_fmadd_ps( a02, b0, c2 );
					c0 = _mm256_fmadd_ps( a10, b1, c0 );
					c1 = _mm256_fmadd_ps( a11, b1, c1 );
					c2 = _mm256_fmadd_ps( a12, b1, c2 );
				}

				if( k > 0 ) {
					b0 = _mm256_broadcast_ps( reinterpret_cast<__m128 const *>( bPtr ) );
					a0 = _mm256_broadcast_ps( aPtrVec++ );
					a1 = _mm256_broadcast_ps( aPtrVec++ );

					a00 = _mm256_permutevar_ps( a0, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					a01 = _mm256_permutevar_ps( a0, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
					a02 = _mm256_permutevar_ps( a1, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );

					c0 = _mm256_fmadd_ps( a00, b0, c0 );
					c1 = _mm256_fmadd_ps( a01, b0, c1 );
					c2 = _mm256_fmadd_ps( a02, b0, c2 );
				}

				_mm256_storeu2_m128( cPtr + cRowSize, cPtr, _mm256_add_ps( c0, _mm256_loadu2_m128( cPtr + cRowSize, cPtr ) ) );
				cPtr += 2 * cRowSize;
				_mm256_storeu2_m128( cPtr + cRowSize, cPtr, _mm256_add_ps( c1, _mm256_loadu2_m128( cPtr + cRowSize, cPtr ) ) );
				cPtr += 2 * cRowSize;
				_mm256_storeu2_m128( cPtr + cRowSize, cPtr, _mm256_add_ps( c2, _mm256_loadu2_m128( cPtr + cRowSize, cPtr ) ) );
	}
};

struct CMicroKernel_6x2 : public CMicroKernelBase<6, 2> {
	static void Calculate( const float* aPtr, const float* bPtr, float* cPtr, size_t cRowSize, size_t k ) {
				__m256 c0 = _mm256_setzero_ps();
				__m256 c1 = _mm256_setzero_ps();
				__m256 c2 = _mm256_setzero_ps();

				__m256 b00, b01, b02, b10, b11, b12;
				__m256 a00, a01, a02, a10, a11, a12;
				__m256i premVar = PERMUTE8( 3, 3, 2, 2, 1, 1, 0, 0 );

				// TODO: Check bPtr align
				double const * bPtrDouble = reinterpret_cast<double const *>( bPtr );
				__m128 const * aPtrVec = reinterpret_cast<__m128 const *>( aPtr );

				for( ; k >= 4; k -= 4 ) {
					//            b0k0 b1k0
					//            b0k1 b1k1
					// a0k0 a0k1  c00  c01
					// a1k0 a1k1  c10  c11
					// a2k0 a2k1  c20  c21
					// a3k0 a3k1  c30  c31
					// a4k0 a4k1  c40  c41
					// a5k0 a5k1  c50  c51

					// b: b0k0 b1k0 b0k1 b1k1 b2k0 b2k1 b3k0 b3k1
					__m256d b = _mm256_loadu_pd( bPtrDouble );
					bPtrDouble += 4;

					// a00: a0k0 a1k0 a2k0 a3k0 a0k0 a1k0 a2k0 a3k0
					a00 = _mm256_broadcast_ps( aPtrVec++ );
					// a01: a4k0 a5k0 a0k1 a1k1 a4k0 a5k0 a0k1 a1k1
					a01 = _mm256_broadcast_ps( aPtrVec++ );
					// a02: a2k1 a3k1 a4k1 a5k1 a2k1 a3k1 a4k1 a5k1
					a02 = _mm256_broadcast_ps( aPtrVec++ );

					// b00: b0k0 b1k0 b0k1 b1k1 b0k0 b1k0 b0k1 b1k1
					b00 = _mm256_castpd_ps( _mm256_permute2f128_pd( b, b, PERMUTE2( 0, 0 ) ) );
					b02 = b00;
					// b10: b2k0 b2k1 b3k0 b3k1 b2k0 b2k1 b3k0 b3k1
					b10 = _mm256_castpd_ps( _mm256_permute2f128_pd( b, b, PERMUTE2( 1, 1 ) ) );
					b12 = b10;

					// b00: b0k0 b1k0 b0k0 b1k0 b0k0 b1k0 b0k0 b1k0
					b00 = _mm256_castpd_ps( _mm256_shuffle_pd( _mm256_castps_pd( b00 ), _mm256_castps_pd( b00 ), SHUFFLE4( 0, 0, 0, 0 ) ) );
					// b02: b0k1 b1k1 b0k1 b1k1 b0k1 b1k1 b0k1 b1k1
					b02 = _mm256_castpd_ps( _mm256_shuffle_pd( _mm256_castps_pd( b02 ), _mm256_castps_pd( b02 ), SHUFFLE4( 1, 1, 1, 1 ) ) );
					b10 = _mm256_castpd_ps( _mm256_shuffle_pd( _mm256_castps_pd( b10 ), _mm256_castps_pd( b10 ), SHUFFLE4( 0, 0, 0, 0 ) ) );
					b12 = _mm256_castpd_ps( _mm256_shuffle_pd( _mm256_castps_pd( b12 ), _mm256_castps_pd( b12 ), SHUFFLE4( 1, 1, 1, 1 ) ) );

					a10 = _mm256_broadcast_ps( aPtrVec++ );
					a11 = _mm256_broadcast_ps( aPtrVec++ );
					a12 = _mm256_broadcast_ps( aPtrVec++ );

					// b10: b0k0 b1k0 b0k0 b1k0 b0k1 b1k1 b0k1 b1k1
					b01 = _mm256_blend_ps( b00, b02, BLEND8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					b11 = _mm256_blend_ps( b10, b12, BLEND8( 1, 1, 1, 1, 0, 0, 0, 0 ) );

					// a00: a0k0 a0k0 a1k0 a1k0 a2k0 a2k0 a3k0 a3k0
					a00 = _mm256_permutevar_ps( a00, premVar );
					a01 = _mm256_permutevar_ps( a01, premVar );
					a02 = _mm256_permutevar_ps( a02, premVar );
					a10 = _mm256_permutevar_ps( a10, premVar );
					a11 = _mm256_permutevar_ps( a11, premVar );
					a12 = _mm256_permutevar_ps( a12, premVar );

					// a00: a0k0 a0k0 a1k0 a1k0 a2k0 a2k0 a3k0 a3k0
					// b00: b0k0 b1k0 b0k0 b1k0 b0k0 b1k0 b0k0 b1k0
					c0 = _mm256_fmadd_ps( a00, b00, c0 );
					c1 = _mm256_fmadd_ps( a01, b01, c1 );
					c2 = _mm256_fmadd_ps( a02, b02, c2 );
					c0 = _mm256_fmadd_ps( a10, b10, c0 );
					c1 = _mm256_fmadd_ps( a11, b11, c1 );
					c2 = _mm256_fmadd_ps( a12, b12, c2 );
				}

				if( k >= 2 ) {
					k -= 2;
					b00 = _mm256_castpd_ps( _mm256_broadcast_sd( bPtrDouble++ ) );
					b02 = _mm256_castpd_ps( _mm256_broadcast_sd( bPtrDouble++ ) );
					a00 = _mm256_broadcast_ps( aPtrVec++ );
					a01 = _mm256_broadcast_ps( aPtrVec++ );
					a02 = _mm256_broadcast_ps( aPtrVec++ );

					b01 = _mm256_blend_ps( b00, b02, BLEND8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					a00 = _mm256_permutevar_ps( a00, premVar );
					a01 = _mm256_permutevar_ps( a01, premVar );
					a02 = _mm256_permutevar_ps( a02, premVar );

					c0 = _mm256_fmadd_ps( a00, b00, c0 );
					c1 = _mm256_fmadd_ps( a01, b01, c1 );
					c2 = _mm256_fmadd_ps( a02, b02, c2 );
				}

				if( k == 1 ) {
					// b00: b0k0 b1k0 b0k0 b1k0 b0k0 b1k0 b0k0 b1k0
					b00 = _mm256_castpd_ps( _mm256_broadcast_sd( bPtrDouble++ ) );
					b02 = _mm256_setzero_ps();
					// a00: a0k0 a1k0 a2k0 a3k0 a0k0 a1k0 a2k0 a3k0
					a00 = _mm256_broadcast_ps( aPtrVec++ );
					// a01: a4k0 a5k0 0 0 0 0 0 0
					a01 = _mm256_maskload_ps( reinterpret_cast<float const *>( aPtrVec ), _mm256_set_epi32( 0, 0, 0, 0, 0, 0, -1, -1 ) );

					// b01: b0k0 b1k0 b0k0 b1k0 0 0 0 0
					b01 = _mm256_blend_ps( b00, b02, BLEND8( 1, 1, 1, 1, 0, 0, 0, 0 ) );

					// a00: a0k0 a0k0 a1k0 a1k0 a2k0 a2k0 a3k0 a3k0
					a00 = _mm256_permutevar_ps( a00, premVar );
					// a00: a4k0 a4k0 a5k0 a5k0 0 0 0 0
					a01 = _mm256_permutevar_ps( a01, premVar );

					c0 = _mm256_fmadd_ps( a00, b00, c0 );
					c1 = _mm256_fmadd_ps( a01, b01, c1 );
				}

				__m256 c0t = _mm256_castpd_ps( _mm256_permute2f128_pd( _mm256_castps_pd( c1 ), _mm256_castps_pd( c2 ), PERMUTE2( 2, 1 ) ) );
				__m256 c1t = _mm256_castpd_ps( _mm256_permute2f128_pd( _mm256_castps_pd( c2 ), _mm256_castps_pd( c2 ), PERMUTE2( 4, 1 ) ) );
				c0 = _mm256_add_ps( c0, c0t );
				c1 = _mm256_add_ps( c1, c1t );


				__m256i mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0, -1, -1 );
				_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps ( c0, _mm256_maskload_ps( cPtr, mask ) ) );
				// Decrease cRowSize because _mm256_storeu2_m128 treate start address as cPtr, but we should shift left our'c' value by 2.
				cPtr += cRowSize - 2;
				mask = _mm256_set_epi32( 0, 0, 0, 0, -1, -1, 0, 0 );
				_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps ( c0, _mm256_maskload_ps( cPtr, mask ) ) );
				cPtr += cRowSize - 2;
				mask = _mm256_set_epi32( 0, 0, -1, -1, 0, 0, 0, 0 );
				_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps ( c0, _mm256_maskload_ps( cPtr, mask ) ) );
				cPtr += cRowSize - 2;
				mask = _mm256_set_epi32( -1, -1, 0, 0, 0, 0, 0, 0 );
				_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps ( c0, _mm256_maskload_ps( cPtr, mask ) ) );
				// Get back start address
				cPtr += cRowSize + 6;
				mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0, -1, -1 );
				_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps ( c1, _mm256_maskload_ps( cPtr, mask ) ) );
				cPtr += cRowSize - 2;
				mask = _mm256_set_epi32( 0, 0, 0, 0, -1, -1, 0, 0 );
				_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps ( c1, _mm256_maskload_ps( cPtr, mask ) ) );
	}
};

struct CMicroKernel_6x1 : public CMicroKernelBase<6, 1> {
	static void Calculate( const float* aPtr, const float* bPtr, float* cPtr, size_t cRowSize, size_t k ) {
				__m256 c0 = _mm256_setzero_ps();
				__m256 c1 = _mm256_setzero_ps();
				__m256 c2 = _mm256_setzero_ps();

				__m256 b00, b01, b02, b0t;
				__m256 a00, a01, a02;

				for( ; k >= 4; k -= 4 ) {
					b00 = _mm256_broadcast_ss( bPtr + 0 );
					b01 = _mm256_broadcast_ss( bPtr + 1 );
					b02 = _mm256_broadcast_ss( bPtr + 2 );
					b0t = _mm256_broadcast_ss( bPtr + 3 );

					a00 = _mm256_loadu_ps( aPtr + 0 );
					a01 = _mm256_loadu_ps( aPtr + 8 );
					a02 = _mm256_loadu_ps( aPtr + 16 );

					b00 = _mm256_blend_ps( b00, b01, BLEND8( 1, 1, 0, 0, 0, 0, 0, 0 ) );
					b01 = _mm256_blend_ps( b01, b02, BLEND8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
					b02 = _mm256_blend_ps( b02, b0t, BLEND8( 1, 1, 1, 1, 1, 1, 0, 0 ) );

					c0 = _mm256_fmadd_ps( a00, b00, c0 );
					c1 = _mm256_fmadd_ps( a01, b01, c1 );
					c2 = _mm256_fmadd_ps( a02, b02, c2 );

					bPtr += 4; aPtr += 24;
				}

				if( k >= 2 ) {
					k -= 2;
					__m256i mask = _mm256_set_epi64x( 0, 0, -1, -1 );
					b00 = _mm256_broadcast_ss( bPtr + 0 );
					b01 = _mm256_broadcast_ss( bPtr + 1 );

					a00 = _mm256_loadu_ps( aPtr + 0 );

					b00 = _mm256_blend_ps( b00, b01, BLEND8( 1, 1, 0, 0, 0, 0, 0, 0 ) );

					a01 = _mm256_castps128_ps256( _mm_loadu_ps( aPtr + 8 ) );
					a01 = _mm256_and_ps( a01, _mm256_castsi256_ps( mask ) );

					c0 = _mm256_fmadd_ps( a00, b00, c0 );
					c1 = _mm256_fmadd_ps( a01, b01, c1 );

					bPtr += 2; aPtr += 12;
				}

				if( k == 1 ) {
					__m256i mask = _mm256_set_epi64x( 0, -1, -1, -1 );
					b00 = _mm256_broadcast_ss( bPtr );
					a00 = _mm256_loadu_ps( aPtr );
					b00 = _mm256_and_ps( b00, _mm256_castsi256_ps( mask ) );
					c0 = _mm256_fmadd_ps( a00, b00, c0 );
				}

				// c0: a0 a1 a2 a3 a4 a5 b0 b1
				// c1: b2 b3 b4 c5 c0 c1 c2 c3
				// c2: c4 c5 d0 d1 d2 d3 d4 d5
				// c1t: a4 a5 b0 b1 b2 b3 b4 c5
				__m256 c1t = _mm256_castpd_ps( _mm256_permute2f128_pd( _mm256_castps_pd( c0 ), _mm256_castps_pd( c1 ), PERMUTE2( 2, 1 ) ) );
				// c3t: d2 d3 d4 d5 d2 d3 d4 d5
				__m256 c3t = _mm256_castpd_ps( _mm256_permute2f128_pd( _mm256_castps_pd( c2 ), _mm256_castps_pd( c2 ), PERMUTE2( 1, 1 ) ) );
				// c2t: c0 c1 c2 c3 c4 c5 d0 d1
				__m256 c2t = _mm256_castpd_ps( _mm256_permute2f128_pd( _mm256_castps_pd( c1 ), _mm256_castps_pd( c2 ), PERMUTE2( 2, 1 ) ) );
				// c1t: b0 b1 b2 b3 b4 b5 c0 c1
				c1t = _mm256_castpd_ps ( _mm256_shuffle_pd( _mm256_castps_pd( c1t ), _mm256_castps_pd( c1 ), SHUFFLE4( 0, 1, 0, 1 ) ) );
				// c3t: d0 d1 d2 d3 d4 d5 d2 d3
				c3t = _mm256_castpd_ps ( _mm256_shuffle_pd( _mm256_castps_pd( c2 ), _mm256_castps_pd( c3t ), SHUFFLE4( 0, 1, 0, 1 ) ) );

				c0 = _mm256_add_ps( c0, c1t );
				c2t = _mm256_add_ps( c2t, c3t );
				c0 = _mm256_add_ps( c0, c2t );

				// Decrease cRowSize because _mm256_storeu2_m128 treate start address as cPtr, but we should shift left our'c' value by 1.
				cRowSize--;
				__m256i mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0, 0, -1 );
				_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps (c0, _mm256_maskload_ps( cPtr, mask ) ) );
				cPtr += cRowSize;
				mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0, -1, 0 );
				_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps (c0, _mm256_maskload_ps( cPtr, mask ) ) );
				cPtr += cRowSize;
				mask = _mm256_set_epi32( 0, 0, 0, 0, 0, -1, 0, 0 );
				_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps (c0, _mm256_maskload_ps( cPtr, mask ) ) );
				cPtr += cRowSize;
				mask = _mm256_set_epi32( 0, 0, 0, 0, -1, 0, 0, 0 );
				_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps (c0, _mm256_maskload_ps( cPtr, mask ) ) );
				cPtr += cRowSize;
				mask = _mm256_set_epi32( 0, 0, 0, -1, 0, 0, 0, 0 );
				_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps (c0, _mm256_maskload_ps( cPtr, mask ) ) );
				cPtr += cRowSize;
				mask = _mm256_set_epi32( 0, 0, -1, 0, 0, 0, 0, 0 );
				_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps (c0, _mm256_maskload_ps( cPtr, mask ) ) );
	}
};

using CKernelCombi = CKernelCombineHorizontal<CMicroKernel_6x16, CMicroKernel_6x8, CMicroKernel_6x4, CMicroKernel_6x2/*, CMicroKernel_6x1*/>;

