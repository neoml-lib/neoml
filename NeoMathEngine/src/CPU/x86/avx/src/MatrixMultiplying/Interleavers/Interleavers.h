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

#pragma once

#include <AvxCommon.h>
#include <Interleavers/InterleaverBase.h>

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

//// Prepare and transpose the matrix
template<>
struct CInterleaverBase<true, 16> {
	static void Prepare( float* out, const float* in, size_t stride, size_t width, size_t height )
	{
		const int Len = 16;

		const size_t iStep = stride * Len;
		const size_t oStep = width * Len;

		__m256 a, b, c, d, e, f, j, h;
		for( ; height >= Len; height -= Len ) {
			const float* tempIn = in;
			float* tempOut = out;
			size_t tempWidth = width;
			for( ; tempWidth >= 8; tempWidth -= 8 ) {
				a = _mm256_loadu_ps( tempIn );
				b = _mm256_loadu_ps( tempIn + stride );
				c = _mm256_loadu_ps( tempIn + 2 * stride );
				d = _mm256_loadu_ps( tempIn + 3 * stride );
				e = _mm256_loadu_ps( tempIn + 4 * stride );
				f = _mm256_loadu_ps( tempIn + 5 * stride );
				j = _mm256_loadu_ps( tempIn + 6 * stride );
				h = _mm256_loadu_ps( tempIn + 7 * stride );

				{
				// ab0145: a0 b0 a1 b1 a4 b4 a5 b5
				__m256d ab0145 = _mm256_castps_pd( _mm256_unpacklo_ps( a, b ) );
				__m256d cd0145 = _mm256_castps_pd( _mm256_unpacklo_ps( c, d ) );
				__m256d ef0145 = _mm256_castps_pd( _mm256_unpacklo_ps( e, f ) );
				__m256d jh0145 = _mm256_castps_pd( _mm256_unpacklo_ps( j, h ) );

				// abcd04: a0 b0 c0 d0 a4 b4 c4 d4
				__m256d abcd04 = _mm256_unpacklo_pd( ab0145, cd0145 );
				__m256d efjh04 = _mm256_unpacklo_pd( ef0145, jh0145 );
				// abcd15: a1 b1 c1 d1 a5 b5 c5 d5
				__m256d abcd15 = _mm256_unpackhi_pd( ab0145, cd0145 );
				__m256d efjh15 = _mm256_unpackhi_pd( ef0145, jh0145 );

				// abcd0: a0 b0 c0 d0 e0 f0 j0 h0
				__m256d abcdefjh0 = _mm256_permute2f128_pd ( abcd04, efjh04, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh4 = _mm256_permute2f128_pd ( abcd04, efjh04, PERMUTE2( 3, 1 ) );
				__m256d abcdefjh1 = _mm256_permute2f128_pd ( abcd15, efjh15, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh5 = _mm256_permute2f128_pd ( abcd15, efjh15, PERMUTE2( 3, 1 ) );

				// Store 0, 1, 5, 4
				_mm256_storeu_ps( tempOut, _mm256_castpd_ps( abcdefjh0 ) );
				_mm256_storeu_ps( tempOut + Len, _mm256_castpd_ps( abcdefjh1 ) );
				_mm256_storeu_ps( tempOut + 4 * Len, _mm256_castpd_ps( abcdefjh4 ) );
				_mm256_storeu_ps( tempOut + 5 * Len, _mm256_castpd_ps( abcdefjh5 ) );

				////////////////////////////////////////////
				// Same permutations as above
				__m256d ab2367 = _mm256_castps_pd( _mm256_unpackhi_ps( a, b ) );
				__m256d cd2367 = _mm256_castps_pd( _mm256_unpackhi_ps( c, d ) );
				__m256d ef2367 = _mm256_castps_pd( _mm256_unpackhi_ps( e, f ) );
				__m256d jh2367 = _mm256_castps_pd( _mm256_unpackhi_ps( j, h ) );

				__m256d abcd26 = _mm256_unpacklo_pd( ab2367, cd2367 );
				__m256d efjh26 = _mm256_unpacklo_pd( ef2367, jh2367 );
				__m256d abcd37 = _mm256_unpackhi_pd( ab2367, cd2367 );
				__m256d efjh37 = _mm256_unpackhi_pd( ef2367, jh2367 );

				__m256d abcdefjh2 = _mm256_permute2f128_pd ( abcd26, efjh26, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh6 = _mm256_permute2f128_pd ( abcd26, efjh26, PERMUTE2( 3, 1 ) );
				__m256d abcdefjh3 = _mm256_permute2f128_pd ( abcd37, efjh37, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh7 = _mm256_permute2f128_pd ( abcd37, efjh37, PERMUTE2( 3, 1 ) );

				// Store 0, 1, 5, 4
				_mm256_storeu_ps( tempOut + 2 * Len, _mm256_castpd_ps( abcdefjh2 ) );
				_mm256_storeu_ps( tempOut + 3 * Len, _mm256_castpd_ps( abcdefjh3 )  );
				_mm256_storeu_ps( tempOut + 6 * Len, _mm256_castpd_ps( abcdefjh6 ) );
				_mm256_storeu_ps( tempOut + 7 * Len, _mm256_castpd_ps( abcdefjh7 ) );
				}

				a = _mm256_loadu_ps( tempIn + 8 * stride );
				b = _mm256_loadu_ps( tempIn + 9 * stride );
				c = _mm256_loadu_ps( tempIn + 10 * stride );
				d = _mm256_loadu_ps( tempIn + 11 * stride );
				e = _mm256_loadu_ps( tempIn + 12 * stride );
				f = _mm256_loadu_ps( tempIn + 13 * stride );
				j = _mm256_loadu_ps( tempIn + 14 * stride );
				h = _mm256_loadu_ps( tempIn + 15 * stride );

				{
				// ab0145: a0 b0 a1 b1 a4 b4 a5 b5
				__m256d ab0145 = _mm256_castps_pd( _mm256_unpacklo_ps( a, b ) );
				__m256d cd0145 = _mm256_castps_pd( _mm256_unpacklo_ps( c, d ) );
				__m256d ef0145 = _mm256_castps_pd( _mm256_unpacklo_ps( e, f ) );
				__m256d jh0145 = _mm256_castps_pd( _mm256_unpacklo_ps( j, h ) );

				// abcd04: a0 b0 c0 d0 a4 b4 c4 d4
				__m256d abcd04 = _mm256_unpacklo_pd( ab0145, cd0145 );
				__m256d efjh04 = _mm256_unpacklo_pd( ef0145, jh0145 );
				// abcd15: a1 b1 c1 d1 a5 b5 c5 d5
				__m256d abcd15 = _mm256_unpackhi_pd( ab0145, cd0145 );
				__m256d efjh15 = _mm256_unpackhi_pd( ef0145, jh0145 );

				// abcd0: a0 b0 c0 d0 e0 f0 j0 h0
				__m256d abcdefjh0 = _mm256_permute2f128_pd ( abcd04, efjh04, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh4 = _mm256_permute2f128_pd ( abcd04, efjh04, PERMUTE2( 3, 1 ) );
				__m256d abcdefjh1 = _mm256_permute2f128_pd ( abcd15, efjh15, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh5 = _mm256_permute2f128_pd ( abcd15, efjh15, PERMUTE2( 3, 1 ) );

				// Store 0, 1, 5, 4
				_mm256_storeu_ps( tempOut + 8, _mm256_castpd_ps( abcdefjh0 ) );
				_mm256_storeu_ps( tempOut + 8 + Len, _mm256_castpd_ps( abcdefjh1 ) );
				_mm256_storeu_ps( tempOut + 8 + 4 * Len, _mm256_castpd_ps( abcdefjh4 ) );
				_mm256_storeu_ps( tempOut + 8 + 5 * Len, _mm256_castpd_ps( abcdefjh5 ) );

				////////////////////////////////////////////
				// Same permutations as above
				__m256d ab2367 = _mm256_castps_pd( _mm256_unpackhi_ps( a, b ) );
				__m256d cd2367 = _mm256_castps_pd( _mm256_unpackhi_ps( c, d ) );
				__m256d ef2367 = _mm256_castps_pd( _mm256_unpackhi_ps( e, f ) );
				__m256d jh2367 = _mm256_castps_pd( _mm256_unpackhi_ps( j, h ) );

				__m256d abcd26 = _mm256_unpacklo_pd( ab2367, cd2367 );
				__m256d efjh26 = _mm256_unpacklo_pd( ef2367, jh2367 );
				__m256d abcd37 = _mm256_unpackhi_pd( ab2367, cd2367 );
				__m256d efjh37 = _mm256_unpackhi_pd( ef2367, jh2367 );

				__m256d abcdefjh2 = _mm256_permute2f128_pd ( abcd26, efjh26, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh6 = _mm256_permute2f128_pd ( abcd26, efjh26, PERMUTE2( 3, 1 ) );
				__m256d abcdefjh3 = _mm256_permute2f128_pd ( abcd37, efjh37, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh7 = _mm256_permute2f128_pd ( abcd37, efjh37, PERMUTE2( 3, 1 ) );

				// Store 0, 1, 5, 4
				_mm256_storeu_ps( tempOut + 8 + 2 * Len, _mm256_castpd_ps( abcdefjh2 ) );
				_mm256_storeu_ps( tempOut + 8 + 3 * Len, _mm256_castpd_ps( abcdefjh3 )  );
				_mm256_storeu_ps( tempOut + 8 + 6 * Len, _mm256_castpd_ps( abcdefjh6 ) );
				_mm256_storeu_ps( tempOut + 8 + 7 * Len, _mm256_castpd_ps( abcdefjh7 ) );
				}
				tempIn += 8;
				tempOut += 128;
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

//// Prepare and transpose the matrix
template<>
struct CInterleaverBase<true, 8> {
	static void Prepare( float* out, const float* in, size_t stride, size_t width, size_t height )
	{
		const int Len = 8;

		const size_t iStep = stride * Len;
		const size_t oStep = width * Len;

		__m256 a, b, c, d, e, f, j, h;
		for( ; height >= Len; height -= Len ) {
			const float* tempIn = in;
			float* tempOut = out;
			size_t tempWidth = width;
			for( ; tempWidth >= 8; tempWidth -= 8 ) {
				a = _mm256_loadu_ps( tempIn );
				b = _mm256_loadu_ps( tempIn + stride );
				c = _mm256_loadu_ps( tempIn + 2 * stride );
				d = _mm256_loadu_ps( tempIn + 3 * stride );
				e = _mm256_loadu_ps( tempIn + 4 * stride );
				f = _mm256_loadu_ps( tempIn + 5 * stride );
				j = _mm256_loadu_ps( tempIn + 6 * stride );
				h = _mm256_loadu_ps( tempIn + 7 * stride );

				// ab0145: a0 b0 a1 b1 a4 b4 a5 b5
				__m256d ab0145 = _mm256_castps_pd( _mm256_unpacklo_ps( a, b ) );
				__m256d cd0145 = _mm256_castps_pd( _mm256_unpacklo_ps( c, d ) );
				__m256d ef0145 = _mm256_castps_pd( _mm256_unpacklo_ps( e, f ) );
				__m256d jh0145 = _mm256_castps_pd( _mm256_unpacklo_ps( j, h ) );

				// abcd04: a0 b0 c0 d0 a4 b4 c4 d4
				__m256d abcd04 = _mm256_unpacklo_pd( ab0145, cd0145 );
				__m256d efjh04 = _mm256_unpacklo_pd( ef0145, jh0145 );
				// abcd15: a1 b1 c1 d1 a5 b5 c5 d5
				__m256d abcd15 = _mm256_unpackhi_pd( ab0145, cd0145 );
				__m256d efjh15 = _mm256_unpackhi_pd( ef0145, jh0145 );

				// abcd0: a0 b0 c0 d0 e0 f0 j0 h0
				__m256d abcdefjh0 = _mm256_permute2f128_pd ( abcd04, efjh04, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh4 = _mm256_permute2f128_pd ( abcd04, efjh04, PERMUTE2( 3, 1 ) );
				__m256d abcdefjh1 = _mm256_permute2f128_pd ( abcd15, efjh15, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh5 = _mm256_permute2f128_pd ( abcd15, efjh15, PERMUTE2( 3, 1 ) );

				// Store 0, 1, 5, 4
				_mm256_storeu_ps( tempOut, _mm256_castpd_ps( abcdefjh0 ) );
				_mm256_storeu_ps( tempOut + Len, _mm256_castpd_ps( abcdefjh1 ) );
				_mm256_storeu_ps( tempOut + 4 * Len, _mm256_castpd_ps( abcdefjh4 ) );
				_mm256_storeu_ps( tempOut + 5 * Len, _mm256_castpd_ps( abcdefjh5 ) );

				////////////////////////////////////////////
				// Same permutations as above
				__m256d ab2367 = _mm256_castps_pd( _mm256_unpackhi_ps( a, b ) );
				__m256d cd2367 = _mm256_castps_pd( _mm256_unpackhi_ps( c, d ) );
				__m256d ef2367 = _mm256_castps_pd( _mm256_unpackhi_ps( e, f ) );
				__m256d jh2367 = _mm256_castps_pd( _mm256_unpackhi_ps( j, h ) );

				__m256d abcd26 = _mm256_unpacklo_pd( ab2367, cd2367 );
				__m256d efjh26 = _mm256_unpacklo_pd( ef2367, jh2367 );
				__m256d abcd37 = _mm256_unpackhi_pd( ab2367, cd2367 );
				__m256d efjh37 = _mm256_unpackhi_pd( ef2367, jh2367 );

				__m256d abcdefjh2 = _mm256_permute2f128_pd ( abcd26, efjh26, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh6 = _mm256_permute2f128_pd ( abcd26, efjh26, PERMUTE2( 3, 1 ) );
				__m256d abcdefjh3 = _mm256_permute2f128_pd ( abcd37, efjh37, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh7 = _mm256_permute2f128_pd ( abcd37, efjh37, PERMUTE2( 3, 1 ) );

				// Store 0, 1, 5, 4
				_mm256_storeu_ps( tempOut + 2 * Len, _mm256_castpd_ps( abcdefjh2 ) );
				_mm256_storeu_ps( tempOut + 3 * Len, _mm256_castpd_ps( abcdefjh3 )  );
				_mm256_storeu_ps( tempOut + 6 * Len, _mm256_castpd_ps( abcdefjh6 ) );
				_mm256_storeu_ps( tempOut + 7 * Len, _mm256_castpd_ps( abcdefjh7 ) );

				tempIn += 8;
				tempOut += 64;
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

//// Prepare and transpose the matrix
template<>
struct CInterleaverBase<true, 6> {
	static void Prepare( float* out, const float* in, size_t stride, size_t width, size_t height )
	{
		const int Len = 6;

		const size_t iStep = stride * Len;
		const size_t oStep = width * Len;

		__m256 a, b, c, d, e, f;
		__m256i storeMask = _mm256_set_epi32( 0, 0, -1, -1, -1, -1, -1, -1 );
		for( ; height >= Len; height -= Len ) {
			const float* tempIn = in;
			float* tempOut = out;
			size_t tempWidth = width;
			for( ; tempWidth >= 8; tempWidth -= 8 ) {
				a = _mm256_loadu_ps( tempIn );
				b = _mm256_loadu_ps( tempIn + stride );
				c = _mm256_loadu_ps( tempIn + 2 * stride );
				d = _mm256_loadu_ps( tempIn + 3 * stride );
				e = _mm256_loadu_ps( tempIn + 4 * stride );
				f = _mm256_loadu_ps( tempIn + 5 * stride );

				// ab0145: a0 b0 a1 b1 a4 b4 a5 b5
				__m256d ab0145 = _mm256_castps_pd( _mm256_unpacklo_ps( a, b ) );
				__m256d cd0145 = _mm256_castps_pd( _mm256_unpacklo_ps( c, d ) );
				__m256d ef0145 = _mm256_castps_pd( _mm256_unpacklo_ps( e, f ) );

				// abcd04: a0 b0 c0 d0 a4 b4 c4 d4
				__m256d abcd04 = _mm256_unpacklo_pd( ab0145, cd0145 );
				 // No matter what we ill use as second operand, because it will be dropped later
				__m256d efjh04 = _mm256_unpacklo_pd( ef0145, ef0145 );
				// abcd15: a1 b1 c1 d1 a5 b5 c5 d5
				__m256d abcd15 = _mm256_unpackhi_pd( ab0145, cd0145 );
				// No matter what we ill use as second operand, because it will be dropped later
				__m256d efjh15 = _mm256_unpackhi_pd( ef0145, ef0145 );

				// abcd0: a0 b0 c0 d0 e0 f0 j0 h0
				__m256d abcdefjh0 = _mm256_permute2f128_pd ( abcd04, efjh04, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh4 = _mm256_permute2f128_pd ( abcd04, efjh04, PERMUTE2( 3, 1 ) );
				__m256d abcdefjh1 = _mm256_permute2f128_pd ( abcd15, efjh15, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh5 = _mm256_permute2f128_pd ( abcd15, efjh15, PERMUTE2( 3, 1 ) );

				// Store 0, 1, 5, 4
				_mm256_maskstore_ps( tempOut, storeMask, _mm256_castpd_ps( abcdefjh0 ) );
				_mm256_maskstore_ps( tempOut + Len, storeMask, _mm256_castpd_ps( abcdefjh1 ) );
				_mm256_maskstore_ps( tempOut + 4 * Len, storeMask, _mm256_castpd_ps( abcdefjh4 ) );
				_mm256_maskstore_ps( tempOut + 5 * Len, storeMask, _mm256_castpd_ps( abcdefjh5 ) );

				////////////////////////////////////////////
				// Same permutations as above
				__m256d ab2367 = _mm256_castps_pd( _mm256_unpackhi_ps( a, b ) );
				__m256d cd2367 = _mm256_castps_pd( _mm256_unpackhi_ps( c, d ) );
				__m256d ef2367 = _mm256_castps_pd( _mm256_unpackhi_ps( e, f ) );

				__m256d abcd26 = _mm256_unpacklo_pd( ab2367, cd2367 );
				__m256d efjh26 = _mm256_unpacklo_pd( ef2367, ef2367 );
				__m256d abcd37 = _mm256_unpackhi_pd( ab2367, cd2367 );
				__m256d efjh37 = _mm256_unpackhi_pd( ef2367, ef2367 );

				__m256d abcdefjh2 = _mm256_permute2f128_pd ( abcd26, efjh26, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh6 = _mm256_permute2f128_pd ( abcd26, efjh26, PERMUTE2( 3, 1 ) );
				__m256d abcdefjh3 = _mm256_permute2f128_pd ( abcd37, efjh37, PERMUTE2( 2, 0 ) );
				__m256d abcdefjh7 = _mm256_permute2f128_pd ( abcd37, efjh37, PERMUTE2( 3, 1 ) );

				// Store 0, 1, 5, 4
				_mm256_maskstore_ps( tempOut + 2 * Len, storeMask, _mm256_castpd_ps( abcdefjh2 ) );
				_mm256_maskstore_ps( tempOut + 3 * Len, storeMask, _mm256_castpd_ps( abcdefjh3 )  );
				_mm256_maskstore_ps( tempOut + 6 * Len, storeMask, _mm256_castpd_ps( abcdefjh6 ) );
				_mm256_maskstore_ps( tempOut + 7 * Len, storeMask, _mm256_castpd_ps( abcdefjh7 ) );

				tempIn += 8;
				tempOut += 48;
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
