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
#include <MicroKernels/MicroKernelBase.h>

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
		_mm256_maskstore_ps( cPtr, mask, _mm256_add_ps ( c1	, _mm256_maskload_ps( cPtr, mask ) ) );
	}
};
