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

