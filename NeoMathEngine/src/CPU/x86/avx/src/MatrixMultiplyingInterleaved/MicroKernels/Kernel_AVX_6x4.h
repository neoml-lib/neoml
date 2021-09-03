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
			a0 = _mm256_broadcast_ps( aPtrVec + 0 );
			a1 = _mm256_broadcast_ps( aPtrVec + 1 );
			a2 = _mm256_broadcast_ps( aPtrVec + 2 );
			b0 = _mm256_permute2f128_ps( b, b, PERMUTE2( 0, 0 ) );
			b1 = _mm256_permute2f128_ps( b, b, PERMUTE2( 1, 1 ) );
			_mm_prefetch( reinterpret_cast<const char*>( aPtrVec + 6 ), _MM_HINT_T0 );

			a00 = _mm256_permutevar_ps( a0, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
			a02 = _mm256_permutevar_ps( a1, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
			a11 = _mm256_permutevar_ps( a2, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
			a01 = _mm256_permutevar_ps( a0, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
			a10 = _mm256_permutevar_ps( a1, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
			a12 = _mm256_permutevar_ps( a2, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );

			b = _mm256_loadu_ps( bPtr + 8 );

			c0 = _mm256_fmadd_ps( a00, b0, c0 );
			c1 = _mm256_fmadd_ps( a01, b0, c1 );
			c2 = _mm256_fmadd_ps( a02, b0, c2 );

			a0 = _mm256_broadcast_ps( aPtrVec + 3 );
			a1 = _mm256_broadcast_ps( aPtrVec + 4 );
			a2 = _mm256_broadcast_ps( aPtrVec + 5 );

			c0 = _mm256_fmadd_ps( a10, b1, c0 );
			c1 = _mm256_fmadd_ps( a11, b1, c1 );
			c2 = _mm256_fmadd_ps( a12, b1, c2 );

			b0 = _mm256_permute2f128_ps( b, b, PERMUTE2( 0, 0 ) );
			b1 = _mm256_permute2f128_ps( b, b, PERMUTE2( 1, 1 ) );
			_mm_prefetch( reinterpret_cast<const char*>( aPtrVec + 9 ), _MM_HINT_T0 );

			a00 = _mm256_permutevar_ps( a0, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
			a02 = _mm256_permutevar_ps( a1, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
			a11 = _mm256_permutevar_ps( a2, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
			a01 = _mm256_permutevar_ps( a0, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
			a10 = _mm256_permutevar_ps( a1, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
			a12 = _mm256_permutevar_ps( a2, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );

			c0 = _mm256_fmadd_ps( a00, b0, c0 );
			c1 = _mm256_fmadd_ps( a01, b0, c1 );
			c2 = _mm256_fmadd_ps( a02, b0, c2 );
			c0 = _mm256_fmadd_ps( a10, b1, c0 );
			c1 = _mm256_fmadd_ps( a11, b1, c1 );
			c2 = _mm256_fmadd_ps( a12, b1, c2 );
			bPtr += 16;
			aPtrVec += 6;

		}

		if( k >= 2 ) {
			k -= 2;
			b = _mm256_loadu_ps( bPtr );
			a0 = _mm256_broadcast_ps( aPtrVec );
			a1 = _mm256_broadcast_ps( aPtrVec + 1 );
			a2 = _mm256_broadcast_ps( aPtrVec + 2 );
			b0 = _mm256_permute2f128_ps( b, b, PERMUTE2( 0, 0 ) );
			b1 = _mm256_permute2f128_ps( b, b, PERMUTE2( 1, 1 ) );
			bPtr += 8;

			a00 = _mm256_permutevar_ps( a0, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
			a02 = _mm256_permutevar_ps( a1, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
			a11 = _mm256_permutevar_ps( a2, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
			a01 = _mm256_permutevar_ps( a0, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
			a10 = _mm256_permutevar_ps( a1, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );
			a12 = _mm256_permutevar_ps( a2, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );

			c0 = _mm256_fmadd_ps( a00, b0, c0 );
			c1 = _mm256_fmadd_ps( a01, b0, c1 );
			c2 = _mm256_fmadd_ps( a02, b0, c2 );
			c0 = _mm256_fmadd_ps( a10, b1, c0 );
			c1 = _mm256_fmadd_ps( a11, b1, c1 );
			c2 = _mm256_fmadd_ps( a12, b1, c2 );
			aPtrVec += 3;
		}

		if( k > 0 ) {
			b0 = _mm256_broadcast_ps( reinterpret_cast<__m128 const *>( bPtr ) );
			a0 = _mm256_broadcast_ps( aPtrVec );
			a1 = _mm256_broadcast_ps( aPtrVec + 1 );

			a00 = _mm256_permutevar_ps( a0, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
			a02 = _mm256_permutevar_ps( a1, PERMUTE8( 1, 1, 1, 1, 0, 0, 0, 0 ) );
			a01 = _mm256_permutevar_ps( a0, PERMUTE8( 3, 3, 3, 3, 2, 2, 2, 2 ) );

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
