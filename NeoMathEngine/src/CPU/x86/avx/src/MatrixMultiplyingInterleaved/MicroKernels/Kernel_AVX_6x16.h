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

struct CMicroKernel_6x16 : public CMicroKernelBase<6, 16> {
	static void Calculate( const float* aPtr, const float* bPtr, float* cPtr, size_t cRowSize, size_t k ) {
		_mm_prefetch( reinterpret_cast<const char*>( cPtr + 0 * cRowSize ), _MM_HINT_T0 );
		_mm_prefetch( reinterpret_cast<const char*>( cPtr + 1 * cRowSize ), _MM_HINT_T0 );
		_mm_prefetch( reinterpret_cast<const char*>( cPtr + 2 * cRowSize ), _MM_HINT_T0 );
		_mm_prefetch( reinterpret_cast<const char*>( cPtr + 3 * cRowSize ), _MM_HINT_T0 );
		_mm_prefetch( reinterpret_cast<const char*>( cPtr + 4 * cRowSize ), _MM_HINT_T0 );
		_mm_prefetch( reinterpret_cast<const char*>( cPtr + 5 * cRowSize ), _MM_HINT_T0 );
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
			//      b0   b1
			// a0   c00  c01
			// a1   c10  c11
			// a2   c20  c21
			// a3   c30  c31
			// a4   c40  c41
			// a5   c50  c51
			// Iteration 0
			_mm_prefetch(  reinterpret_cast<const char*>( aPtr + 48 ), _MM_HINT_T0 );
			// b0: b0[0-7]
			b0 = _mm256_loadu_ps( bPtr + 0 );
			// b1: b1[0-7]
			b1 = _mm256_loadu_ps( bPtr + 8 );
			// a0: a0[0] a0[0] a0[0] a0[0] a0[0] a0[0] a0[0] a0[0]
			a0 = _mm256_broadcast_ss( aPtr + 0 );
			// a1: a1[0] a1[0] a1[0] a1[0] a1[0] a1[0] a1[0] a1[0]
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
			_mm_prefetch( reinterpret_cast<const char*>( aPtr + 60 ), _MM_HINT_T0 );
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

