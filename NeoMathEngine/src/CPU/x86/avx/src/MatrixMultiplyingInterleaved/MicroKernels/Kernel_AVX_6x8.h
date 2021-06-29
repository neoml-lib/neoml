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
