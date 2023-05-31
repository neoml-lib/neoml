/* Copyright © 2017-2023

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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_SSE

#include "Avx2Functions.h"

#include <immintrin.h>
#include <cstdio>

static constexpr int AvxBlockSize = 8;

#define AVX_LOAD_32_FLOATS(varPrefix, srcPtr) \
	__m256 varPrefix##0 = _mm256_loadu_ps( srcPtr + 0 * AvxBlockSize ); \
	__m256 varPrefix##1 = _mm256_loadu_ps( srcPtr + 1 * AvxBlockSize ); \
	__m256 varPrefix##2 = _mm256_loadu_ps( srcPtr + 2 * AvxBlockSize ); \
	__m256 varPrefix##3 = _mm256_loadu_ps( srcPtr + 3 * AvxBlockSize );

#define AVX_STORE_32_FLOATS(varPrefix, dstPtr) \
	_mm256_storeu_ps( dstPtr + 0 * AvxBlockSize, varPrefix##0 ); \
	_mm256_storeu_ps( dstPtr + 1 * AvxBlockSize, varPrefix##1 ); \
	_mm256_storeu_ps( dstPtr + 2 * AvxBlockSize, varPrefix##2 ); \
	_mm256_storeu_ps( dstPtr + 3 * AvxBlockSize, varPrefix##3 );

namespace NeoML {

namespace Avx2 {

void dataCopy( float* dst, const float* src, int vectorSize )
{
	int simdSize = vectorSize / AvxBlockSize;
	int nonSimdSize = vectorSize % AvxBlockSize;

	while( simdSize >= 4 ) {
		AVX_LOAD_32_FLOATS( data, src );
		AVX_STORE_32_FLOATS( data, dst );
		dst += 4 * AvxBlockSize;
		src += 4 * AvxBlockSize;
		simdSize -= 4;
	}

	while( simdSize > 0 ) {
		_mm256_storeu_ps( dst, _mm256_loadu_ps( src ) );
		dst += AvxBlockSize;
		src += AvxBlockSize;
		--simdSize;
	}

	switch( nonSimdSize ) {
		case 7:
			dst[6] = src[6];
			// fall through
		case 6:
			dst[5] = src[5];
			// fall through
		case 5:
			dst[4] = src[4];
			// fall through
		case 4:
			dst[3] = src[3];
			// fall through
		case 3:
			dst[2] = src[2];
			// fall through
		case 2:
			dst[1] = src[1];
			// fall through
		case 1:
			dst[0] = src[0];
	}
}

} // namespace Avx2

} // namespace NeoML

#endif
