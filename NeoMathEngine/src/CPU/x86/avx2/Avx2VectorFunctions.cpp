/* Copyright © 2017-2023 ABBYY

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
#include <algorithm>

static constexpr int AvxBlockSize = 8;

#define AVX_LOAD_32_FLOATS(varPrefix, srcPtr) \
	__m256 varPrefix##0 = _mm256_loadu_ps( srcPtr + 0 * AvxBlockSize ); \
	__m256 varPrefix##1 = _mm256_loadu_ps( srcPtr + 1 * AvxBlockSize ); \
	__m256 varPrefix##2 = _mm256_loadu_ps( srcPtr + 2 * AvxBlockSize ); \
	__m256 varPrefix##3 = _mm256_loadu_ps( srcPtr + 3 * AvxBlockSize )

#define AVX_STORE_32_FLOATS(varPrefix, dstPtr) \
	_mm256_storeu_ps( dstPtr + 0 * AvxBlockSize, varPrefix##0 ); \
	_mm256_storeu_ps( dstPtr + 1 * AvxBlockSize, varPrefix##1 ); \
	_mm256_storeu_ps( dstPtr + 2 * AvxBlockSize, varPrefix##2 ); \
	_mm256_storeu_ps( dstPtr + 3 * AvxBlockSize, varPrefix##3 )

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

void vectorFill( float* result, float value, int vectorSize )
{
	int simdSize = vectorSize / AvxBlockSize;
	int nonSimdSize = vectorSize % AvxBlockSize;

	__m256 valueSimd = _mm256_set1_ps( value );

	while( simdSize >= 4 ) {
		_mm256_storeu_ps( result + 0 * AvxBlockSize, valueSimd );
		_mm256_storeu_ps( result + 1 * AvxBlockSize, valueSimd );
		_mm256_storeu_ps( result + 2 * AvxBlockSize, valueSimd );
		_mm256_storeu_ps( result + 3 * AvxBlockSize, valueSimd );
		result += 4 * AvxBlockSize;
		simdSize -= 4;
	}

	while( simdSize > 0 ) {
		_mm256_storeu_ps( result, valueSimd );
		result += AvxBlockSize;
		--simdSize;
	}

	for( int i = 0; i < nonSimdSize; ++i ) {
		*result++ = value;
	}
}

void vectorFill0( float* result, int vectorSize )
{
	int simdSize = vectorSize / AvxBlockSize;
	int nonSimdSize = vectorSize % AvxBlockSize;

	__m256 valueSimd = _mm256_setzero_ps();

	while( simdSize >= 4 ) {
		_mm256_storeu_ps( result + 0 * AvxBlockSize, valueSimd );
		_mm256_storeu_ps( result + 1 * AvxBlockSize, valueSimd );
		_mm256_storeu_ps( result + 2 * AvxBlockSize, valueSimd );
		_mm256_storeu_ps( result + 3 * AvxBlockSize, valueSimd );
		result += 4 * AvxBlockSize;
		simdSize -= 4;
	}

	while( simdSize > 0 ) {
		_mm256_storeu_ps( result, valueSimd );
		result += AvxBlockSize;
		--simdSize;
	}

	for( int i = 0; i < nonSimdSize; ++i ) {
		*result++ = 0;
	}
}

void vectorAdd( const float* first, const float* second, float* result, int vectorSize )
{
	int simdSize = vectorSize / AvxBlockSize;
	int nonSimdSize = vectorSize % AvxBlockSize;

	while( simdSize >= 4 ) {
		AVX_LOAD_32_FLOATS( first, first );
		AVX_LOAD_32_FLOATS( second, second );
		first0 = _mm256_add_ps( first0, second0 );
		first1 = _mm256_add_ps( first1, second1 );
		first2 = _mm256_add_ps( first2, second2 );
		first3 = _mm256_add_ps( first3, second3 );
		AVX_STORE_32_FLOATS( first, result );
		first += 4 * AvxBlockSize;
		second += 4 * AvxBlockSize;
		result += 4 * AvxBlockSize;
		simdSize -= 4;
	}

	while( simdSize > 0 ) {
		_mm256_storeu_ps( result,
			_mm256_add_ps( _mm256_loadu_ps( first ), _mm256_loadu_ps( second ) ) );
		first += AvxBlockSize;
		second += AvxBlockSize;
		result += AvxBlockSize;
		--simdSize;
	}

	for( int i = 0; i < nonSimdSize; ++i ) {
		*result++ = *first++ + *second++;
	}
}

void vectorEltwiseMultiply( const float* first, const float* second, float* result, int vectorSize )
{
	int simdSize = vectorSize / AvxBlockSize;
	int nonSimdSize = vectorSize % AvxBlockSize;

	while( simdSize >= 4 ) {
		AVX_LOAD_32_FLOATS( first, first );
		AVX_LOAD_32_FLOATS( second, second );
		first0 = _mm256_mul_ps( first0, second0 );
		first1 = _mm256_mul_ps( first1, second1 );
		first2 = _mm256_mul_ps( first2, second2 );
		first3 = _mm256_mul_ps( first3, second3 );
		AVX_STORE_32_FLOATS( first, result );
		first += 4 * AvxBlockSize;
		second += 4 * AvxBlockSize;
		result += 4 * AvxBlockSize;
		simdSize -= 4;
	}

	while( simdSize > 0 ) {
		_mm256_storeu_ps( result,
			_mm256_mul_ps( _mm256_loadu_ps( first ), _mm256_loadu_ps( second ) ) );
		first += AvxBlockSize;
		second += AvxBlockSize;
		result += AvxBlockSize;
		--simdSize;
	}

	for( int i = 0; i < nonSimdSize; ++i ) {
		*result++ = *first++ * *second++;
	}
}

void vectorEltwiseMultiplyAdd( const float* first, const float* second, float* result, int vectorSize )
{
	int simdSize = vectorSize / AvxBlockSize;
	int nonSimdSize = vectorSize % AvxBlockSize;

	while( simdSize >= 4 ) {
		AVX_LOAD_32_FLOATS( first, first );
		AVX_LOAD_32_FLOATS( second, second );
		AVX_LOAD_32_FLOATS( result, result );
		result0 = _mm256_fmadd_ps( first0, second0, result0 );
		result1 = _mm256_fmadd_ps( first1, second1, result1 );
		result2 = _mm256_fmadd_ps( first2, second2, result2 );
		result3 = _mm256_fmadd_ps( first3, second3, result3 );
		AVX_STORE_32_FLOATS( result, result );
		first += 4 * AvxBlockSize;
		second += 4 * AvxBlockSize;
		result += 4 * AvxBlockSize;
		simdSize -= 4;
	}

	while( simdSize > 0 ) {
		_mm256_storeu_ps( result,
			_mm256_fmadd_ps( _mm256_loadu_ps( first ), _mm256_loadu_ps( second ), _mm256_loadu_ps( result ) ) );
		first += AvxBlockSize;
		second += AvxBlockSize;
		result += AvxBlockSize;
		--simdSize;
	}

	for( int i = 0; i < nonSimdSize; ++i ) {
		*result++ += *first++ * *second++;
	}
}

void vectorReLU( const float* first, float* result, int vectorSize )
{
	int simdSize = vectorSize / AvxBlockSize;
	int nonSimdSize = vectorSize % AvxBlockSize;

	const __m256 zeroSimd = _mm256_setzero_ps();

	while( simdSize >= 4 ) {
		AVX_LOAD_32_FLOATS( first, first );
		first0 = _mm256_max_ps( first0, zeroSimd );
		first1 = _mm256_max_ps( first1, zeroSimd );
		first2 = _mm256_max_ps( first2, zeroSimd );
		first3 = _mm256_max_ps( first3, zeroSimd );
		AVX_STORE_32_FLOATS( first, result );
		first += 4 * AvxBlockSize;
		result += 4 * AvxBlockSize;
		simdSize -= 4;
	}

	while( simdSize > 0 ) {
		_mm256_storeu_ps( result,
			_mm256_max_ps( _mm256_loadu_ps( first ), zeroSimd ) );
		first += AvxBlockSize;
		result += AvxBlockSize;
		--simdSize;
	}

	for( int i = 0; i < nonSimdSize; ++i ) {
		*result++ = std::max<float>( *first++, 0 );
	}
}

void vectorReLU( const float* first, float* result, int vectorSize, float threshold )
{
	int simdSize = vectorSize / AvxBlockSize;
	int nonSimdSize = vectorSize % AvxBlockSize;

	const __m256 zeroSimd = _mm256_setzero_ps();
	const __m256 thresholdSimd = _mm256_set1_ps( threshold );

	while( simdSize >= 4 ) {
		AVX_LOAD_32_FLOATS( first, first );
		first0 = _mm256_min_ps( _mm256_max_ps( first0, zeroSimd ), thresholdSimd );
		first1 = _mm256_min_ps( _mm256_max_ps( first1, zeroSimd ), thresholdSimd );
		first2 = _mm256_min_ps( _mm256_max_ps( first2, zeroSimd ), thresholdSimd );
		first3 = _mm256_min_ps( _mm256_max_ps( first3, zeroSimd ), thresholdSimd );
		AVX_STORE_32_FLOATS( first, result );
		first += 4 * AvxBlockSize;
		result += 4 * AvxBlockSize;
		simdSize -= 4;
	}

	while( simdSize > 0 ) {
		_mm256_storeu_ps( result,
			_mm256_min_ps( _mm256_max_ps( _mm256_loadu_ps( first ), zeroSimd ), thresholdSimd ) );
		first += AvxBlockSize;
		result += AvxBlockSize;
		--simdSize;
	}

	for( int i = 0; i < nonSimdSize; ++i ) {
		*result++ = std::min<float>( std::max<float>( *first++, 0 ), threshold );
	}
}

void vectorHSwish( const float* first, float* result, int vectorSize )
{
	int simdSize = vectorSize / AvxBlockSize;
	int nonSimdSize = vectorSize % AvxBlockSize;

	if( simdSize > 0 ) {
		const __m256 minusThreeSimd = _mm256_set1_ps( -3.f );
		const __m256 threeSimd = _mm256_set1_ps( 3.f );
		const __m256 oneSixthSimd = _mm256_set1_ps( 1.f / 6.f );
		for( int i = 0; i < simdSize; ++i ) {
			__m256 input = _mm256_loadu_ps( first );
			__m256 middlePart = _mm256_cmp_ps( minusThreeSimd, input, _CMP_LT_OQ );
			middlePart = _mm256_and_ps( middlePart, _mm256_cmp_ps( input, threeSimd, _CMP_LT_OQ ) ); // mask for (-3; 3)
			middlePart = _mm256_and_ps( middlePart, _mm256_mul_ps( _mm256_mul_ps( input, oneSixthSimd ),
				_mm256_add_ps( input, threeSimd ) ) );
			__m256 rightPart = _mm256_cmp_ps( input, threeSimd, _CMP_GE_OQ );
			rightPart = _mm256_and_ps( rightPart, input );
			_mm256_storeu_ps( result, _mm256_add_ps( middlePart, rightPart ) );

			first += AvxBlockSize;
			result += AvxBlockSize;
		}
	}

	for( int i = 0; i < nonSimdSize; ++i ) {
		if( *first <= -3.f ) {
			*result = 0.f;
		} else if( *first >= 3.f ) {
			*result = *first;
		} else {
			*result = *first * ( *first + 3 ) / 6.f;
		}
		++result;
		++first;
	}
}

} // namespace Avx2

} // namespace NeoML

#endif
