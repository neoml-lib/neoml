/* Copyright © 2023-2024 ABBYY

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

#include "Avx512Functions.h"

#include <immintrin.h>
#include <algorithm>

namespace NeoML {

namespace Avx512 {

static constexpr int AvxBlockSize = 16;

#define AVX512_IO_MASK( N ) \
	_cvtu32_mask16( ( 1u << N ) - 1u )


#define AVX512_LOAD_64_FLOATS( varPrefix, srcPtr ) \
	__m512 varPrefix##0 = _mm512_loadu_ps( srcPtr + 0 * AvxBlockSize ); \
	__m512 varPrefix##1 = _mm512_loadu_ps( srcPtr + 1 * AvxBlockSize ); \
	__m512 varPrefix##2 = _mm512_loadu_ps( srcPtr + 2 * AvxBlockSize ); \
	__m512 varPrefix##3 = _mm512_loadu_ps( srcPtr + 3 * AvxBlockSize )

#define AVX512_STORE_64_FLOATS( varPrefix, dstPtr ) \
	_mm512_storeu_ps( dstPtr + 0 * AvxBlockSize, varPrefix##0 ); \
	_mm512_storeu_ps( dstPtr + 1 * AvxBlockSize, varPrefix##1 ); \
	_mm512_storeu_ps( dstPtr + 2 * AvxBlockSize, varPrefix##2 ); \
	_mm512_storeu_ps( dstPtr + 3 * AvxBlockSize, varPrefix##3 )


//---------------------------------------------------------------------------------

void dataCopy( float* dst, const float* src, int vectorSize )
{
	while( vectorSize >= 4 * AvxBlockSize ) {
		AVX512_LOAD_64_FLOATS( data, src );
		AVX512_STORE_64_FLOATS( data, dst );
		dst += 4 * AvxBlockSize;
		src += 4 * AvxBlockSize;
		vectorSize -= 4 * AvxBlockSize;
	}

	while( vectorSize >= AvxBlockSize ) {
		_mm512_storeu_ps( dst, _mm512_loadu_ps( src ) );
		dst += AvxBlockSize;
		src += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
		const __mmask16 mask = AVX512_IO_MASK( vectorSize );
		_mm512_mask_storeu_ps( dst, mask, _mm512_mask_loadu_ps( _mm512_setzero_ps(), mask, src ) );
	}
}

void vectorFill( float* result, int vectorSize, float value )
{
	const __m512 valueSimd = _mm512_set1_ps( value );
	while( vectorSize >= 4 * AvxBlockSize ) {
		_mm512_storeu_ps( result + 0 * AvxBlockSize, valueSimd );
		_mm512_storeu_ps( result + 1 * AvxBlockSize, valueSimd );
		_mm512_storeu_ps( result + 2 * AvxBlockSize, valueSimd );
		_mm512_storeu_ps( result + 3 * AvxBlockSize, valueSimd );
		result += 4 * AvxBlockSize;
		vectorSize -= 4 * AvxBlockSize;
	}

	while( vectorSize >= AvxBlockSize ) {
		_mm512_storeu_ps( result, valueSimd );
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
		_mm512_mask_storeu_ps( result, AVX512_IO_MASK( vectorSize ), valueSimd );
	}
}

void vectorAdd( const float* first, const float* second, float* result, int vectorSize )
{
	while( vectorSize >= 4 * AvxBlockSize ) {
		AVX512_LOAD_64_FLOATS( first, first );
		AVX512_LOAD_64_FLOATS( second, second );
		first0 = _mm512_add_ps( first0, second0 );
		first1 = _mm512_add_ps( first1, second1 );
		first2 = _mm512_add_ps( first2, second2 );
		first3 = _mm512_add_ps( first3, second3 );
		AVX512_STORE_64_FLOATS( first, result );
		first += 4 * AvxBlockSize;
		second += 4 * AvxBlockSize;
		result += 4 * AvxBlockSize;
		vectorSize -= 4 * AvxBlockSize;
	}

	while( vectorSize >= AvxBlockSize ) {
		_mm512_storeu_ps( result,
			_mm512_add_ps( _mm512_loadu_ps( first ), _mm512_loadu_ps( second ) ) );
		first += AvxBlockSize;
		second += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
		const __m512 zeroSimd = _mm512_setzero_ps(); // copy data from here, where mask bits are false
		const __mmask16 mask = AVX512_IO_MASK( vectorSize );

		const __m512 firstSimd = _mm512_mask_loadu_ps( zeroSimd, mask, first );
		const __m512 secondSimd = _mm512_mask_loadu_ps( zeroSimd, mask, second );
		_mm512_mask_storeu_ps( result, mask, _mm512_add_ps( firstSimd, secondSimd ) );
	}
}

void vectorEltwiseMultiply( const float* first, const float* second, float* result, int vectorSize )
{
	while( vectorSize >= 4 * AvxBlockSize ) {
		AVX512_LOAD_64_FLOATS( first, first );
		AVX512_LOAD_64_FLOATS( second, second );
		first0 = _mm512_mul_ps( first0, second0 );
		first1 = _mm512_mul_ps( first1, second1 );
		first2 = _mm512_mul_ps( first2, second2 );
		first3 = _mm512_mul_ps( first3, second3 );
		AVX512_STORE_64_FLOATS( first, result );
		first += 4 * AvxBlockSize;
		second += 4 * AvxBlockSize;
		result += 4 * AvxBlockSize;
		vectorSize -= 4 * AvxBlockSize;
	}

	while( vectorSize >= AvxBlockSize ) {
		_mm512_storeu_ps( result,
			_mm512_mul_ps( _mm512_loadu_ps( first ), _mm512_loadu_ps( second ) ) );
		first += AvxBlockSize;
		second += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
		const __m512 zeroSimd = _mm512_setzero_ps(); // copy data from here, where mask bits are false
		const __mmask16 mask = AVX512_IO_MASK( vectorSize );

		const __m512 firstSimd = _mm512_mask_loadu_ps( zeroSimd, mask, first );
		const __m512 secondSimd = _mm512_mask_loadu_ps( zeroSimd, mask, second );
		_mm512_mask_storeu_ps( result, mask, _mm512_mul_ps( firstSimd, secondSimd ) );
	}
}

void vectorEltwiseMultiplyAdd( const float* first, const float* second, float* result, int vectorSize )
{
	while( vectorSize >= 4 * AvxBlockSize ) {
		AVX512_LOAD_64_FLOATS( first, first );
		AVX512_LOAD_64_FLOATS( second, second );
		AVX512_LOAD_64_FLOATS( result, result );
		result0 = _mm512_fmadd_ps( first0, second0, result0 );
		result1 = _mm512_fmadd_ps( first1, second1, result1 );
		result2 = _mm512_fmadd_ps( first2, second2, result2 );
		result3 = _mm512_fmadd_ps( first3, second3, result3 );
		AVX512_STORE_64_FLOATS( result, result );
		first += 4 * AvxBlockSize;
		second += 4 * AvxBlockSize;
		result += 4 * AvxBlockSize;
		vectorSize -= 4 * AvxBlockSize;
	}

	while( vectorSize >= AvxBlockSize ) {
		_mm512_storeu_ps( result,
			_mm512_fmadd_ps( _mm512_loadu_ps( first ), _mm512_loadu_ps( second ), _mm512_loadu_ps( result ) ) );
		first += AvxBlockSize;
		second += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
		const __m512 zeroSimd = _mm512_setzero_ps(); // copy data from here, where mask bits are false
		const __mmask16 mask = AVX512_IO_MASK( vectorSize );

		const __m512 firstSimd = _mm512_mask_loadu_ps( zeroSimd, mask, first );
		const __m512 secondSimd = _mm512_mask_loadu_ps( zeroSimd, mask, second );
		const __m512 resultSimd = _mm512_mask_loadu_ps( zeroSimd, mask, result );
		_mm512_mask_storeu_ps( result, mask, _mm512_fmadd_ps( firstSimd, secondSimd, resultSimd ) );
	}
}

void vectorReLU( const float* first, float* result, int vectorSize )
{
	const __m512 zeroSimd = _mm512_setzero_ps();

	while( vectorSize >= AvxBlockSize ) {
		_mm512_storeu_ps( result,
			_mm512_max_ps( _mm512_loadu_ps( first ), zeroSimd ) );
		first += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
		const __mmask16 mask = AVX512_IO_MASK( vectorSize );
		_mm512_mask_storeu_ps( result, mask, _mm512_max_ps( _mm512_mask_loadu_ps( zeroSimd, mask, first ), zeroSimd ) );
	}
}

void vectorReLU( const float* first, float* result, int vectorSize, float threshold )
{
	const __m512 zeroSimd = _mm512_setzero_ps();
	const __m512 thresholdSimd = _mm512_set1_ps( threshold );

	while( vectorSize >= AvxBlockSize ) {
		_mm512_storeu_ps( result,
			_mm512_min_ps( _mm512_max_ps( _mm512_loadu_ps( first ), zeroSimd ), thresholdSimd ) );
		first += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
		const __mmask16 mask = AVX512_IO_MASK( vectorSize );
		const __m512 firstSimd = _mm512_mask_loadu_ps( zeroSimd, mask, first );
		_mm512_mask_storeu_ps( result, mask, _mm512_min_ps( _mm512_max_ps( firstSimd, zeroSimd ), thresholdSimd ) );
	}
}


} // namespace Avx512

} // namespace NeoML

#endif // NEOML_USE_SSE
