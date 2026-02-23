/* Copyright © 2017-2024 ABBYY

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

//#define NEOML_USE_AVX_MASK // need (/AVX512 DQ+F+VL) for functions:  _cvtu32_mask8, _mm256_mask_storeu_ps, _mm256_mask_loadu_ps
#ifdef  NEOML_USE_AVX_MASK

#define AVX_IO_MASK( N ) \
	_cvtu32_mask8( ( 1u << N ) - 1u )

#else  // !NEOML_USE_AVX_MASK
static_assert( sizeof( int ) == sizeof( float ), "Avx2: invalid size int != float" );
static constexpr int avxIOMask[2 * ( AvxBlockSize - 1 )]{ -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0 };

#define AVX_IO_MASK( N ) \
	_mm256_lddqu_si256( reinterpret_cast<const __m256i*>( avxIOMask + AvxBlockSize - 1 - N ) )
#endif // !NEOML_USE_AVX_MASK


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
	while( vectorSize >= 4 * AvxBlockSize ) {
		AVX_LOAD_32_FLOATS( data, src );
		AVX_STORE_32_FLOATS( data, dst );
		dst += 4 * AvxBlockSize;
		src += 4 * AvxBlockSize;
		vectorSize -= 4 * AvxBlockSize;
	}

	while( vectorSize >= AvxBlockSize ) {
		_mm256_storeu_ps( dst, _mm256_loadu_ps( src ) );
		dst += AvxBlockSize;
		src += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
#ifdef  NEOML_USE_AVX_MASK
		const __mmask8 mask = AVX_IO_MASK( vectorSize );
		_mm256_mask_storeu_ps( dst, mask, _mm256_mask_loadu_ps( _mm256_setzero_ps(), mask, src ) );
#else  // !NEOML_USE_AVX_MASK
		const __m256i mask = AVX_IO_MASK( vectorSize );
		_mm256_maskstore_ps( dst, mask, _mm256_maskload_ps( src, mask ) );
#endif // !NEOML_USE_AVX_MASK
	}
}

void vectorFill( float* result, int vectorSize, float value )
{
	const __m256 valueSimd = _mm256_set1_ps( value );

	while( vectorSize >= 4 * AvxBlockSize ) {
		_mm256_storeu_ps( result + 0 * AvxBlockSize, valueSimd );
		_mm256_storeu_ps( result + 1 * AvxBlockSize, valueSimd );
		_mm256_storeu_ps( result + 2 * AvxBlockSize, valueSimd );
		_mm256_storeu_ps( result + 3 * AvxBlockSize, valueSimd );
		result += 4 * AvxBlockSize;
		vectorSize -= 4 * AvxBlockSize;
	}

	while( vectorSize >= AvxBlockSize ) {
		_mm256_storeu_ps( result, valueSimd );
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
#ifdef  NEOML_USE_AVX_MASK
		const __mmask8 mask = AVX_IO_MASK( vectorSize );
		_mm256_mask_storeu_ps( result, mask, valueSimd );
#else  // !NEOML_USE_AVX_MASK
		_mm256_maskstore_ps( result, AVX_IO_MASK( vectorSize ), valueSimd );
#endif // !NEOML_USE_AVX_MASK
	}
}

void vectorAdd( const float* first, const float* second, float* result, int vectorSize )
{
	while( vectorSize >= 4 * AvxBlockSize ) {
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
		vectorSize -= 4 * AvxBlockSize;
	}

	while( vectorSize >= AvxBlockSize ) {
		_mm256_storeu_ps( result,
			_mm256_add_ps( _mm256_loadu_ps( first ), _mm256_loadu_ps( second ) ) );
		first += AvxBlockSize;
		second += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
#ifdef  NEOML_USE_AVX_MASK
		const __m256 zeroSimd = _mm256_setzero_ps();
		const __mmask8 mask = AVX_IO_MASK( vectorSize );

		const __m256 firstSimd = _mm256_mask_loadu_ps( zeroSimd, mask, first );
		const __m256 secondSimd = _mm256_mask_loadu_ps( zeroSimd, mask, second );
		_mm256_mask_storeu_ps( result, mask, _mm256_add_ps( firstSimd, secondSimd ) );
#else  // !NEOML_USE_AVX_MASK
		const __m256i mask = AVX_IO_MASK( vectorSize );
		const __m256 firstSimd = _mm256_maskload_ps( first, mask );
		const __m256 secondSimd = _mm256_maskload_ps( second, mask );
		_mm256_maskstore_ps( result, mask, _mm256_add_ps( firstSimd, secondSimd ) );
#endif // !NEOML_USE_AVX_MASK
	}
}

void vectorAddValue( const float* first, float* result, int vectorSize, float value )
{
	const __m256 valueSimd = _mm256_set1_ps( value );

	while( vectorSize >= AvxBlockSize ) {
		_mm256_storeu_ps( result,
			_mm256_add_ps( _mm256_loadu_ps( first ), valueSimd ) );
		first += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;		
	}

	if( vectorSize > 0 ) { 
		const __m256i mask = AVX_IO_MASK( vectorSize );
		_mm256_maskstore_ps( result, mask,
			_mm256_add_ps( _mm256_maskload_ps( first, mask ), valueSimd ) );
	}
}

void vectorMultiply( const float* first, float* result, int vectorSize, float multiplier )
{
	const __m256 multSimd = _mm256_set1_ps( multiplier );

	while( vectorSize >= AvxBlockSize ) {
		_mm256_storeu_ps( result,
			_mm256_mul_ps( _mm256_loadu_ps( first ), multSimd ) );
		first += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;		
	}

	if( vectorSize > 0 ) { 
		const __m256i mask = AVX_IO_MASK( vectorSize );
		_mm256_maskstore_ps( result, mask,
			_mm256_mul_ps( _mm256_maskload_ps( first, mask ), multSimd ) );
	}
}

void vectorEltwiseMultiply( const float* first, const float* second, float* result, int vectorSize )
{
	while( vectorSize >= 4 * AvxBlockSize ) {
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
		vectorSize -= 4 * AvxBlockSize;
	}

	while( vectorSize >= AvxBlockSize ) {
		_mm256_storeu_ps( result,
			_mm256_mul_ps( _mm256_loadu_ps( first ), _mm256_loadu_ps( second ) ) );
		first += AvxBlockSize;
		second += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
#ifdef  NEOML_USE_AVX_MASK
		const __m256 zeroSimd = _mm256_setzero_ps();
		const __mmask8 mask = AVX_IO_MASK( vectorSize );

		const __m256 firstSimd = _mm256_mask_loadu_ps( zeroSimd, mask, first );
		const __m256 secondSimd = _mm256_mask_loadu_ps( zeroSimd, mask, second );
		_mm256_mask_storeu_ps( result, mask, _mm256_mul_ps( firstSimd, secondSimd ) );
#else  // !NEOML_USE_AVX_MASK
		const __m256i mask = AVX_IO_MASK( vectorSize );
		const __m256 firstSimd = _mm256_maskload_ps( first, mask );
		const __m256 secondSimd = _mm256_maskload_ps( second, mask );
		_mm256_maskstore_ps( result, mask, _mm256_mul_ps( firstSimd, secondSimd ) );
#endif // !NEOML_USE_AVX_MASK
	}
}

void vectorEltwiseMultiplyAdd( const float* first, const float* second, float* result, int vectorSize )
{
	while( vectorSize >= 4 * AvxBlockSize ) {
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
		vectorSize -= 4 * AvxBlockSize;
	}

	while( vectorSize >= AvxBlockSize ) {
		_mm256_storeu_ps( result,
			_mm256_fmadd_ps( _mm256_loadu_ps( first ), _mm256_loadu_ps( second ), _mm256_loadu_ps( result ) ) );
		first += AvxBlockSize;
		second += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
#ifdef  NEOML_USE_AVX_MASK
		const __m256 zeroSimd = _mm256_setzero_ps();
		const __mmask8 mask = AVX_IO_MASK( vectorSize );

		const __m256 firstSimd = _mm256_mask_loadu_ps( zeroSimd, mask, first );
		const __m256 secondSimd = _mm256_mask_loadu_ps( zeroSimd, mask, second );
		const __m256 resultSimd = _mm256_mask_loadu_ps( zeroSimd, mask, result );
		_mm256_mask_storeu_ps( result, mask, _mm256_fmadd_ps( firstSimd, secondSimd, resultSimd ) );
#else  // !NEOML_USE_AVX_MASK
		const __m256i mask = AVX_IO_MASK( vectorSize );
		const __m256 firstSimd = _mm256_maskload_ps( first, mask );
		const __m256 secondSimd = _mm256_maskload_ps( second, mask );
		const __m256 resultSimd = _mm256_maskload_ps( result, mask );
		_mm256_maskstore_ps( result, mask, _mm256_fmadd_ps( firstSimd, secondSimd, resultSimd ) );
#endif // !NEOML_USE_AVX_MASK
	}
}

void vectorReLU( const float* first, float* result, int vectorSize )
{
	const __m256 zeroSimd = _mm256_setzero_ps();

	while( vectorSize >= AvxBlockSize ) {
		_mm256_storeu_ps( result,
			_mm256_max_ps( _mm256_loadu_ps( first ), zeroSimd ) );
		first += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
#ifdef  NEOML_USE_AVX_MASK
		const __mmask8 mask = AVX_IO_MASK( vectorSize );
		_mm256_mask_storeu_ps( result, mask, _mm256_max_ps( _mm256_mask_loadu_ps( zeroSimd, mask, first ), zeroSimd ) );
#else  // !NEOML_USE_AVX_MASK
		const __m256i mask = AVX_IO_MASK( vectorSize );
		_mm256_maskstore_ps( result, mask, _mm256_max_ps( _mm256_maskload_ps( first, mask ), zeroSimd ) );
#endif // !NEOML_USE_AVX_MASK
	}
}

void vectorReLU( const float* first, float* result, int vectorSize, float threshold )
{
	const __m256 zeroSimd = _mm256_setzero_ps();
	const __m256 thresholdSimd = _mm256_set1_ps( threshold );

	while( vectorSize >= AvxBlockSize ) {
		_mm256_storeu_ps( result,
			_mm256_min_ps( _mm256_max_ps( _mm256_loadu_ps( first ), zeroSimd ), thresholdSimd ) );
		first += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
#ifdef  NEOML_USE_AVX_MASK
		const __mmask8 mask = AVX_IO_MASK( vectorSize );
		const __m256 firstSimd = _mm256_mask_loadu_ps( zeroSimd, mask, first );
		_mm256_mask_storeu_ps( result, mask, _mm256_min_ps( _mm256_max_ps( firstSimd, zeroSimd ), thresholdSimd ) );
#else  // !NEOML_USE_AVX_MASK
		const __m256i mask = AVX_IO_MASK( vectorSize );
		const __m256 firstSimd = _mm256_maskload_ps( first, mask );
		_mm256_maskstore_ps( result, mask, _mm256_min_ps( _mm256_max_ps( firstSimd, zeroSimd ), thresholdSimd ) );
#endif // !NEOML_USE_AVX_MASK
	}
}

void vectorHSwish( const float* first, float* result, int vectorSize )
{
	const __m256 zeroSimd = _mm256_setzero_ps();
	const __m256 threeSimd = _mm256_set1_ps( 3.f );
	const __m256 oneSixthSimd = _mm256_set1_ps( 1.f / 6.f );

	while( vectorSize >= AvxBlockSize ) {
		__m256 firstSimd = _mm256_loadu_ps( first );
		__m256 middlePart = _mm256_max_ps( _mm256_add_ps( firstSimd, threeSimd ), zeroSimd );
		middlePart = _mm256_mul_ps( _mm256_mul_ps( firstSimd, oneSixthSimd ), middlePart );
		_mm256_storeu_ps( result, _mm256_min_ps( middlePart, _mm256_max_ps( firstSimd, threeSimd ) ) );

		first += AvxBlockSize;
		result += AvxBlockSize;
		vectorSize -= AvxBlockSize;
	}

	if( vectorSize > 0 ) {
#ifdef  NEOML_USE_AVX_MASK
		const __mmask8 mask = AVX_IO_MASK( vectorSize );
		__m256 firstSimd = _mm256_mask_loadu_ps( _mm256_setzero_ps(), mask, first );
#else  // !NEOML_USE_AVX_MASK
		const __m256i mask = AVX_IO_MASK( vectorSize );
		__m256 firstSimd = _mm256_maskload_ps( first, mask );
		__m256 middlePart = _mm256_max_ps( _mm256_add_ps( firstSimd, threeSimd ), zeroSimd );
		middlePart = _mm256_mul_ps( _mm256_mul_ps( firstSimd, oneSixthSimd ), middlePart );
		_mm256_maskstore_ps( result, mask, _mm256_min_ps( middlePart, _mm256_max_ps( firstSimd, threeSimd ) ) );
#endif // !NEOML_USE_AVX_MASK
	}
}

} // namespace Avx2

} // namespace NeoML

#endif // NEOML_USE_SSE
