/* Copyright © 2017-2020 ABBYY Production LLC

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

// These functions work with raw pointers, contain no OMP sections and perform no checks

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_SSE

#include "CpuX86.h"

#ifdef NEOML_USE_MKL
#if FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
#include <mkl.h>
#else
#error Unknown platform
#endif
#endif

namespace NeoML {

inline void channelwiseConvolution1x3Kernel( const float* source0, const float* source1, const float* source2, const float* source3,
	const float* filter0, const float* filter1, const float* filter2,
	float* result0, float* result1 )
{
	__m128 result0_4 = _mm_loadu_ps(result0);
	__m128 result1_4 = _mm_loadu_ps(result1);

	__m128 filter_4 = _mm_loadu_ps(filter0);
	__m128 source_4 = _mm_loadu_ps(source0);
	result0_4 = _mm_add_ps( result0_4, _mm_mul_ps( source_4, filter_4 ) );

	source_4 = _mm_loadu_ps(source1);
	result1_4 = _mm_add_ps( result1_4, _mm_mul_ps( source_4, filter_4 ) );

	filter_4 = _mm_loadu_ps(filter1);
	result0_4 = _mm_add_ps( result0_4, _mm_mul_ps( source_4, filter_4 ) );

	source_4 = _mm_loadu_ps(source2);
	result1_4 = _mm_add_ps( result1_4, _mm_mul_ps( source_4, filter_4 ) );

	filter_4 = _mm_loadu_ps(filter2);
	result0_4 = _mm_add_ps( result0_4, _mm_mul_ps( source_4, filter_4 ) );

	source_4 = _mm_loadu_ps(source3);
	result1_4 = _mm_add_ps( result1_4, _mm_mul_ps( source_4, filter_4 ) );

	_mm_storeu_ps(result0, result0_4);
	_mm_storeu_ps(result1, result1_4);
}

inline void channelwise1x3( const float* source, const float* filter0, const float* filter1, const float* filter2, float* result, int channels )
{
	const int shift1 = channels;
	const int shift2 = 2 * channels;
	const int shift3 = 3 * channels;
	while( channels > 0 ) {
		channelwiseConvolution1x3Kernel( source, source + shift1, source + shift2, source + shift3,
			filter0, filter1, filter2,
			result, result + shift1 );

		source += 4;
		filter0 += 4; filter1 += 4; filter2 += 4;
		result += 4;
		channels -= 4;
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorFill( float* result, float value, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	__m128 valueSse = _mm_set_ps1(value);

	while( sseSize >= 4 ) {
		_mm_storeu_ps(result, valueSse);
		result += 4;
		_mm_storeu_ps(result, valueSse);
		result += 4;
		_mm_storeu_ps(result, valueSse);
		result += 4;
		_mm_storeu_ps(result, valueSse);
		result += 4;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		_mm_storeu_ps(result, valueSse);
		result += 4;
		sseSize--;
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result++ = value;
	}
}

inline void vectorFill( int* result, int value, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	__m128i valueSse = _mm_set1_epi32( value );

	while( sseSize >= 4 ) {
		_mm_storeu_si128( ( __m128i* )result, valueSse );
		result += 4;
		_mm_storeu_si128( ( __m128i* )result, valueSse );
		result += 4;
		_mm_storeu_si128( ( __m128i* )result, valueSse );
		result += 4;
		_mm_storeu_si128( ( __m128i* )result, valueSse );
		result += 4;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		_mm_storeu_si128( ( __m128i* )result, valueSse );
		result += 4;
		sseSize--;
	}

#if FINE_PLATFORM(FINE_WINDOWS)
	if( nonSseSize > 0 ) {
		__stosd( (DWORD*) result, value, nonSseSize );
	}
#elif FINE_PLATFORM(FINE_LINUX) || FINE_PLATFORM(FINE_DARWIN) || FINE_PLATFORM(FINE_ANDROID) || FINE_PLATFORM(FINE_IOS)
	for( int i = 0; i < nonSseSize; ++i ) {
		*result++ = value;
	}
#else
#error "Platform isn't supported!"
#endif
}

inline void vectorFill0( float* result, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	__m128 valueSse = _mm_setzero_ps();

	while( sseSize >= 4 ) {
		_mm_storeu_ps(result, valueSse);
		result += 4;
		_mm_storeu_ps(result, valueSse);
		result += 4;
		_mm_storeu_ps(result, valueSse);
		result += 4;
		_mm_storeu_ps(result, valueSse);
		result += 4;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		_mm_storeu_ps(result, valueSse);
		result += 4;
		sseSize--;
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result++ = 0;
	}
}

inline void vectorEltwiseMax( const float* first, const float* second, float* result, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	while( sseSize >= 4 ) {
		_mm_storeu_ps(result, _mm_max_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
		_mm_storeu_ps(result, _mm_max_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
		_mm_storeu_ps(result, _mm_max_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
		_mm_storeu_ps(result, _mm_max_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		_mm_storeu_ps(result, _mm_max_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
		sseSize--;
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result++ = max(*first, *second);
		first++;
		second++;
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorAdd(const float* first, const float* second, float* result, int vectorSize)
{
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	while( sseSize >= 4 ) {
		_mm_storeu_ps(result, _mm_add_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
		_mm_storeu_ps(result, _mm_add_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
		_mm_storeu_ps(result, _mm_add_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
		_mm_storeu_ps(result, _mm_add_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		_mm_storeu_ps(result, _mm_add_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
		sseSize--;
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		result[i] = first[i] + second[i];
	}
}

inline void vectorAdd( const int* first, const int* second, int* result, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	while( sseSize >= 4 ) {
		SSE_LOAD_16_INTS( first, first );
		first += 16;

		SSE_LOAD_16_INTS( second, second );
		second += 16;

		__m128i result0 = _mm_add_epi32( first0, second0 );
		__m128i result1 = _mm_add_epi32( first1, second1 );
		__m128i result2 = _mm_add_epi32( first2, second2 );
		__m128i result3 = _mm_add_epi32( first3, second3 );

		SSE_STORE_16_INTS( result, result );

		result += 16;
		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		StoreIntSse4( _mm_add_epi32( LoadIntSse4( first ), LoadIntSse4( second ) ), result );
		first += 4;
		second += 4;
		result += 4;
		sseSize--;
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		result[i] = first[i] + second[i];
	}
}

//------------------------------------------------------------------------------------------------------------

inline void alignedVectorAdd( float* first, const float* second, int vectorSize )
{
	int sseSize = vectorSize / 4;
	while( sseSize >= 4 ) {
		_mm_store_ps( first, _mm_add_ps( _mm_load_ps( first ), _mm_load_ps( second ) ) );
		first += 4;
		second += 4;
		_mm_store_ps( first, _mm_add_ps( _mm_load_ps( first ), _mm_load_ps( second ) ) );
		first += 4;
		second += 4;
		_mm_store_ps( first, _mm_add_ps( _mm_load_ps( first ), _mm_load_ps( second ) ) );
		first += 4;
		second += 4;
		_mm_store_ps( first, _mm_add_ps( _mm_load_ps( first ), _mm_load_ps( second ) ) );
		first += 4;
		second += 4;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		_mm_store_ps( first, _mm_add_ps( _mm_load_ps( first ), _mm_load_ps( second ) ) );
		first += 4;
		second += 4;
		sseSize--;
	}
}

inline void alignedVectorMultiplyAndAdd( const float* first, const float* second,
	float* result, int vectorSize, const float* mult )
{
	int sseSize = vectorSize / 4;
	__m128 multSse = _mm_set_ps1( *mult );
	while( sseSize >= 4 ) {
		_mm_store_ps( result, _mm_add_ps( _mm_load_ps( first ), _mm_mul_ps( _mm_load_ps( second ), multSse ) ) );
		first += 4;
		second += 4;
		result += 4;
		_mm_store_ps( result, _mm_add_ps( _mm_load_ps( first ), _mm_mul_ps( _mm_load_ps( second ), multSse ) ) );
		first += 4;
		second += 4;
		result += 4;
		_mm_store_ps( result, _mm_add_ps( _mm_load_ps( first ), _mm_mul_ps( _mm_load_ps( second ), multSse ) ) );
		first += 4;
		second += 4;
		result += 4;
		_mm_store_ps( result, _mm_add_ps( _mm_load_ps( first ), _mm_mul_ps( _mm_load_ps( second ), multSse ) ) );
		first += 4;
		second += 4;
		result += 4;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		_mm_store_ps( result, _mm_add_ps( _mm_load_ps( first ), _mm_mul_ps( _mm_load_ps( second ), multSse ) ) );
		first += 4;
		second += 4;
		result += 4;
		sseSize--;
	}
}

//------------------------------------------------------------------------------------------------------------

// Due to SSE 2.0 requirement we can't use _mm_mullo_epi32
inline __m128i sse2Multiply4SignedInts( const __m128i& first, const __m128i& second )
{
	__m128i prod02 = _mm_mul_epu32( first, second ); // multiplies 0'th and 2'nd elems
	__m128i prod13 = _mm_mul_epu32(
		_mm_srli_si128( first, 4 ), // shift right by one integer in order to get 1'st and 3'rd elems
		_mm_srli_si128( second, 4 )
	);
	return _mm_unpacklo_epi32(
		_mm_shuffle_epi32( prod02, _MM_SHUFFLE( 0, 0, 2, 0 ) ), // move 0'th and 2'nd productions into 2 lower integers
		_mm_shuffle_epi32( prod13, _MM_SHUFFLE( 0, 0, 2, 0 ) ) // move 1'st adn 3'rd productions into 2 lower integers
	);
}

//------------------------------------------------------------------------------------------------------------

inline void vectorMultiply( const float* first, float* result, float multiplier, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		__m128 multSse = _mm_set_ps1( multiplier );
		for( int i = 0; i < sseSize; ++i ) {
			_mm_storeu_ps( result, _mm_mul_ps( _mm_loadu_ps( first ), multSse ) );
			first += 4;
			result += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result++ = *first++ * multiplier;
	}
}

inline void vectorMultiply( const int* first, int* result, int multiplier, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		__m128i multSse = _mm_set1_epi32( multiplier );
		for( int i = 0; i < sseSize; ++i ) {
			StoreIntSse4( sse2Multiply4SignedInts( LoadIntSse4( first ), multSse ), result );
			first += 4;
			result += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result++ = *first++ * multiplier;
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorEltwiseMultiply( const float* first, const float* second, float* result, int sseSize, int nonSseSize )
{
	while( sseSize >= 4 ) {
		__m128 first0 = LoadSse4( first );
		__m128 first1 = LoadSse4( first + 4 );
		__m128 first2 = LoadSse4( first + 8 );
		__m128 first3 = LoadSse4( first + 12 );
		first += 16;

		__m128 second0 = LoadSse4( second );
		__m128 second1 = LoadSse4( second + 4 );
		__m128 second2 = LoadSse4( second + 8 );
		__m128 second3 = LoadSse4( second + 12 );
		second += 16;

		__m128 res0 = _mm_mul_ps( first0, second0 );
		__m128 res1 = _mm_mul_ps( first1, second1 );
		__m128 res2 = _mm_mul_ps( first2, second2 );
		__m128 res3 = _mm_mul_ps( first3, second3 );

		StoreSse4( res0, result );
		StoreSse4( res1, result + 4 );
		StoreSse4( res2, result + 8 );
		StoreSse4( res3, result + 12 );
		result += 16;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		__m128 first0 = LoadSse4( first );
		first += 4;

		__m128 second0 = LoadSse4( second );
		second += 4;

		__m128 res0 = _mm_mul_ps( first0, second0 );
		StoreSse4( res0, result );
		result += 4;

		sseSize--;
	}

	if( nonSseSize ) {
		__m128 first0 = LoadSse( first, nonSseSize );
		__m128 second0 = LoadSse( second, nonSseSize );
		__m128 res0 = _mm_mul_ps( first0, second0 );
		StoreSse( res0, result, nonSseSize );
	}
}

inline void vectorEltwiseMultiply( const float* first, const float* second, float* result, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);
	vectorEltwiseMultiply( first, second, result, sseSize, nonSseSize );
}

inline void vectorEltwiseMultiply( const int* first, const int* second, int* result, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	while( sseSize >= 4 ) {
		__m128i first0 = LoadIntSse4( first );
		__m128i first1 = LoadIntSse4( first + 4 );
		__m128i first2 = LoadIntSse4( first + 8 );
		__m128i first3 = LoadIntSse4( first + 12 );
		first += 16;

		__m128i second0 = LoadIntSse4( second );
		__m128i second1 = LoadIntSse4( second + 4 );
		__m128i second2 = LoadIntSse4( second + 8 );
		__m128i second3 = LoadIntSse4( second + 12 );
		second += 16;

		__m128i res0 = sse2Multiply4SignedInts( first0, second0 );
		__m128i res1 = sse2Multiply4SignedInts( first1, second1 );
		__m128i res2 = sse2Multiply4SignedInts( first2, second2 );
		__m128i res3 = sse2Multiply4SignedInts( first3, second3 );

		StoreIntSse4( res0, result );
		StoreIntSse4( res1, result + 4 );
		StoreIntSse4( res2, result + 8 );
		StoreIntSse4( res3, result + 12 );
		result += 16;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		__m128i first0 = LoadIntSse4( first );
		first += 4;

		__m128i second0 = LoadIntSse4( second );
		second += 4;

		__m128i res0 = sse2Multiply4SignedInts( first0, second0 );
		StoreIntSse4( res0, result );
		result += 4;

		sseSize--;
	}

	if( nonSseSize ) {
		__m128i first0 = LoadIntSse( first, nonSseSize );
		__m128i second0 = LoadIntSse( second, nonSseSize );
		__m128i res0 = sse2Multiply4SignedInts( first0, second0 );
		StoreIntSse( res0, result, nonSseSize );
	}
}

inline void vectorEltwiseMultiplyAdd( const float* first, const float* second, float* result, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	while( sseSize >= 4 ) {
		__m128 product = _mm_mul_ps( _mm_loadu_ps(first), _mm_loadu_ps(second) );
		_mm_storeu_ps( result, _mm_add_ps( _mm_loadu_ps(result), product ) );
		first += 4;
		second += 4;
		result += 4;
		product = _mm_mul_ps( _mm_loadu_ps(first), _mm_loadu_ps(second) );
		_mm_storeu_ps( result, _mm_add_ps( _mm_loadu_ps(result), product ) );
		first += 4;
		second += 4;
		result += 4;
		product = _mm_mul_ps( _mm_loadu_ps(first), _mm_loadu_ps(second) );
		_mm_storeu_ps( result, _mm_add_ps( _mm_loadu_ps(result), product ) );
		first += 4;
		second += 4;
		result += 4;
		product = _mm_mul_ps( _mm_loadu_ps(first), _mm_loadu_ps(second) );
		_mm_storeu_ps( result, _mm_add_ps( _mm_loadu_ps(result), product ) );
		first += 4;
		second += 4;
		result += 4;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		const __m128 product = _mm_mul_ps( _mm_loadu_ps(first), _mm_loadu_ps(second) );
		_mm_storeu_ps( result, _mm_add_ps( _mm_loadu_ps(result), product ) );
		first += 4;
		second += 4;
		result += 4;
		sseSize--;
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result++ += *first++ * *second++;
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorReLU( const float* first, float* result, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	const __m128 zeroSse = _mm_setzero_ps();
	while( sseSize >= 4 ) {
		_mm_storeu_ps(result, _mm_max_ps(_mm_loadu_ps(first), zeroSse));
		first += 4;
		result += 4;
		_mm_storeu_ps(result, _mm_max_ps(_mm_loadu_ps(first), zeroSse));
		first += 4;
		result += 4;
		_mm_storeu_ps(result, _mm_max_ps(_mm_loadu_ps(first), zeroSse));
		first += 4;
		result += 4;
		_mm_storeu_ps(result, _mm_max_ps(_mm_loadu_ps(first), zeroSse));
		first += 4;
		result += 4;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		_mm_storeu_ps(result, _mm_max_ps(_mm_loadu_ps(first), zeroSse));
		first += 4;
		result += 4;
		sseSize--;
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result = max(*first, 0.f);
		result++;
		first++;
	}
}

inline void vectorReLU( const float* first, float* result, int vectorSize, float threshold )
{
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	const __m128 zeroSse = _mm_setzero_ps();
	const __m128 thresholdSse = _mm_set_ps1(threshold);
	while( sseSize >= 4 ) {
		_mm_storeu_ps(result, _mm_min_ps(_mm_max_ps(_mm_loadu_ps(first), zeroSse), thresholdSse));
		first += 4;
		result += 4;
		_mm_storeu_ps(result, _mm_min_ps(_mm_max_ps(_mm_loadu_ps(first), zeroSse), thresholdSse));
		first += 4;
		result += 4;
		_mm_storeu_ps(result, _mm_min_ps(_mm_max_ps(_mm_loadu_ps(first), zeroSse), thresholdSse));
		first += 4;
		result += 4;
		_mm_storeu_ps(result, _mm_min_ps(_mm_max_ps(_mm_loadu_ps(first), zeroSse), thresholdSse));
		first += 4;
		result += 4;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		_mm_storeu_ps(result, _mm_min_ps(_mm_max_ps(_mm_loadu_ps(first), zeroSse), thresholdSse));
		first += 4;
		result += 4;
		sseSize--;
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result = min(max(*first, 0.f), threshold);
		result++;
		first++;
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorAddValue( const float* first, float* result, int vectorSize, float value )
{
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		const __m128 addSse = _mm_set_ps1(value);
		for(int i = 0; i < sseSize; ++i) {
			_mm_storeu_ps(result, _mm_add_ps(_mm_loadu_ps(first), addSse));
			first += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		result[i] = first[i] + value;
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorDotProduct( const float* first, const float* second, int vectorSize, float* result )
{
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	float acc = 0;

	if(sseSize > 0) {
		__m128 sum = _mm_setzero_ps();
		for(int i = 0; i < sseSize; ++i) {
			sum = _mm_add_ps(sum, _mm_mul_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));

			first += 4;
			second += 4;
		}

		__m128 tmp = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 3, 2, 1));
		sum = _mm_add_ps(sum, tmp);
		tmp = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2));
		sum = _mm_add_ss(sum, tmp);

		acc += _mm_cvtss_f32(sum);
	}

	for(int i = 0; i < nonSseSize; ++i) {
		acc += *first++ * *second++;
	}

	*result = acc;
}

//------------------------------------------------------------------------------------------------------------

// QRNN primitives

// res = z * ( 1 - f )
static inline void qrnnFPoolingFirstStep( const float* z, const float* f,
	float* res, int sseSize, int nonSseSize )
{
	__m128 ones = _mm_set1_ps( 1.f );
	while( sseSize >= 4 ) {
		__m128 z0 = LoadSse4( z );
		__m128 z1 = LoadSse4( z + 4 );
		__m128 z2 = LoadSse4( z + 8 );
		__m128 z3 = LoadSse4( z + 12 );
		z += 16;

		__m128 f0 = LoadSse4( f );
		__m128 f1 = LoadSse4( f + 4 );
		__m128 f2 = LoadSse4( f + 8 );
		__m128 f3 = LoadSse4( f + 12 );
		f += 16;

		__m128 res0 = _mm_mul_ps( z0, _mm_sub_ps( ones, f0 ) );
		__m128 res1 = _mm_mul_ps( z1, _mm_sub_ps( ones, f1 ) );
		__m128 res2 = _mm_mul_ps( z2, _mm_sub_ps( ones, f2 ) );
		__m128 res3 = _mm_mul_ps( z3, _mm_sub_ps( ones, f3 ) );

		StoreSse4( res0, res );
		StoreSse4( res1, res + 4 );
		StoreSse4( res2, res + 8 );
		StoreSse4( res3, res + 12 );
		res += 16;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		__m128 z0 = LoadSse4( z );
		z += 4;

		__m128 f0 = LoadSse4( f );
		f += 4;

		__m128 res0 = _mm_mul_ps( z0, _mm_sub_ps( ones, f0 ) );
		StoreSse4( res0, res );
		res += 4;

		--sseSize;
	}

	if( nonSseSize > 0 ) {
		__m128 z0 = LoadSse( z, nonSseSize );
		__m128 f0 = LoadSse( f, nonSseSize );
		__m128 res0 = _mm_mul_ps( z0, _mm_sub_ps( ones, f0 ) );
		StoreSse( res0, res, nonSseSize );
	}
}

// res = f * h + (1 - f) * z
// where h - res of previous step
static inline void qrnnFPoolingStep( const float* z, const float* f, const float* h,
	float* res, int sseSize, int nonSseSize )
{
	__m128 ones = _mm_set1_ps( 1.f );
	while( sseSize >= 4 ) {
		__m128 z0 = LoadSse4( z );
		__m128 z1 = LoadSse4( z + 4 );
		__m128 z2 = LoadSse4( z + 8 );
		__m128 z3 = LoadSse4( z + 12 );
		z += 16;

		__m128 f0 = LoadSse4( f );
		__m128 f1 = LoadSse4( f + 4 );
		__m128 f2 = LoadSse4( f + 8 );
		__m128 f3 = LoadSse4( f + 12 );
		f += 16;

		__m128 h0 = LoadSse4( h );
		__m128 h1 = LoadSse4( h + 4 );
		__m128 h2 = LoadSse4( h + 8 );
		__m128 h3 = LoadSse4( h + 12 );
		h += 16;

		__m128 res0 = _mm_add_ps( _mm_mul_ps( f0, h0 ), _mm_mul_ps( _mm_sub_ps( ones, f0 ), z0 ) );
		__m128 res1 = _mm_add_ps( _mm_mul_ps( f1, h1 ), _mm_mul_ps( _mm_sub_ps( ones, f1 ), z1 ) );
		__m128 res2 = _mm_add_ps( _mm_mul_ps( f2, h2 ), _mm_mul_ps( _mm_sub_ps( ones, f2 ), z2 ) );
		__m128 res3 = _mm_add_ps( _mm_mul_ps( f3, h3 ), _mm_mul_ps( _mm_sub_ps( ones, f3 ), z3 ) );

		StoreSse4( res0, res );
		StoreSse4( res1, res + 4 );
		StoreSse4( res2, res + 8 );
		StoreSse4( res3, res + 12 );
		res += 16;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		__m128 z0 = LoadSse4( z );
		z += 4;

		__m128 f0 = LoadSse4( f );
		f += 4;

		__m128 h0 = LoadSse4( h );
		h += 4;

		__m128 res0 = _mm_add_ps( _mm_mul_ps( f0, h0 ), _mm_mul_ps( _mm_sub_ps( ones, f0 ), z0 ) );
		StoreSse4( res0, res );
		res += 4;

		sseSize--;
	}

	if( nonSseSize > 0 ) {
		__m128 z0 = LoadSse( z, nonSseSize );
		__m128 f0 = LoadSse( f, nonSseSize );
		__m128 h0 = LoadSse( h, nonSseSize );
		__m128 res0 = _mm_add_ps( _mm_mul_ps( f0, h0 ), _mm_mul_ps( _mm_sub_ps( ones, f0 ), z0 ) );
		StoreSse( res0, res, nonSseSize );
	}
}

// res = f * h + i * z
// where h is res of previous step
static inline void qrnnIfPoolingStep( const float* z, const float* f, const float* i, const float* h,
	float* res, int sseSize, int nonSseSize )
{
	while( sseSize >= 4 ) {
		__m128 z0 = LoadSse4( z );
		__m128 z1 = LoadSse4( z + 4 );
		__m128 z2 = LoadSse4( z + 8 );
		__m128 z3 = LoadSse4( z + 12 );
		z += 16;

		__m128 f0 = LoadSse4( f );
		__m128 f1 = LoadSse4( f + 4 );
		__m128 f2 = LoadSse4( f + 8 );
		__m128 f3 = LoadSse4( f + 12 );
		f += 16;

		__m128 i0 = LoadSse4( i );
		__m128 i1 = LoadSse4( i + 4 );
		__m128 i2 = LoadSse4( i + 8 );
		__m128 i3 = LoadSse4( i + 12 );
		i += 16;

		__m128 h0 = LoadSse4( h );
		__m128 h1 = LoadSse4( h + 4 );
		__m128 h2 = LoadSse4( h + 8 );
		__m128 h3 = LoadSse4( h + 12 );
		h += 16;

		__m128 res0 = _mm_add_ps( _mm_mul_ps( f0, h0 ), _mm_mul_ps( i0, z0 ) );
		__m128 res1 = _mm_add_ps( _mm_mul_ps( f1, h1 ), _mm_mul_ps( i1, z1 ) );
		__m128 res2 = _mm_add_ps( _mm_mul_ps( f2, h2 ), _mm_mul_ps( i2, z2 ) );
		__m128 res3 = _mm_add_ps( _mm_mul_ps( f3, h3 ), _mm_mul_ps( i3, z3 ) );

		StoreSse4( res0, res );
		StoreSse4( res1, res + 4 );
		StoreSse4( res2, res + 8 );
		StoreSse4( res3, res + 12 );
		res += 16;

		sseSize -= 4;
	}

	while( sseSize > 0 ) {
		__m128 z0 = LoadSse4( z );
		z += 4;

		__m128 f0 = LoadSse4( f );
		f += 4;

		__m128 i0 = LoadSse4( i );
		i += 4;

		__m128 h0 = LoadSse4( h );
		h += 4;

		__m128 res0 = _mm_add_ps( _mm_mul_ps( f0, h0 ), _mm_mul_ps( i0, z0 ) );
		StoreSse4( res0, res );
		res += 4;

		sseSize--;
	}

	if( nonSseSize > 0 ) {
		__m128 z0 = LoadSse( z, nonSseSize );
		__m128 f0 = LoadSse( f, nonSseSize );
		__m128 i0 = LoadSse( i, nonSseSize );
		__m128 h0 = LoadSse( h, nonSseSize );
		__m128 res0 = _mm_add_ps( _mm_mul_ps( f0, h0 ), _mm_mul_ps( i0, z0 ) );
		StoreSse( res0, res, nonSseSize );
	}
}

inline void vectorMinMax( const float* first, float* result, const float minValue, const float maxValue, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	const __m128 minSse = _mm_set_ps1(minValue);
	const __m128 maxSse = _mm_set_ps1(maxValue);

	auto minMaxWorker = [&minSse, &maxSse](const __m128& value) -> __m128 {
		return _mm_min_ps(maxSse, _mm_max_ps(minSse, value));
	};

	if(sseSize > 0) {
		for(int i = 0; i < sseSize; ++i) {
			StoreSse4(minMaxWorker(LoadSse4(first)), result);
			first += 4;
			result += 4;
		}
	}

	if(nonSseSize > 0) {
		StoreSse(minMaxWorker(LoadSse(first, nonSseSize)), result, nonSseSize);
	}
}

inline void vectorTanh( const float* first, float* result, int vectorSize )
{
#ifdef NEOML_USE_MKL
	vsTanh( vectorSize, first, result );
#else
	for( int i = 0; i < vectorSize; ++i ) {
		result[i] = -1.f + 2 / ( 1.f + ExponentFunc( -2 * first[i] ) );
	}
#endif
}

inline void vectorExp( const float* first, float* result, int vectorSize )
{
#ifdef NEOML_USE_MKL
	vectorMinMax( first, result, FLT_MIN_LOG, FLT_MAX_LOG, vectorSize );
	vsExp( vectorSize, result, result );
#else
	for( int i = 0; i < vectorSize; ++i ) {
		result[i] = ExponentFunc( first[i] );
	}
#endif
}

inline void vectorSigmoid( const float* first, float* result, int vectorSize )
{
	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	vectorExp( first, result, vectorSize );

	if( sseSize > 0 ) {
		const __m128 oneSse = _mm_set_ps1( 1 );
		for( int i = 0; i < sseSize; ++i ) {
			__m128 value = _mm_loadu_ps( result );
			value = _mm_div_ps( value, _mm_add_ps( value, oneSse ) );
			_mm_storeu_ps( result, value );

			result += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result = *result / ( *result + 1 );
		++result;
	}
}

} // namespace NeoML

#endif
