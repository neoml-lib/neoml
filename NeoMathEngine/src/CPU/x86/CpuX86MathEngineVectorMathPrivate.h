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

#include <CpuX86.h>

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

} // namespace NeoML

#endif
