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

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_SSE
#if FINE_PLATFORM( FINE_WINDOWS )
#include <intrin.h>
#elif FINE_PLATFORM(FINE_LINUX) || FINE_PLATFORM(FINE_DARWIN) || FINE_PLATFORM(FINE_ANDROID) || FINE_PLATFORM(FINE_IOS)
#include <x86intrin.h>
#else
#error "Platform isn't supported!"
#endif
#include <xmmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <cmath>
#include <cfloat>
#include <cassert>

#include <NeoMathEngine/CrtAllocatedObject.h>

namespace NeoML {

// define for logarithms FLT_MIN/MAX. define is used to avoid problems with CUDA
#define FLT_MIN_LOG -87.336544f
#define FLT_MAX_LOG 88.f

// Split the vector length into registers of 4 elements each + the remainder
inline int GetCount4(int& size)
{
	int ret = size / 4;
	size %= 4;
	return ret;
}

// Exponent with limitations to avoid NaN
inline float ExponentFunc(float f)
{
	if(f < FLT_MIN_LOG) {
		return 0;
	} else if(f > FLT_MAX_LOG) {
		return FLT_MAX;
	} else {
		return expf(f);
	}
}

// Loading and unloading SSE registers
inline __m128 LoadSse4(const float* data)
{
	return _mm_loadu_ps(data);
}

inline __m128 LoadSse(const float* data, int count, float defVal = 0)
{
	assert( count >= 1 && count <= 3 );
	switch(count) {
		case 1:
			return _mm_set_ps(defVal, defVal, defVal, data[0]);
		case 2:
			return _mm_set_ps(defVal, defVal, data[1], data[0]);
	}
	return _mm_set_ps(defVal, data[2], data[1], data[0]);
}

inline __m128i LoadIntSse4(const int* data)
{
	return _mm_loadu_si128((const __m128i*)data);
}

inline __m128i LoadIntSse(const int* data, int count, int defVal = 0)
{
	assert( count >= 1 && count <= 3 );
	switch(count) {
		case 1:
			return _mm_set_epi32(defVal, defVal, defVal, data[0]);
		case 2:
			return _mm_set_epi32(defVal, defVal, data[1], data[0]);
	}
	return _mm_set_epi32(defVal, data[2], data[1], data[0]);
}

inline void StoreSse4(const __m128&val, float* data)
{
	_mm_storeu_ps(data, val);
}

inline void StoreSse(__m128 val, float* data, int count)
{
	switch(count) {
	default:
	case 0:
		break;
	case 1:
		*data = _mm_cvtss_f32(val);
		break;
	case 2:
		*data++ = _mm_cvtss_f32(val);
		val = _mm_shuffle_ps(val, val, _MM_SHUFFLE(0, 3, 2, 1));
		*data = _mm_cvtss_f32(val);
		break;
	case 3:
		*data++ = _mm_cvtss_f32(val);
		val = _mm_shuffle_ps(val, val, _MM_SHUFFLE(0, 3, 2, 1));
		*data++ = _mm_cvtss_f32(val);
		val = _mm_shuffle_ps(val, val, _MM_SHUFFLE(0, 3, 2, 1));
		*data = _mm_cvtss_f32(val);
		break;
	}
}

inline void StoreIntSse4(const __m128i& val, int* data)
{
	_mm_storeu_si128((__m128i*)data, val);
}

inline void StoreIntSse(__m128i val, int* data, int count)
{
	switch(count) {
		default:
		case 0:
			break;
		case 1:
			*data = _mm_cvtsi128_si32(val);
			break;
		case 2:
			*data++ = _mm_cvtsi128_si32(val);
			val = _mm_shuffle_epi32(val, _MM_SHUFFLE(0, 3, 2, 1));
			*data = _mm_cvtsi128_si32(val);
			break;
		case 3:
			*data++ = _mm_cvtsi128_si32(val);
			val = _mm_shuffle_epi32(val, _MM_SHUFFLE(0, 3, 2, 1));
			*data++ = _mm_cvtsi128_si32(val);
			val = _mm_shuffle_epi32(val, _MM_SHUFFLE(0, 3, 2, 1));
			*data = _mm_cvtsi128_si32(val);
			break;
	}
}

#define SSE_LOAD_16_FLOATS(varPrefix, src) \
    __m128 varPrefix##0 = LoadSse4(src + 4 * 0); \
    __m128 varPrefix##1 = LoadSse4(src + 4 * 1); \
    __m128 varPrefix##2 = LoadSse4(src + 4 * 2); \
    __m128 varPrefix##3 = LoadSse4(src + 4 * 3);

#define SSE_STORE_16_FLOATS(varPrefix, dst) \
    StoreSse4(varPrefix##0, dst + 4 * 0); \
    StoreSse4(varPrefix##1, dst + 4 * 1); \
    StoreSse4(varPrefix##2, dst + 4 * 2); \
    StoreSse4(varPrefix##3, dst + 4 * 3);

#define SSE_LOAD_16_INTS(varPrefix, src) \
    __m128i varPrefix##0 = LoadIntSse4(src + 4 * 0); \
    __m128i varPrefix##1 = LoadIntSse4(src + 4 * 1); \
    __m128i varPrefix##2 = LoadIntSse4(src + 4 * 2); \
    __m128i varPrefix##3 = LoadIntSse4(src + 4 * 3);

#define SSE_STORE_16_INTS(varPrefix, dst) \
    StoreIntSse4(varPrefix##0, dst + 4 * 0); \
    StoreIntSse4(varPrefix##1, dst + 4 * 1); \
    StoreIntSse4(varPrefix##2, dst + 4 * 2); \
    StoreIntSse4(varPrefix##3, dst + 4 * 3);

inline __m128 GetPhaseMask4(int phase)
{
	switch(phase) {
		default:
		case 0:
			return _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, -1));
		case 1:
			return _mm_castsi128_ps(_mm_set_epi32(0, 0, -1, 0));
		case 2:
			return _mm_castsi128_ps(_mm_set_epi32(0, -1, 0, 0));
		case 3:
			return _mm_castsi128_ps(_mm_set_epi32(-1, 0, 0, 0));
	}
}

// Sets all register fields to a value from another register
inline __m128 GetPhaseValue4(const __m128& value, int phase)
{
	switch(phase) {
		default:
		case 0:
			return _mm_shuffle_ps(value, value, _MM_SHUFFLE(0, 0, 0, 0));
		case 1:
			return _mm_shuffle_ps(value, value, _MM_SHUFFLE(1, 1, 1, 1));
		case 2:
			return _mm_shuffle_ps(value, value, _MM_SHUFFLE(2, 2, 2, 2));
		case 3:
			return _mm_shuffle_ps(value, value, _MM_SHUFFLE(3, 3, 3, 3));
	}
}

// Adds up all register fields, writes the result into all fields
inline __m128 HorizontalAddSse(const __m128& value)
{
	__m128 res = _mm_shuffle_ps(value, value, _MM_SHUFFLE(1, 0, 3, 2));
	res = _mm_add_ps(res, value);
	__m128 val = _mm_shuffle_ps(res, res, _MM_SHUFFLE(2, 3, 0, 1));
	res = _mm_add_ps(res, val);
	return res;
}

// Finds the maximum over the register fields, writes the result into all fields
inline __m128 HorizontalMaxSse(const __m128& value)
{
	__m128 res = _mm_shuffle_ps(value, value, _MM_SHUFFLE(1, 0, 3, 2));
	res = _mm_max_ps(res, value);
	__m128 val = _mm_shuffle_ps(res, res, _MM_SHUFFLE(2, 3, 0, 1));
	res = _mm_max_ps(res, val);
	return res;
}

// The auxiliary class for a matrix block
class alignas(16) CMatrixBlock4x4 : public CCrtAllocatedObject {
public:
	__m128 Rows[4];

	void Load4x4(const float* data, int rowSize)
	{
		Rows[0] = LoadSse4(data);
		data += rowSize;
		Rows[1] = LoadSse4(data);
		data += rowSize;
		Rows[2] = LoadSse4(data);
		data += rowSize;
		Rows[3] = LoadSse4(data);
	}

	void Load4xX(const float* data, int width, int rowSize)
	{
		Rows[0] = LoadSse(data, width);
		data += rowSize;
		Rows[1] = LoadSse(data, width);
		data += rowSize;
		Rows[2] = LoadSse(data, width);
		data += rowSize;
		Rows[3] = LoadSse(data, width);
	}

	void LoadYx4(const float* data, int height, int rowSize)
	{
		for(int j = 0; j < height; ++j) {
			Rows[j] = LoadSse4(data);
			data += rowSize;
		}
		for(int j = height; j < 4; ++j) {
			Rows[j] = _mm_setzero_ps();
		}
	}

	void LoadYxX(const float* data, int height, int width, int rowSize)
	{
		for(int j = 0; j < height; ++j) {
			Rows[j] = LoadSse(data, width);
			data += rowSize;
		}
		for(int j = height; j < 4; ++j) {
			Rows[j] = _mm_setzero_ps();
		}
	}

	void Store4x4(float* data, int rowSize) const
	{
		StoreSse4(Rows[0], data);
		data += rowSize;
		StoreSse4(Rows[1], data);
		data += rowSize;
		StoreSse4(Rows[2], data);
		data += rowSize;
		StoreSse4(Rows[3], data);
	}

	void Store4xX(float* data, int width, int rowSize) const
	{
		StoreSse(Rows[0], data, width);
		data += rowSize;
		StoreSse(Rows[1], data, width);
		data += rowSize;
		StoreSse(Rows[2], data, width);
		data += rowSize;
		StoreSse(Rows[3], data, width);
	}

	void StoreYx4(float* data, int height, int rowSize) const
	{
		for(int j = 0; j < height; ++j) {
			StoreSse4(Rows[j], data);
			data += rowSize;
		}
	}

	void StoreYxX(float* data, int height, int width, int rowSize) const
	{
		for(int j = 0; j < height; ++j) {
			StoreSse(Rows[j], data, width);
			data += rowSize;
		}
	}

	void Transpose()
	{
		_MM_TRANSPOSE4_PS(Rows[0], Rows[1], Rows[2], Rows[3]);
	}
};

//------------------------------------------------------------------------------------------------------------

inline void checkSse(int size, int& sseSize, int& nonSseSize)
{
	sseSize = size / 4;
	nonSseSize = size % 4;
}

inline void checkSse2(int size, int& sseSize, int& nonSseSize)
{
	return checkSse(size, sseSize, nonSseSize);
}

inline void dataCopy(float* dst, const float* src, int vectorSize)
{
	static_assert( sizeof(float) == sizeof(unsigned int), "Size of float isn't equal to size of unsigned int." );

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	while( sseSize >= 4 ) {
		_mm_storeu_ps(dst, _mm_loadu_ps(src));
		dst += 4;
		src += 4;
		_mm_storeu_ps(dst, _mm_loadu_ps(src));
		dst += 4;
		src += 4;
		_mm_storeu_ps(dst, _mm_loadu_ps(src));
		dst += 4;
		src += 4;
		_mm_storeu_ps(dst, _mm_loadu_ps(src));
		dst += 4;
		src += 4;

		sseSize -= 4;
	}

	for(int i = 0; i < sseSize; ++i) {
		_mm_storeu_ps(dst, _mm_loadu_ps(src));
		dst += 4;
		src += 4;
	}

	switch( nonSseSize ) {
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

inline void dataCopy(int* dst, const int* src, int vectorSize)
{
	int sseSize;
	int nonSseSize;
	checkSse2(vectorSize, sseSize, nonSseSize);

	while( sseSize >= 4 ) {
		_mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((__m128i*)src));
		dst += 4;
		src += 4;
		_mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((__m128i*)src));
		dst += 4;
		src += 4;
		_mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((__m128i*)src));
		dst += 4;
		src += 4;
		_mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((__m128i*)src));
		dst += 4;
		src += 4;

		sseSize -= 4;
	}

	for(int i = 0; i < sseSize; ++i) {
		_mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((const __m128i*)src));
		dst += 4;
		src += 4;
	}

	switch( nonSseSize ) {
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

inline float euclidianNoSSE( const float* x, const float* y, const int size )
{
	float result = 0.f;
	for( int i = 0; i < size; ++i ) {
		const float num = x[i] - y[i];
		result += num * num;
	}
	return result;
}

inline float euclidianSSE( const float* x, const float* y, const int size )
{
	float result = 0;

	int sseSize = size / 4;
	int nonSseSize = size % 4;

	__m128 euclidean = _mm_setzero_ps();
	for( int i = 0; i < sseSize; i++ ) {
		const __m128 sseX = _mm_loadu_ps( x );
		const __m128 sseY = _mm_loadu_ps( y );
		const __m128 diff = _mm_sub_ps( sseX, sseY );
		const __m128 diffSquared = _mm_mul_ps( diff, diff );
		euclidean = _mm_add_ps( euclidean, diffSquared );
		x += 4;
		y += 4;
	}
	// Merge the results
	const __m128 shuffle1 = _mm_shuffle_ps( euclidean, euclidean, _MM_SHUFFLE( 1, 0, 3, 2 ) );
	const __m128 part1 = _mm_add_ps( euclidean, shuffle1 );
	const __m128 shuffle2 = _mm_shuffle_ps( part1, part1, _MM_SHUFFLE( 2, 3, 0, 1 ) );
	const __m128 part2 = _mm_add_ps( part1, shuffle2 );

	_mm_store_ss( &result, part2 );

	if( nonSseSize > 0 ) {
		result += euclidianNoSSE( x, y, nonSseSize );
	}
	return result;
}

} // namespace NeoML

#endif // NEOML_USE_SSE
