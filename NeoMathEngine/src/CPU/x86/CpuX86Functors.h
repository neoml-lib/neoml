/* Copyright © 2017-2022 ABBYY Production LLC

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

#include "CpuX86.h"

namespace NeoML {

// data type
template<class T>
struct CSimd4Struct;

template<>
struct CSimd4Struct<float> {
	typedef __m128 Type;
};

template<>
struct CSimd4Struct<int> {
	typedef __m128i Type;
};

template<class T>
using CSimd4 = typename CSimd4Struct<T>::Type;

// Creates a single 4-element simd filled with a given value
template<class T>
CSimd4<T> SimdFill( T value ) = delete;

template<>
CSimd4<float> SimdFill( float value )
{
	return _mm_set1_ps( value );
}

template<>
CSimd4<int> SimdFill( int value )
{
	return _mm_set1_epi32( value );
}

// Loads full simd4
template<class T>
CSimd4<T> SimdLoad4( const T* src ) = delete;

template<>
CSimd4<float> SimdLoad4<float>( const float* src )
{
	return LoadSse4( src );
}

template<>
CSimd4<int> SimdLoad4<int>( const int* src )
{
	return LoadIntSse4( src );
}

// Loads part of simd4
template<class T>
CSimd4<T> SimdLoad( const T* src, int count, T defaultValue ) = delete;

template<>
CSimd4<float> SimdLoad( const float* src, int count, float defaultValue )
{
	return LoadSse( src, count, defaultValue );
}

template<>
CSimd4<int> SimdLoad( const int* src, int count, int defaultValue )
{
	return LoadIntSse( src, count, defaultValue );
}

// Stores full simd4
template<class T>
void SimdStore4( T* dst, const CSimd4<T>& value ) = delete;

template<>
void SimdStore4<float>( float* dst, const CSimd4<float>& value )
{
	StoreSse4( value, dst );
}

template<>
void SimdStore4<int>( int* dst, const CSimd4<int>& value )
{
	StoreIntSse4( value, dst );
}

// Stores part of the simd4
template<class T>
void SimdStore( T* dst, const CSimd4<T>& value, int count ) = delete;

template<>
void SimdStore<float>( float* dst, const CSimd4<float>& value, int count )
{
	StoreSse( value, dst, count );
}

template<>
void SimdStore<int>( int* dst, const CSimd4<int>& value, int count )
{
	StoreIntSse( value, dst, count );
}

// Equal functor
// a == b ? 1 : 0
template<class T>
class CEqualFunctor;

template<>
class CEqualFunctor<float> {
public:
	CSimd4<int> operator()( const CSimd4<float>& first, const CSimd4<float>& second )
		{ return _mm_and_si128( ones, _mm_castps_si128( _mm_cmpeq_ps( first, second ) ) ); }

private:
	CSimd4<int> ones = SimdFill<int>( 1 );
};

template<>
class CEqualFunctor<int> {
public:
	CSimd4<int> operator()( const CSimd4<int>& first, const CSimd4<int>& second )
		{ return _mm_and_si128( ones, _mm_cmpeq_epi32( first, second ) ); }

private:
	CSimd4<int> ones = SimdFill<int>( 1 );
};

} // namespace NeoML

#endif // NEOML_USE_SSE