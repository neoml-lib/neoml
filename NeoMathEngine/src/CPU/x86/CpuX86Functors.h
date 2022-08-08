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
#include "../CpuFunctorCommon.h"

namespace NeoML {

// Data types
template<>
struct CSimd4Struct<float> {
	typedef __m128 Type;
};

template<>
struct CSimd4Struct<int> {
	typedef __m128i Type;
};

// Fill functions
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

// Load functions
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

// Store functions
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

// --------------------------------------------------------------------------------------------------------------------
// Equal functor
// a == b ? 1 : 0
template<class T>
class CEqualFunctor : public CFunctorBase<int, T> {
public:
	CSimd4<int> operator()( const CSimd4<T>& first, const CSimd4<T>& second );

private:
	CSimd4<int> ones = SimdFill( 1 );
};

template<>
CSimd4<int> CEqualFunctor<float>::operator()( const CSimd4<float>& first, const CSimd4<float>& second )
{
	return _mm_and_si128( ones, _mm_castps_si128( _mm_cmpeq_ps( first, second ) ) );
}

template<>
CSimd4<int> CEqualFunctor<int>::operator()( const CSimd4<int>& first, const CSimd4<int>& second )
{
	return _mm_and_si128( ones, _mm_cmpeq_epi32( first, second ) );
}

// --------------------------------------------------------------------------------------------------------------------
// Where functor
// a != 0 ? b : c
template<class T>
class CWhereFunctor : public CFunctorBase<T, int, T> {
public:
	CSimd4<T> operator()( const CSimd4<int>& first, const CSimd4<T>& second, const CSimd4<T>& third );

private:
	CSimd4<int> zeros = SimdFill( 0 );
};

template<>
CSimd4<float> CWhereFunctor<float>::operator()( const CSimd4<int>& first, const CSimd4<float>& second,
	const CSimd4<float>& third )
{
	const __m128 mask = _mm_castsi128_ps( _mm_cmpeq_epi32( first, zeros ) );
	return _mm_or_ps( _mm_andnot_ps( mask, second ), _mm_and_ps( mask, third ) );
}

template<>
CSimd4<int> CWhereFunctor<int>::operator()( const CSimd4<int>& first, const CSimd4<int>& second,
	const CSimd4<int>& third )
{
	const __m128i mask = _mm_cmpeq_epi32( first, zeros );
	return _mm_or_si128( _mm_andnot_si128( mask, second ), _mm_and_si128( mask, third ) );
}

} // namespace NeoML

#endif // NEOML_USE_SSE