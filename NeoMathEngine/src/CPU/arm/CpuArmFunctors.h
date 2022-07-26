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

#ifdef NEOML_USE_NEON

#include "CpuArm.h"

namespace NeoML {

// data type
template<class T>
struct CSimd4Struct;

template<>
struct CSimd4Struct<float> {
	typedef float32x4_t Type;
};

template<>
struct CSimd4Struct<int> {
	typedef int32x4_t Type;
};

template<class T>
using CSimd4 = typename CSimd4Struct<T>::Type;

// Creates a single 4-element simd filled with a given value
template<class T>
CSimd4<T> SimdFill( T value ) = delete;

template<>
CSimd4<float> SimdFill( float value )
{
	return vdupq_n_f32( value );
}

template<>
CSimd4<int> SimdFill( int value )
{
	return vdupq_n_s32( value );
}

// Loads full simd4
template<class T>
CSimd4<T> SimdLoad4( const T* src ) = delete;

template<>
CSimd4<float> SimdLoad4<float>( const float* src )
{
	return LoadNeon4( src );
}

template<>
CSimd4<int> SimdLoad4<int>( const int* src )
{
	return LoadIntNeon4( src );
}

// Loads part of simd4
template<class T>
CSimd4<T> SimdLoad( const T* src, int count, T defaultValue ) = delete;

template<>
CSimd4<float> SimdLoad( const float* src, int count, float defaultValue )
{
	return LoadNeon( src, count, defaultValue );
}

template<>
CSimd4<int> SimdLoad( const int* src, int count, int defaultValue )
{
	return LoadIntNeon( src, count, defaultValue );
}

// Stores full simd4
template<class T>
void SimdStore4( T* dst, const CSimd4<T>& value ) = delete;

template<>
void SimdStore4<float>( float* dst, const CSimd4<float>& value )
{
	StoreNeon4( value, dst );
}

template<>
void SimdStore4<int>( int* dst, const CSimd4<int>& value )
{
	StoreIntNeon4( value, dst );
}

// Stores part of the simd4
template<class T>
void SimdStore( T* dst, const CSimd4<T>& value, int count ) = delete;

template<>
void SimdStore<float>( float* dst, const CSimd4<float>& value, int count )
{
	StoreNeon( value, dst, count );
}

template<>
void SimdStore<int>( int* dst, const CSimd4<int>& value, int count )
{
	StoreIntNeon( value, dst, count );
}

// Equal functor
// a == b ? 1 : 0
template<class T>
class CEqualFunctor;

template<>
class CEqualFunctor<float> {
public:
	using TFirst = float;
	using TSecond = float;
	using TResult = int;
	CSimd4<int> operator()( const CSimd4<float>& first, const CSimd4<float>& second )
		{ return vandq_s32( ones, vreinterpretq_s32_u32( vceqq_f32( first, second ) ) ); }

private:
	CSimd4<int> ones = SimdFill<int>( 1 );
};

template<>
class CEqualFunctor<int> {
public:
	using TFirst = int;
	using TSecond = int;
	using TResult = int;
	CSimd4<int> operator()( const CSimd4<int>& first, const CSimd4<int>& second )
		{ return vandq_s32( ones, vreinterpretq_s32_u32( vceqq_s32( first, second ) ) ); }

private:
	CSimd4<int> ones = SimdFill<int>( 1 );
};

// Where functor
// a != 0 ? b : c
template<class T>
class CWhereFunctor;

template<>
class CWhereFunctor<float> {
public:
	using TFirst = int;
	using TSecond = float;
	using TThird = float;
	using TResult = float;

	CSimd4<float> operator()( const CSimd4<int>& first, const CSimd4<float>& second, const CSimd4<float>& third )
	{
		const uint32x4_t mask = vceqzq_s32( first );
		return vreinterpretq_f32_u32( vorrq_u32( 
			vandq_u32( vmvnq_u32( mask ), vreinterpretq_u32_f32( second ) ),
			vandq_u32( mask, vreinterpretq_u32_f32( third ) )
		) );
	}
};

template<>
class CWhereFunctor<int> {
public:
	using TFirst = int;
	using TSecond = int;
	using TThird = int;
	using TResult = int;

	CSimd4<int> operator()( const CSimd4<int>& first, const CSimd4<int>& second, const CSimd4<int>& third )
	{
		const uint32x4_t mask = vceqzq_s32( first );
		return vreinterpretq_s32_u32( vorrq_u32( 
			vandq_u32( vmvnq_u32( mask ), vreinterpretq_u32_s32( second ) ),
			vandq_u32( mask, vreinterpretq_u32_s32( third ) )
		) );
	}
};

} // namespace NeoML

#endif // NEOML_USE_NEON
