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

namespace NeoML {

// This file contains some template declarations
// The specializations are provided in platform-specific headers

// --------------------------------------------------------------------------------------------------------------------
// Data type
// Structure used in Simd which contains 4 elements of type T
template<class T>
struct CSimd4Struct;

template<class T>
using CSimd4 = typename CSimd4Struct<T>::Type;

// --------------------------------------------------------------------------------------------------------------------
// Basic functions

// Returns CSimd4 filled with a given value
template<class T>
CSimd4<T> SimdFill( T value ) = delete;

// Loads 4 elements into CSimd4 from the pointer
template<class T>
CSimd4<T> SimdLoad4( const T* src ) = delete;

// Loads count elements into CSimd4 from the pointer
// The rest of the CSimd4 is filled with defaultValue
template<class T>
CSimd4<T> SimdLoad( const T* src, int count, T defaultValue ) = delete;

// Writes 4 elements from CSimd4 to the pointer
template<class T>
void SimdStore4( T* dst, const CSimd4<T>& value ) = delete;

// Writes count elements from CSimd4 to the pointer
template<class T>
void SimdStore( T* dst, const CSimd4<T>& value, int count ) = delete;

// --------------------------------------------------------------------------------------------------------------------
// In this context functor is an entity with () operator performing its function over single SIMD element
// CFunctorBase provides public aliases for argument types
// Supports up to 3 input arguments
// When partially specialized transfers last given argument type to the rest of the arguments
template<class TResultArg, class TFirstArg = TResultArg, class TSecondArg = TFirstArg, class TThirdArg = TSecondArg>
class CFunctorBase {
public:
	using TResult = TResultArg;
	using TFirst = TFirstArg;
	using TSecond = TSecondArg;
	using TThird = TThirdArg;
};

// --------------------------------------------------------------------------------------------------------------------
// Macro for easy vectorization-friendly load/store operations
#define LOAD_4_SIMD4( type, varPrefix, ptr ) \
	CSimd4<type> varPrefix##0 = SimdLoad4( ptr + 4 * 0 ); \
	CSimd4<type> varPrefix##1 = SimdLoad4( ptr + 4 * 1 ); \
	CSimd4<type> varPrefix##2 = SimdLoad4( ptr + 4 * 2 ); \
	CSimd4<type> varPrefix##3 = SimdLoad4( ptr + 4 * 3 )

#define STORE_4_SIMD4( varPrefix, ptr ) \
	SimdStore4( ptr + 4 * 0, varPrefix##0 ); \
	SimdStore4( ptr + 4 * 1, varPrefix##1 ); \
	SimdStore4( ptr + 4 * 2, varPrefix##2 ); \
	SimdStore4( ptr + 4 * 3, varPrefix##3 )

// --------------------------------------------------------------------------------------------------------------------
// Class that wraps binary vector functor into the interface
// which takes 3 pointers and the number of elements.
template<class TFunctor>
class CBinaryVectorFunction {
public:
	using TFirst = typename TFunctor::TFirst;
	using TSecond = typename TFunctor::TSecond;
	using TResult = typename TFunctor::TResult;
	CBinaryVectorFunction( const TFunctor& functor = TFunctor(), TFirst firstDefaultValue = 1, TSecond secondDefaultValue = 1 ) :
		functor( functor ), firstDefaultValue( firstDefaultValue ), secondDefaultValue( secondDefaultValue ) {}

	void operator()( const TFirst* first, const TSecond* second, TResult* result, int vectorSize )
	{
		int simdSize = vectorSize / 4;
		int nonSimdSize = vectorSize % 4;

		// Ugly code for vectorization
		while( simdSize >= 4 ) {
			LOAD_4_SIMD4( TFirst, first, first );
			first += 16;
			LOAD_4_SIMD4( TSecond, second, second );
			second += 16;

			CSimd4<TResult> result0 = functor( first0, second0 );
			CSimd4<TResult> result1 = functor( first1, second1 );
			CSimd4<TResult> result2 = functor( first2, second2 );
			CSimd4<TResult> result3 = functor( first3, second3 );

			STORE_4_SIMD4( result, result );
			result += 16;
			simdSize -= 4;
		}

		while( simdSize > 0 ) {
			SimdStore4( result, functor( SimdLoad4( first ), SimdLoad4( second ) ) );
			first += 4;
			second += 4;
			result += 4;
			--simdSize;
		}

		if( nonSimdSize > 0 ) {
			const CSimd4<TFirst> simdFirst = SimdLoad( first, nonSimdSize, firstDefaultValue );
			const CSimd4<TSecond> simdSecond = SimdLoad( second, nonSimdSize, secondDefaultValue );
			CSimd4<TResult> simdResult = functor( simdFirst, simdSecond );
			SimdStore( result, simdResult, nonSimdSize );
		}
	}

private:
	TFunctor functor; // Functor to be applied
	TFirst firstDefaultValue; // Filler for the unused elements of the first argument SIMD
	TSecond secondDefaultValue; // Filler for the unused elements of the second argument SIMD
};

// Class that wraps ternary vector functor into the interface
// which takes 4 pointers and the number of elements.
template<class TFunctor>
class CTernaryVectorFunction {
public:
	using TFirst = typename TFunctor::TFirst;
	using TSecond = typename TFunctor::TSecond;
	using TThird = typename TFunctor::TThird;
	using TResult = typename TFunctor::TResult;
	CTernaryVectorFunction( const TFunctor& functor = TFunctor(), TFirst firstDefaultValue = 1,
		TSecond secondDefaultValue = 1, TThird thirdDefaultValue = 1 ) :
		functor( functor ),
		firstDefaultValue( firstDefaultValue ),
		secondDefaultValue( secondDefaultValue ),
		thirdDefaultValue( thirdDefaultValue )
	{}

	void operator()( const TFirst* first, const TSecond* second, const TThird* third, TResult* result, int vectorSize )
	{
		int simdSize = vectorSize / 4;
		int nonSimdSize = vectorSize % 4;

		// Ugly code for vectorization
		while( simdSize >= 4 ) {
			LOAD_4_SIMD4( TFirst, first, first );
			first += 16;
			LOAD_4_SIMD4( TSecond, second, second );
			second += 16;
			LOAD_4_SIMD4( TThird, third, third );
			third += 16;

			CSimd4<TResult> result0 = functor( first0, second0, third0 );
			CSimd4<TResult> result1 = functor( first1, second1, third1 );
			CSimd4<TResult> result2 = functor( first2, second2, third2 );
			CSimd4<TResult> result3 = functor( first3, second3, third3 );

			STORE_4_SIMD4( result, result );
			result += 16;
			simdSize -= 4;
		}

		while( simdSize > 0 ) {
			SimdStore4( result, functor( SimdLoad4( first ), SimdLoad4( second ), SimdLoad4( third ) ) );
			first += 4;
			second += 4;
			third += 4;
			result += 4;
			--simdSize;
		}

		if( nonSimdSize > 0 ) {
			const CSimd4<TFirst> simdFirst = SimdLoad( first, nonSimdSize, firstDefaultValue );
			const CSimd4<TSecond> simdSecond = SimdLoad( second, nonSimdSize, secondDefaultValue );
			const CSimd4<TThird> simdThird = SimdLoad( third, nonSimdSize, thirdDefaultValue );
			CSimd4<TResult> simdResult = functor( simdFirst, simdSecond, simdThird );
			SimdStore( result, simdResult, nonSimdSize );
		}
	}

private:
	TFunctor functor; // Functor to be applied
	TFirst firstDefaultValue; // Filler for the unused elements of the first argument SIMD
	TSecond secondDefaultValue; // Filler for the unused elements of the second argument SIMD
	TSecond thirdDefaultValue; // Filler for the unused elements of the third argument SIMD
};

} // namespace NeoML
