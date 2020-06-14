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

#include <climits>

namespace FObj {

// These functions let you get the object address when its class has overloaded the &() operator
template<typename Type>
inline Type* AddressOfObject( Type& what )
{
	return reinterpret_cast<Type*>( &( reinterpret_cast<char&>( what ) ) );
}

template<typename Type>
inline Type* AddressOfObject( const Type& what )
{
	return const_cast<Type*>( reinterpret_cast<const Type*>( &( reinterpret_cast<const char&>( what ) ) ) );
}

const int Megabyte = 1024 * 1024;
const int Gigabyte = 1024 * Megabyte;
const double Pi = 3.1415926535897932384626433832795;

static const int primeList[] =
{
	31, 53, 97, 193, 389, 769,
	1543, 3079, 6151, 12289, 24593,
	49157, 98317, 196613, 393241, 786433,
	1572869, 3145739, 6291469, 12582917, 25165843,
	50331653, 100663319, 201326611, 402653189, 805306457, 1610612741
};

inline int UpperPrimeNumber( int number )
{
	for( int i = 0; i < _countof( primeList ); i++ ) {
		if( primeList[i] > number ) {
			return primeList[i];
		}
	}
	AssertFO( false );
	return INT_MAX;
}

// Arithmetics with range checking
// The first parameter of the template should be specified directly when called
// The second parameter will be determined by the function argument type
// For example:
//		int i = 10000;
//		short s = to<short>( i );

#pragma warning( push )
#pragma warning( disable : 4389 ) // '==' : signed/unsigned mismatch

template<class To, class From>
inline To to( From x )
{
	To result = static_cast<To>( x );
	PresumeFO( static_cast<From>( result ) == x ); // checks that the cast is reversible
	PresumeFO( ( x >= 0 ) == ( result >= 0 ) ); // sign check
	return result;
}

#pragma warning( pop )

// Turn on the specified flags in the set
inline void SetFlags( DWORD& set, DWORD flags ) 
{
	set |= flags;
}

// Checks if at least one of the specified flags is present in the set
inline bool HasFlag( DWORD set, DWORD flag )
{
	return ( set & flag ) != 0;
}

inline int Ceil( int val, int discret )
{
	PresumeFO( discret > 0 );
	if( val > 0 ) {
		return ( val + discret - 1 ) / discret;
	}
	return val / discret;
}

inline int CeilTo( int val, int discret )
{
	return Ceil( val, discret ) * discret;
}

inline int Round( double d )
{
	const double result = d > 0 ? ( d + 0.5 ) : ( d - 0.5 );
	PresumeFO( INT_MIN <= result && result <= INT_MAX );
	return static_cast<int>( result );
}

const int NotFound = -1;

using std::max;
using std::min;
using std::sqrt;
using std::abs;

} // namespace FObj
