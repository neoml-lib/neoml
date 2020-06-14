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

#include "../ErrorsFOL.h"
#include "../ArchiveFOL.h"
#include "HashTableFOL.h"

namespace FObj {

class CInterval {
public:
	int Begin;
	int End;

	CInterval();
	CInterval( int begin, int end );

	int HashKey() const;

	bool IsValid() const;
	bool IsEmpty() const;

	void SetEmpty();

	int Length() const;
	int Width() const;
	int Center() const;

	void operator &= ( const CInterval& other );
	void operator |= ( const CInterval& other );
	void operator += ( int offset );
	void operator -= ( int offset );
	void operator *= ( int factor );
	void operator /= ( int factor );

	bool Overlaps( const CInterval& other ) const;
	bool Contains( int x ) const;
	bool Contains( const CInterval& other ) const;
};

//------------------------------------------------------------------------------------------------------------

inline CInterval::CInterval() :
	Begin(0),
	End(0)
{
}

inline CInterval::CInterval( int begin, int end ) :
	Begin( begin ),
	End( end )
{
	PresumeFO( IsValid() );
}

inline int CInterval::HashKey() const
{
	int result = Begin;
	AddToHashKey( End, result );
	return result;
}

inline bool CInterval::IsValid() const
{
	return Begin <= End;
}

inline bool CInterval::IsEmpty() const
{
	PresumeFO( IsValid() );
	return Begin == End;
}

inline void CInterval::SetEmpty()
{
	Begin = 0;
	End = 0;
	PresumeFO( IsValid() );
}

inline int CInterval::Length() const
{
	PresumeFO( IsValid() );
	return End - Begin;
}

inline int CInterval::Width() const
{
	return Length();
}

inline int CInterval::Center() const
{
	PresumeFO( IsValid() );
	return ( Begin + End ) / 2; 
}

inline void CInterval::operator &= ( const CInterval& other )
{
	PresumeFO( other.IsValid() );
	PresumeFO( IsValid() );

	Begin = max( Begin, other.Begin );
	End = min( End, other.End );
	if( Begin >= End ) {
		SetEmpty();
	}
}

inline void CInterval::operator |= ( const CInterval& other )
{
	PresumeFO( other.IsValid() );
	PresumeFO( IsValid() );

	if( IsEmpty() ) {
		*this = other;
	} else if( other.IsEmpty() ) {
		;
	} else {
		Begin = min( Begin, other.Begin );
		End = max( End, other.End );
	}
}

inline void CInterval::operator += ( int offset )
{
	PresumeFO( IsValid() );

	Begin += offset;
	End += offset;
}

inline void CInterval::operator -= ( int offset )
{
	PresumeFO( IsValid() );

	Begin -= offset;
	End -= offset;
}

inline void CInterval::operator *= ( int factor )
{
	PresumeFO( factor >= 0 );
	PresumeFO( IsValid() );

	Begin *= factor;
	End *= factor;
}

inline void CInterval::operator /= ( int factor )
{
	PresumeFO( factor > 0 );
	PresumeFO( IsValid() );

	Begin /= factor;
	End /= factor;
}

inline bool CInterval::Overlaps( const CInterval& other ) const
{
	PresumeFO( other.IsValid() );
	PresumeFO( IsValid() );

	return Begin < other.End && End > other.Begin;
}

inline bool CInterval::Contains( int x ) const
{
	PresumeFO( Begin <= End );
	return Begin <= x && x < End;
}

inline bool CInterval::Contains( const CInterval& other ) const
{
	PresumeFO( other.IsValid() );
	PresumeFO( IsValid() );

	if( other.IsEmpty() ) {
		return true;
	}
	return Begin <= other.Begin && other.End <= End;
}

inline bool operator == ( const CInterval& a, const CInterval& b )
{
	PresumeFO( a.IsValid() );
	PresumeFO( b.IsValid() );

	if( a.IsEmpty() && b.IsEmpty() ) {
		return true;
	}

	return a.Begin == b.Begin && a.End == b.End;
}

inline bool operator != ( const CInterval& a, const CInterval& b )
{
	return !(a == b);
}

inline CInterval operator & ( const CInterval& a, const CInterval& b )
{
	CInterval result(a);
	result &= b;
	return result;
}

inline CInterval operator | ( const CInterval& a, const CInterval& b )
{
	CInterval result(a);
	result |= b;
	return result;
}

inline CInterval operator + ( const CInterval& interval, int offset )
{
	CInterval result( interval );
	result += offset;
	return result;
}

inline CInterval operator + ( int offset, const CInterval& interval )
{
	return interval + offset;
}

inline CInterval operator - ( const CInterval& interval, int offset )
{
	CInterval result( interval );
	result -= offset;
	return result;
}

inline CInterval operator * ( const CInterval& interval, int factor )
{
	CInterval result( interval );
	result *= factor;
	return result;
}

inline CInterval operator * ( int factor, const CInterval& interval )
{
	return interval * factor;
}

inline CInterval operator / ( const CInterval& interval, int factor )
{
	CInterval result( interval );
	result /= factor;
	return result;
}

inline int Gap( const CInterval& a, const CInterval& b )
{
	PresumeFO( a.IsValid() );
	PresumeFO( b.IsValid() );
	PresumeFO( !a.IsEmpty() );
	PresumeFO( !b.IsEmpty() );

	return max( b.Begin - a.End, a.Begin - b.End );
}

inline CArchive& operator<<( CArchive& archive, const CInterval& interval )
{
	archive << interval.Begin;
	archive << interval.End;
	return archive;
}

inline CArchive& operator>>( CArchive& archive, CInterval& interval )
{
	archive >> interval.Begin;
	archive >> interval.End;
	return archive;
}

} // namespace FObj
