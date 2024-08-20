/* Copyright © 2024 ABBYY

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

#include <MapFOL.h>

#pragma once

namespace FObj {

//------------------------------------------------------------------------------------------------------------

template<class TMap>
class CMapPositionIterator {
public:
	CMapPositionIterator() : map( 0 ), pos( NotFound ) {}
	CMapPositionIterator( const TMap* _map, TMapPosition _pos ) : map( _map ), pos( _pos ) {}

	bool operator==( const CMapPositionIterator& other ) const;
	bool operator!=( const CMapPositionIterator& other ) const { return !( *this == other ); }
	CMapPositionIterator& operator++();
	CMapPositionIterator operator++( int );
	TMapPosition operator*() const { return pos; }

private:
	const TMap* map;
	TMapPosition pos;

	const TMap& safeMap() const { PresumeFO( map != 0 ); return *map; }
};

//------------------------------------------------------------------------------------------------------------

template<class TMap>
inline bool CMapPositionIterator<TMap>::operator==( const CMapPositionIterator& other ) const
{
	PresumeFO( map == other.map );
	return pos == other.pos;
}

template<class TMap>
inline CMapPositionIterator<TMap>& CMapPositionIterator<TMap>::operator++()
{
	pos = safeMap().GetNextPosition( pos );
	return *this;
}

template<class TMap>
inline CMapPositionIterator<TMap> CMapPositionIterator<TMap>::operator++( int )
{
	const CMapPositionIterator old( *this );
	pos = safeMap().GetNextPosition( pos );
	return old;
}

//------------------------------------------------------------------------------------------------------------

template<class TMap>
class CMapPositionRange {
public:
	typedef CMapPositionIterator<TMap> TConstIterator;
	typedef TConstIterator TIterator;

	CMapPositionRange( const TMap& _map ) : map( _map ) {}

	TConstIterator begin() const { return TConstIterator( &map, map.GetFirstPosition() ); }
	TConstIterator end() const { return TConstIterator( &map, NotFound ); }

private:
	const TMap& map;
};

//------------------------------------------------------------------------------------------------------------

template<class TKey, class TValue, class TKeyHashInfo, class TAllocator>
inline CMapPositionRange< CMap<TKey, TValue, TKeyHashInfo, TAllocator> > MapPositionRange( 
	const CMap<TKey, TValue, TKeyHashInfo, TAllocator>& map )
{
	return CMapPositionRange< CMap<TKey, TValue, TKeyHashInfo, TAllocator> >( map );
}

} // namespace FObj
