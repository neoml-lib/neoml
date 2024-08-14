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

#pragma once

namespace FObj {

typedef int TMapPosition;
typedef TMapPosition MAP_POSITION;

template<class TMap>
class CMapIterator;

//------------------------------------------------------------------------------------------------------------

template<class TMap>
class CConstMapIterator {
public:
	CConstMapIterator() : map( 0 ), pos( NotFound ) {}
	CConstMapIterator( const TMap* _map, TMapPosition _pos ) : map( _map ), pos( _pos ) {}
	CConstMapIterator( const CMapIterator<TMap>& other ) : map( other.map ), pos( other.pos ) {}

	bool operator==( const CConstMapIterator& other ) const;
	bool operator!=( const CConstMapIterator& other ) const { return !( *this == other ); }
	bool operator==( const CMapIterator<TMap>& other ) const;
	bool operator!=( const CMapIterator<TMap>& other ) const { return !( *this == other ); }
	CConstMapIterator& operator++();
	CConstMapIterator operator++( int );
	const typename TMap::TElement& operator*() const { return safeMap().GetKeyValue( pos ); }
	const typename TMap::TElement* operator->() const { return &safeMap().GetKeyValue( pos ); }

	friend class CMapIterator<TMap>;

private:
	const TMap* map;
	TMapPosition pos;

	const TMap& safeMap() const { PresumeFO( map != 0 ); return *map; }
};

//------------------------------------------------------------------------------------------------------------

template<class TMap>
inline bool CConstMapIterator<TMap>::operator==( const CConstMapIterator& other ) const
{
	PresumeFO( map == other.map );
	return pos == other.pos;
}

template<class TMap>
inline bool CConstMapIterator<TMap>::operator==( const CMapIterator<TMap>& other ) const
{
	PresumeFO( map == other.map );
	return pos == other.pos;
}

template<class TMap>
inline CConstMapIterator<TMap>& CConstMapIterator<TMap>::operator++()
{
	pos = safeMap().GetNextPosition( pos );
	return *this;
}

template<class TMap>
inline CConstMapIterator<TMap> CConstMapIterator<TMap>::operator++( int )
{
	const CConstMapIterator old( *this );
	pos = safeMap().GetNextPosition( pos );
	return old;
}

//------------------------------------------------------------------------------------------------------------

template<class TMap>
class CMapIterator {
public:
	CMapIterator() : map( 0 ), pos( NotFound ) {}
	CMapIterator( TMap* _map, TMapPosition _pos ) : map( _map ), pos( _pos ) {}

	bool operator==( const CMapIterator& other ) const;
	bool operator!=( const CMapIterator& other ) const { return !( *this == other ); }
	bool operator==( const CConstMapIterator<TMap>& other ) const;
	bool operator!=( const CConstMapIterator<TMap>& other ) const { return !( *this == other ); }
	CMapIterator& operator++();
	CMapIterator operator++( int );
	typename TMap::TElement& operator*() const { return safeMap().GetKeyValue( pos ); }
	typename TMap::TElement* operator->() const { return &safeMap().GetKeyValue( pos ); }

	friend class CConstMapIterator<TMap>;

private:
	TMap* map;
	TMapPosition pos;

	TMap& safeMap() const { PresumeFO( map != 0 ); return *map; }
};

//------------------------------------------------------------------------------------------------------------

template<class TMap>
inline bool CMapIterator<TMap>::operator==( const CMapIterator& other ) const
{
	PresumeFO( map == other.map );
	return pos == other.pos;
}

template<class TMap>
inline bool CMapIterator<TMap>::operator==( const CConstMapIterator<TMap>& other ) const
{
	PresumeFO( map == other.map );
	return pos == other.pos;
}

template<class TMap>
inline CMapIterator<TMap>& CMapIterator<TMap>::operator++()
{
	pos = safeMap().GetNextPosition( pos );
	return *this;
}

template<class TMap>
inline CMapIterator<TMap> CMapIterator<TMap>::operator++( int )
{
	const CMapIterator old( *this );
	pos = safeMap().GetNextPosition( pos );
	return old;
}

} // namespace FObj
