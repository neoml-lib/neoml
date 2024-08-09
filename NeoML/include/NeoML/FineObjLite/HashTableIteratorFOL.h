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

// The identifier of the position in the CHashTable when iterating.
typedef int THashTablePosition;

template<class THashTable>
class CHashTableIterator final {
public:
	CHashTableIterator() : table( 0 ), pos( NotFound ) {}
	CHashTableIterator( const THashTable* _table, THashTablePosition _pos ) : table( _table ), pos( _pos ) {}

	bool operator==( const CHashTableIterator& other ) const;
	bool operator!=( const CHashTableIterator& other ) const { return !( *this == other ); }

	CHashTableIterator& operator++();
	CHashTableIterator operator++( int );

	const typename THashTable::TElement& operator*() const { return safeTable().GetValue( pos ); }
	const typename THashTable::TElement* operator->() const { return &safeTable().GetValue( pos ); }

private:
	const THashTable* table;
	THashTablePosition pos;

	const THashTable& safeTable() const { PresumeFO( table != 0 ); return *table; }
};

//---------------------------------------------------------------------------------------------------------------------

template<class THashTable>
inline bool CHashTableIterator<THashTable>::operator==( const CHashTableIterator& other ) const
{
	PresumeFO( table == other.table );
	return pos == other.pos;
}

template<class THashTable>
inline CHashTableIterator<THashTable>& CHashTableIterator<THashTable>::operator++()
{
	pos = safeTable().GetNextPosition( pos );
	return *this;
}

template<class THashTable>
inline CHashTableIterator<THashTable> CHashTableIterator<THashTable>::operator++( int )
{
	const CHashTableIterator old( *this );
	pos = safeTable().GetNextPosition( pos );
	return old;
}

} // namespace FObj
