/* Copyright © 2017-2024 ABBYY

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

#include <HashTableAllocatorFOL.h>
#include <MapIteratorFOL.h>
#include <MathFOL.h>

namespace FObj {

template<class KEY, class VALUE, class HASHER, class ALLOC>
class CMap;

template<class KEY, class VALUE>
struct CMapData;

template<template<class, class, class, class> class TMap, class KEY, class VALUE, class HASHER, class ALLOC>
struct IsMemmoveable<TMap<KEY, VALUE, HASHER, ALLOC>,
	std::enable_if_t<
		std::is_base_of< CMap<KEY, VALUE, HASHER, ALLOC>, TMap<KEY, VALUE, HASHER, ALLOC> >::value &&
		sizeof( CMap<KEY, VALUE, HASHER, ALLOC> ) == sizeof( TMap<KEY, VALUE, HASHER, ALLOC> )>
	>
{
	static constexpr bool Value = true;
};

namespace DetailsCMap {
template<class Arg, class T>
constexpr bool IsRvalueSupported() {
	return std::is_rvalue_reference<Arg&&>::value && !std::is_reference<T>::value &&
		!( std::is_same<Arg, int>::value || std::is_same<Arg, std::nullptr_t>::value );
}
} // namespace DetailsCMap

template<class KEY, class VALUE>
struct CMapData {
	KEY Key;
	VALUE Value;

	CMapData() = default;
	CMapData( CMapData&& ) = default;
	CMapData( const CMapData& ) = default;
	CMapData( const KEY& key, const VALUE& val ) : Key( key ), Value( val ) {}

	template<class Arg, std::enable_if_t< DetailsCMap::IsRvalueSupported<Arg, VALUE>(), int > = 0 >
	CMapData( const KEY& key, Arg&& val ) : Key( key ), Value( std::move( val ) ) {}

	explicit CMapData( const KEY& key ) : Key( key ), Value() {}
	~CMapData() = default;

	auto operator=( const CMapData& ) -> CMapData & = default;
	auto operator=( CMapData&& ) -> CMapData & = default;
};

const int DefMapHashTableSize = 31;
const int MapIndexGroupLength = 4;

//------------------------------------------------------------------------------------------------------------

template<class KEY, class VALUE, class KEYHASHINFO = CDefaultHash<KEY>, class ALLOCATOR = CurrentMemoryManager>
class CMap final {
public:
	typedef KEY KeyType;
	typedef VALUE ValueType;
	typedef ALLOCATOR AllocatorType;
	typedef KEYHASHINFO KeyHashInfoType;
	typedef CMapData<const KEY, VALUE> TElement;
	typedef CConstMapIterator<CMap> TConstIterator;
	typedef CMapIterator<CMap> TIterator;

	CMap();
	explicit CMap( int hashSize );
	template<class MapInitStruct>
	CMap( const MapInitStruct* data, int dataSize );
	CMap( const std::initializer_list<TElement>& list );
	CMap( CMap&& other );
	~CMap();

	CMap( const CMap& ) = delete;
	auto operator=( const CMap& ) -> CMap& = delete;

	auto operator=( const std::initializer_list<TElement>& list ) -> CMap&;
	auto operator=( CMap&& other ) -> CMap&;

	void CopyTo( CMap& ) const;
	void MoveTo( CMap& );
	void AppendTo( CMap& ) const;

	int Size() const;
	void SetHashTableSize( int hashSize );

	bool IsEmpty() const { return Size() == 0; }
	void Empty() { DeleteAll(); }
	void Set( const KEY& key, const VALUE& value);
	template<class Arg, std::enable_if_t< DetailsCMap::IsRvalueSupported<Arg, VALUE>(), int > = 0 >
	void Set( const KEY&, Arg&& );
	void Add( const KEY&, const VALUE& );
	template<class Arg, std::enable_if_t< DetailsCMap::IsRvalueSupported<Arg, VALUE>(), int > = 0 >
	void Add( const KEY&, Arg&& );
	void Add( const std::initializer_list<TElement>& list );
	void Add( const CMap& values );
	void Add( CMap&& values );
	auto CreateValue(const KEY&) -> VALUE&;
	auto CreateOrReplaceValue( const KEY& key ) -> VALUE&;
	auto AddValue( const KEY& ) -> VALUE&;
	auto CreateNewValue( const KEY& key ) -> VALUE&;

	bool Has( const KEY& ) const;
	bool Lookup( const KEY& key, VALUE& resultValue ) const;

	template<class TValue = VALUE>
	auto Lookup( const KEY& key ) -> std::enable_if_t< std::is_same<TValue, VALUE>::value
		&& !std::is_reference<TValue>::value, TValue* >;
	template<class TValue = VALUE>
	auto Lookup( const KEY& key ) const -> std::enable_if_t< std::is_same<TValue, VALUE>::value
		&& !std::is_reference<TValue>::value, const TValue* >;

	template<class ARRAY>
	bool LookupAllValues( const KEY& key, ARRAY& values ) const;
	template<class ARRAY>
	bool LookupAllValues( const KEY& key, ARRAY& values );
	auto Get( const KEY& ) const -> const VALUE&;
	auto Get( const KEY& ) -> VALUE&;
	auto GetOrCreateValue( const KEY& ) -> VALUE&;
	auto GetOrCreateValue( const KEY& key, const VALUE& defaultValue ) -> VALUE&;
	template<class Arg, std::enable_if_t< DetailsCMap::IsRvalueSupported<Arg, VALUE>(), int > = 0 >
	auto GetOrCreateValue( const KEY& key, Arg&& defaultValue ) -> VALUE&;

	auto operator [] ( const KEY& key ) const -> const VALUE& { return Get( key ); }
	auto operator [] ( const KEY& key ) -> VALUE& { return Get( key ); }

	int Delete( const KEY& );
	void DeleteAt( TMapPosition );
	void DeleteAll();
	void FreeBuffer();

	auto GetFirstPosition() const -> TMapPosition;
	auto GetNextPosition( TMapPosition pos ) const -> TMapPosition;

	auto GetFirstPosition( const KEY& ) const -> TMapPosition;
	auto GetNextPosition( const KEY&, TMapPosition pos ) const -> TMapPosition;

	auto GetKey( TMapPosition ) const -> const KEY&;
	auto GetValue( TMapPosition ) const -> const VALUE&;
	auto GetValue( TMapPosition ) -> VALUE&;

	auto GetKeyValue( TMapPosition pos ) const -> const TElement&;
	auto GetKeyValue( TMapPosition pos ) -> TElement&;

	void Serialize( CArchive& ar );

	auto begin() const -> TConstIterator { return TConstIterator( this, GetFirstPosition() ); }
	auto end() const -> TConstIterator { return TConstIterator( this, NotFound ); }
	auto begin() -> TIterator { return TIterator( this, GetFirstPosition() ); }
	auto end() -> TIterator { return TIterator( this, NotFound ); }

	bool operator==( const CMap& values ) const;
	bool operator!=( const CMap& values ) const;

private:
	class CIndexEntry {
	public:
		CIndexEntry() : data( 0 ) { PresumeFO( IsFree() ); }
		explicit CIndexEntry( unsigned int groupStart ) : data( ( groupStart << 1 ) | 1 )
		{ static_assert( sizeof( unsigned int ) == 4, "" ); PresumeFO( ( groupStart >> 31 ) == 0 ); PresumeFO( IsGroupPointer() ); }
		explicit CIndexEntry( CMapData<KEY, VALUE>* dataPointer ) : data( reinterpret_cast<size_t>( dataPointer ) )
		{ PresumeFO( IsDataPointer() ); }

		bool IsFree() const { return data == 0; }
		bool IsDataPointer() const { return !IsFree() && !IsGroupPointer(); }
		bool IsGroupPointer() const { return ( data & 1 ) != 0; }

		CMapData<KEY, VALUE>* DataPointer() const { PresumeFO( IsDataPointer() ); return reinterpret_cast<CMapData<KEY, VALUE>*>( data ); }
		int NextGroupStart() const { PresumeFO( IsGroupPointer() ); return static_cast<int>( data >> 1 ); }

	private:
		size_t data = 0;
	};

	using TIndex = CArray<CIndexEntry, ALLOCATOR>;

	enum {
		AllocatorBlockSize = sizeof( CMapData<KEY, VALUE> ) > MinHashTableAllocatorBlockSize ?
			sizeof( CMapData<KEY, VALUE> ) : MinHashTableAllocatorBlockSize
	};

	TIndex index;
	int valuesCount = 0;
	int hashTableSize = 0;
	const int initialHashTableSize;
	CHashTableAllocator<ALLOCATOR, AllocatorBlockSize> dataAllocator;

	template<class Arg>
	auto getOrCreateValueImpl( const KEY& key, Arg&& arg ) -> VALUE&;

	void growIndex( int minSize );
	void init( int hashSize );
	template<class Arg>
	auto addValue( int hash, const KEY&, Arg&& ) -> VALUE&;
	auto addValue( int hash, const KEY& ) -> VALUE&;
	int deleteAllValues( int hash, const KEY& );
	bool canRehash() const;
	auto findIndexFreePos( int hash, int hashTableSize, TIndex& index ) const -> TMapPosition;
	auto findIndexFreePosWithGrow( int hash ) -> TMapPosition;
	auto findKeyInIndex( const KEY&, TMapPosition from ) const -> TMapPosition;

	auto first( int hash ) const ->	TMapPosition;
	auto first( int hashCode, int hashTableSize, const TIndex& index ) const -> TMapPosition;
	auto next( TMapPosition ) const -> TMapPosition;
	auto next( TMapPosition pos, int hashTableSize, const TIndex& index ) const -> TMapPosition;

	static int getIndexSize( int hashSize );
};

//------------------------------------------------------------------------------------------------------------

extern int UpperPrimeNumber( int number );

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CMap() :
	initialHashTableSize( DefMapHashTableSize )
{
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CMap( int hashSize ) :
	initialHashTableSize( UpperPrimeNumber( hashSize - 1 ) )
{
	PresumeFO( hashSize >= 0 );
	dataAllocator.Reserve( hashSize );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class MapInitStruct>
CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CMap( const MapInitStruct* data, int dataSize ) :
	initialHashTableSize( UpperPrimeNumber( dataSize - 1 ) )
{
	AssertFO( data != 0 );
	AssertFO( dataSize > 0 );

	dataAllocator.Reserve( dataSize );
	init( initialHashTableSize );

	for( int i = 0; i < dataSize; i++ ) {
		const MapInitStruct& entry = *( data + i );
		Add( entry.Key, entry.Value );
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CMap( const std::initializer_list<TElement>& list ) :
	CMap()
{
	Add( list );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CMap( CMap&& other ) :
	CMap()
{
	FObj::swap( *this, other );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::~CMap()
{
	FreeBuffer();
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::operator=( const std::initializer_list<TElement>& list ) -> CMap&
{
	DeleteAll();
	Add( list );
	return *this;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::operator=( CMap&& other ) -> CMap&
{
	FObj::swap( *this, other );
	return *this;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
int CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Size() const
{
	return valuesCount;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::SetHashTableSize( int size )
{
	dataAllocator.Reserve( size );
	if( size > hashTableSize ) {
		growIndex( size );
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::init( int hashSize )
{
	hashTableSize = hashSize;
	index.DeleteAll();
	index.SetSize( hashTableSize );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
int CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::getIndexSize( int hashSize )
{
	return hashSize + CeilTo( hashSize / 2, MapIndexGroupLength );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Set( const KEY& key, const VALUE& value )
{
	const int hash = KEYHASHINFO::HashKey( key );
	deleteAllValues( hash, key );
	addValue( hash, key, value );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class Arg, std::enable_if_t< DetailsCMap::IsRvalueSupported<Arg, VALUE>(), int > >
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Set( const KEY& key, Arg&& value )
{
	const int hash = KEYHASHINFO::HashKey( key );
	deleteAllValues( hash, key );
	addValue( hash, key, std::move( value ) );
}

// DEPRECATED !!! Use CreateOrReplaceValue instead
template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CreateValue( const KEY& key ) -> VALUE&
{
	return CreateOrReplaceValue( key );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CreateOrReplaceValue( const KEY& key ) -> VALUE&
{
	const int hash = KEYHASHINFO::HashKey( key );
	deleteAllValues( hash, key );
	return addValue( hash, key );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CreateNewValue( const KEY& key ) -> VALUE&
{
	const TMapPosition pos = GetFirstPosition( key );
	AssertFO( pos == NotFound );
	const int hash = KEYHASHINFO::HashKey( key );
	return addValue( hash, key );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Add( const KEY& key, const VALUE& value )
{
	const int hash = KEYHASHINFO::HashKey( key );
	addValue( hash, key, value );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class Arg, std::enable_if_t< DetailsCMap::IsRvalueSupported<Arg, VALUE>(), int > >
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Add( const KEY& key, Arg&& value )
{
	const int hash = KEYHASHINFO::HashKey( key );
	addValue( hash, key, std::move( value ) );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Add( const std::initializer_list<TElement>& list )
{
	SetHashTableSize( Size() + static_cast<int>( list.size() ) );
	for( const TElement& element : list ) {
		Add( element.Key, element.Value );
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Add( const CMap& values )
{
	values.AppendTo( *this );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Add( CMap&& values )
{
	SetHashTableSize( Size() + values.Size() );
	for( const TElement& element : values ) {
		Add( element.Key, std::move( element.Value ) );
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::AddValue( const KEY& key ) -> VALUE&
{
	const int hash = KEYHASHINFO::HashKey( key );
	return addValue( hash, key );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
bool CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Lookup( const KEY& key, VALUE& value ) const
{
	const TMapPosition position = GetFirstPosition( key );
	if( position == NotFound ) {
		return false;
	}
	value = GetValue( position );
	return true;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class TValue>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Lookup( const KEY& key )
	-> std::enable_if_t< std::is_same<TValue, VALUE>::value && !std::is_reference<TValue>::value, TValue* >
{
	const TMapPosition position = GetFirstPosition( key );
	if( position == NotFound ) {
		return nullptr;
	}
	return &GetValue( position );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class TValue>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Lookup( const KEY& key ) const
	-> std::enable_if_t< std::is_same<TValue, VALUE>::value && !std::is_reference<TValue>::value, const TValue* >
{
	const TMapPosition position = GetFirstPosition( key );
	if( position == NotFound ) {
		return nullptr;
	}
	return &GetValue( position );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class ARRAY>
bool CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::LookupAllValues( const KEY& key, ARRAY& values ) const
{
	values.DeleteAll();
	const int hash = KEYHASHINFO::HashKey( key );

	for( TMapPosition pos = first( hash ); pos != NotFound; pos = next( pos ) ) {
		if( index[pos].IsDataPointer() && KEYHASHINFO::IsEqual( index[pos].DataPointer()->Key, key ) ) {
			values.Add( GetValue( pos ) );
		}
	}
	return !values.IsEmpty();
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class ARRAY>
bool CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::LookupAllValues( const KEY& key, ARRAY& values )
{
	values.DeleteAll();
	const int hash = KEYHASHINFO::HashKey( key );

	for( TMapPosition pos = first( hash ); pos != NotFound; pos = next( pos ) ) {
		if( index[pos].IsDataPointer() && KEYHASHINFO::IsEqual( index[pos].DataPointer()->Key, key ) ) {
			values.Add( GetValue( pos ) );
		}
	}
	return !values.IsEmpty();
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Get( const KEY& key ) -> VALUE&
{
	TMapPosition position = GetFirstPosition( key );
	AssertFO( position != NotFound );
	return GetValue( position );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Get( const KEY& key ) const -> const VALUE&
{
	TMapPosition position = GetFirstPosition( key );
	AssertFO( position != NotFound );
	return GetValue( position );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetOrCreateValue( const KEY& key ) -> VALUE&
{
	const int hash = KEYHASHINFO::HashKey( key );
	TMapPosition position = findKeyInIndex( key, first( hash ) );
	if( position != NotFound ) {
		return GetValue( position );
	}
	return addValue( hash, key );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetOrCreateValue( const KEY& key, const VALUE& value ) -> VALUE&
{
	return getOrCreateValueImpl( key, value );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class Arg, std::enable_if_t< DetailsCMap::IsRvalueSupported<Arg, VALUE>(), int > >
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetOrCreateValue( const KEY& key, Arg&& value ) -> VALUE&
{
	return getOrCreateValueImpl( key, std::move( value ) );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class Arg>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::getOrCreateValueImpl( const KEY& key, Arg&& arg ) -> VALUE&
{
	const int hash = KEYHASHINFO::HashKey( key );
	const TMapPosition position = findKeyInIndex( key, first( hash ) );
	if( position != NotFound ) {
		return GetValue( position );
	}
	return addValue( hash, key, std::forward<Arg>( arg ) );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class Arg>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::addValue( int hash, const KEY& key, Arg&& arg ) -> VALUE&
{
	const TMapPosition freePos = findIndexFreePosWithGrow( hash );
	CMapData<KEY, VALUE>* ptr = new( dataAllocator.Alloc() ) CMapData<KEY, VALUE>( key, std::forward<Arg>( arg ) );
	valuesCount++;
	index[freePos] = CIndexEntry( ptr );
	return ptr->Value;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::addValue( int hash, const KEY& key ) -> VALUE&
{
	TMapPosition freePos = findIndexFreePosWithGrow( hash );
	CMapData<KEY, VALUE>* ptr = new( dataAllocator.Alloc() ) CMapData<KEY, VALUE>( key );
	valuesCount++;
	index[freePos] = CIndexEntry( ptr );
	return ptr->Value;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
int CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Delete( const KEY& key )
{
	const int hash = KEYHASHINFO::HashKey( key );
	return deleteAllValues( hash, key );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
int CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::deleteAllValues( int hash, const KEY& key )
{
	int count = 0;
	for( TMapPosition pos = first( hash ); pos != NotFound; pos = next( pos ) ) {
		if( index[pos].IsDataPointer() && KEYHASHINFO::IsEqual( index[pos].DataPointer()->Key, key ) ) {
			DeleteAt( pos );
			count++;
		}
	}
	return count;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::DeleteAt( TMapPosition pos )
{
	CMapData<KEY, VALUE>* data = index[pos].DataPointer();
	index[pos] = CIndexEntry();
	valuesCount--;
	data->~CMapData<KEY, VALUE>();
	dataAllocator.Free( data );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::DeleteAll()
{
	if( valuesCount == 0 ) {
		return;
	}
	for( CIndexEntry& entry : index ) {
		if( entry.IsDataPointer() ) {
			entry.DataPointer()->~CMapData<KEY, VALUE>();
			dataAllocator.Free( entry.DataPointer() );
		}
		entry = CIndexEntry();
	}
	index.SetSize( hashTableSize );
	valuesCount = 0;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CopyTo( CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>& dest ) const
{
	if( &dest == this ) {
		return;
	}
	dest.DeleteAll();

	index.CopyTo( dest.index );
	dest.valuesCount = valuesCount;
	dest.hashTableSize = hashTableSize;

	for( CIndexEntry& entry : dest.index ) {
		if( entry.IsDataPointer() ) {
			void* ptr = dest.dataAllocator.Alloc();
			entry = CIndexEntry( new( ptr ) CMapData<KEY, VALUE>( *entry.DataPointer() ) );
		}
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::MoveTo( CMap& dest )
{
	if( &dest == this ) {
		return;
	}
	dest.DeleteAll();

	index.MoveTo( dest.index );
	dest.valuesCount = valuesCount;
	dest.hashTableSize = hashTableSize;

	valuesCount = 0;
	hashTableSize = 0;
	dataAllocator.MoveTo( dest.dataAllocator );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::AppendTo( CMap& dest ) const
{
	if( &dest == this ) {
		return;
	}

	dest.SetHashTableSize( dest.Size() + Size() );

	for( const TElement& data : *this ) {
		dest.Add( data.Key, data.Value );
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::first( int hashCode ) const -> TMapPosition
{
	return first( hashCode, hashTableSize, index );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::first( int hashCode, int tableSize, const TIndex& indexEntry ) const -> TMapPosition
{
	if( indexEntry.Size() == 0 ) {
		return NotFound;
	}

	if( tableSize > 0 ) {
		const int hash = DWORD( hashCode ) % DWORD( tableSize );
		PresumeFO( 0 <= hash && hash < tableSize );
		if( indexEntry[hash].IsGroupPointer() ) {
			return indexEntry[hash].NextGroupStart();
		} else {
			return hash;
		}
	} else {
		AssertFO( false );
		return NotFound;
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::next( TMapPosition pos ) const -> TMapPosition
{
	return next( pos, hashTableSize, index );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::next( TMapPosition pos, int tableSize, const TIndex& indexEntry ) const -> TMapPosition
{
	if( pos < tableSize || ( ( pos - tableSize + 1 ) % MapIndexGroupLength ) == 0 ) {
		return NotFound;
	}

	pos++;
	if( indexEntry[pos].IsGroupPointer() ) {
		return indexEntry[pos].NextGroupStart();
	} else {
		return pos;
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::FreeBuffer()
{
	if( valuesCount != 0 ) {
		for( CIndexEntry& entry : index ) {
			if( entry.IsDataPointer() ) {
				entry.DataPointer()->~CMapData<KEY, VALUE>();
			}
		}
		valuesCount = 0;
	}

	index.FreeBuffer();
	dataAllocator.FreeBuffer();
	hashTableSize = 0;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
bool CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Has( const KEY& key ) const
{
	return GetFirstPosition( key ) != NotFound;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetFirstPosition() const -> TMapPosition
{
	TMapPosition result = 0;
	for( const CIndexEntry& entry : index ) {
		if( entry.IsDataPointer() ) {
			return result;
		}
		result++;
	}
	return NotFound;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetFirstPosition( const KEY& key ) const -> TMapPosition
{
	const int hash = KEYHASHINFO::HashKey( key );
	return findKeyInIndex( key, first( hash ) );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetNextPosition( TMapPosition pos ) const -> TMapPosition
{
	for( TMapPosition i = pos + 1; i < index.Size(); i++ ) {
		if( index[i].IsDataPointer() ) {
			return i;
		}
	}
	return NotFound;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetNextPosition( const KEY& key, TMapPosition prev ) const -> TMapPosition
{
	return findKeyInIndex( key, next( prev ) );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetValue( TMapPosition pos ) const -> const VALUE&
{
	return index[pos].DataPointer()->Value;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetValue( TMapPosition pos ) -> VALUE&
{
	return index[pos].DataPointer()->Value;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetKey( TMapPosition pos ) const -> const KEY&
{
	return index[pos].DataPointer()->Key;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetKeyValue( TMapPosition pos ) const -> const TElement&
{
	return *reinterpret_cast<TElement*>( index[pos].DataPointer() );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetKeyValue( TMapPosition pos ) -> TElement&
{
	return *reinterpret_cast<TElement*>( index[pos].DataPointer() );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
bool CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::operator==( const CMap& values ) const
{
	if( &values == this ) {
		return true;
	}

	if( Size() != values.Size() ) {
		return false;
	}

	for( const TElement& data : *this ) {
		if( !values.Has( data.Key ) ) {
			return false;
		}
		if( values.Get( data.Key ) != data.Value ) {
			return false;
		}
	}
	return true;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
bool CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::operator!=( const CMap& values ) const
{
	return !( *this == values );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::findKeyInIndex( const KEY& key, TMapPosition from ) const -> TMapPosition
{
	for( TMapPosition pos = from; pos != NotFound; pos = next( pos ) ) {
		if( index[pos].IsDataPointer() && KEYHASHINFO::IsEqual( index[pos].DataPointer()->Key, key ) ) {
			return pos;
		}
	}
	return NotFound;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::findIndexFreePos( int hash, int tableSize, TIndex& indexEntry ) const -> TMapPosition
{
	TMapPosition lastDataPos = NotFound;
	TMapPosition pos = first( hash, tableSize, indexEntry );
	for( ; pos != NotFound; pos = next( pos, tableSize, indexEntry ) ) {
		if( indexEntry[pos].IsFree() ) {
			return pos;
		} else if( indexEntry[pos].IsDataPointer() ) {
			lastDataPos = pos;
		} else {
			AssertFO( false );
		}
	}

	PresumeFO( lastDataPos != NotFound );
	PresumeFO( pos < tableSize || ( ( pos - tableSize + 1 ) % MapIndexGroupLength ) == 0 );

	if( indexEntry.Size() + MapIndexGroupLength > getIndexSize( tableSize ) ) {
		return NotFound;
	}
	int groupPos = indexEntry.Size();
	indexEntry.SetSize( groupPos + MapIndexGroupLength );

	indexEntry[groupPos] = indexEntry[lastDataPos];
	indexEntry[lastDataPos] = CIndexEntry( groupPos );
	return groupPos + 1;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
bool CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::canRehash() const
{
	return ( valuesCount + 1 < hashTableSize / 4 );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
auto CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::findIndexFreePosWithGrow( int hash ) -> TMapPosition
{
	if( index.Size() == 0 ) {
		init( initialHashTableSize );
	}

	TMapPosition freePos = findIndexFreePos( hash, hashTableSize, index );
	if( freePos == NotFound && canRehash() ) {
		growIndex( hashTableSize );
		freePos = findIndexFreePos( hash, hashTableSize, index );
	}
	while( freePos == NotFound ) {
		growIndex( UpperPrimeNumber( hashTableSize ) );
		freePos = findIndexFreePos( hash, hashTableSize, index );
	}
	return freePos;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::growIndex( int minSize )
{
	TIndex newIndex;
	int newHashTableSize = minSize - 1;

	int i = 0;
	do {
		newHashTableSize = UpperPrimeNumber( newHashTableSize );
		newIndex.DeleteAll();
		newIndex.SetSize( newHashTableSize );

		for( i = 0; i < index.Size(); i++ ) {
			if( !index[i].IsDataPointer() ) {
				continue;
			}
			CMapData<KEY, VALUE>* data = index[i].DataPointer();
			const int hash = KEYHASHINFO::HashKey( data->Key );
			TMapPosition freePos = findIndexFreePos( hash, newHashTableSize, newIndex );
			if( freePos == NotFound ) {
				break;
			}
			newIndex[freePos] = CIndexEntry( data );
		}
	} while( i < index.Size() );

	newIndex.MoveTo( index );
	hashTableSize = newHashTableSize;
}

//---------------------------------------------------------------------------------------------------------------------

// Serialize

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Serialize( CArchive& ar )
{
	if( ar.IsStoring() ) {
		int count = Size();
		ar << count;
		for( TMapPosition pos = GetFirstPosition(); pos != NotFound; pos = GetNextPosition( pos ) ) {
			ar << GetKey( pos ) << GetValue( pos );
			count--;
		}
	} else {
		DeleteAll();
		int count;
		ar >> count;
		if( count > 0 ) {
			init( UpperPrimeNumber( count - 1 ) );
		}
		for( int i = 0; i < count; i++ ) {
			KEY key;
			ar >> key;
			ar >> AddValue( key );
		}
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
CArchive& operator>>( CArchive& archive, CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>& map )
{
	map.Serialize( archive );
	return archive;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
CArchive& operator<<( CArchive& archive, const CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>& map )
{
	const_cast< CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>& >( map ).Serialize( archive );
	return archive;
}

} // namespace FObj
