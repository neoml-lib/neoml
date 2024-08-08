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

#include <ArrayFOL.h>
#include <HashTableAllocatorFOL.h>
#include <HashTableIteratorFOL.h>
#include <MathFOL.h>
#include <StringFOL.h>

namespace FObj {

const int DefHashTableSize = 31;
const int HashIndexGroupLength = 4;

constexpr int CombineHash( int first, int second ) noexcept
{
	return ( first << 5 ) + first + second;
}

constexpr void AddToHashKey( int valueToAdd, int& result ) noexcept
{
	result = CombineHash( result, valueToAdd );
}

constexpr int GetMBCStringHash( const char* string ) noexcept
{
	if( string == nullptr ) { // invalid pointer
		return 0;
	}
	if( *string == 0 ) { // is empty string
		return 0;
	}
	int result = *string;
	string++;
	while( *string ) {
		AddToHashKey( *string, result );
		string++;
	}
	return result;
}

//---------------------------------------------------------------------------------------------------------------------

template< typename KEY, typename Enable = void >
struct CDefaultHash final {
	static int HashKey( const KEY& key ) { return key.HashKey(); }
	static bool IsEqual( const KEY& first, const KEY& second ) { return (first == second); }
};

template< class KEY >
struct CDefaultHash<KEY*> final {
	static int HashKey( KEY* const& key ) { return static_cast<int>( reinterpret_cast<size_t>( key ) ); }
	static bool IsEqual( KEY* const& first, KEY* const& second ) { return first == second; }
};

//---------------------------------------------------------------------------------------------------------------------

template<> 
inline int CDefaultHash<char>::HashKey( const char& key ) { return static_cast<int>( key ); }

template<>
inline int CDefaultHash<unsigned char>::HashKey( const unsigned char& key ) { return static_cast<int>( key ); }

template<>
inline int CDefaultHash<wchar_t>::HashKey( const wchar_t& key )
{
	static_assert( sizeof( key ) <= sizeof( int ), "" );
	return static_cast<int>( key );
}

template<>
inline int CDefaultHash<char32_t>::HashKey( const char32_t& key )
{
	static_assert( sizeof( key ) <= sizeof( int ), "" );
	return static_cast<int>(key);
}

template<>
inline int CDefaultHash<short>::HashKey( const short& key ) { return static_cast<int>( key ); }

template<>
inline int CDefaultHash<unsigned short>::HashKey( const unsigned short& key ) { return static_cast<int>( key ); }

template<>
inline int CDefaultHash<int>::HashKey( const int& key ) { return key; }

template<>
inline int CDefaultHash<unsigned int>::HashKey( const unsigned int& key ) { return static_cast<int>( key ); }

template<>
inline int CDefaultHash<double>::HashKey( const double& key )
{
	//static_assert( sizeof( key ) == sizeof( unsigned __int64 ), "" );
	if( key == 0.0 ) { // 0.0 == -0.0
		return 0;
	}
	return CDefaultHash<unsigned int>::HashKey( *reinterpret_cast<const unsigned int*>( &key ) );
}

template<>
inline int CDefaultHash<float>::HashKey( const float& key )
{
	static_assert( sizeof( key ) == sizeof( unsigned int ), "" );

	if( key == 0.0f ) { // 0.0f == -0.0f
		return 0;
	}
	return CDefaultHash<unsigned int>::HashKey( *reinterpret_cast<const unsigned int*>( &key ) );
}

template<>
inline int CDefaultHash<bool>::HashKey( const bool& key ) { return CDefaultHash<int>::HashKey( key ); }

template <>
inline int CDefaultHash<CString>::HashKey( const CString& str )
{
	const char* ptr = str.Ptr();
	if( *ptr == 0 ) {
		return 0;
	}
	int result = *ptr;
	ptr++;
	while( *ptr ) {
		result = ( result << 5 ) + result + *ptr;
		ptr++;
	}
	return result;
}

//---------------------------------------------------------------------------------------------------------------------

template<>
struct CDefaultHash<const char*> final {
	static int HashKey( const char* string ) { return GetMBCStringHash( string ); }
	static bool IsEqual( const char* first, const char* second ) { return ::strcmp( first, second ) == 0; }
};

//---------------------------------------------------------------------------------------------------------------------

template<typename TKey>
int GetDefaultHash( const TKey& key ) { return CDefaultHash<TKey>::HashKey( key ); }

template<class T, class HashInfo, class Alloc>
class CHashTable;

template<template<class T, class HashInfo, class Alloc> class THashTable, class T, class HashInfo, class Alloc>
struct IsMemmoveable<THashTable<T, HashInfo, Alloc>,
	std::enable_if_t<
		std::is_base_of<CHashTable<T, HashInfo, Alloc>, THashTable<T, HashInfo, Alloc>>::value &&
		std::is_same<typename THashTable<T, HashInfo, Alloc>::TElement, T>::value &&
		sizeof( CHashTable<T, HashInfo, Alloc> ) == sizeof( THashTable<T, HashInfo, Alloc> ) > >
{
	static constexpr bool Value = true;
};

//---------------------------------------------------------------------------------------------------------------------

template<class T, class HASHINFO = CDefaultHash<T>, class ALLOCATOR = CurrentMemoryManager>
class CHashTable final {
public:
	typedef T TElement;
	typedef ALLOCATOR AllocatorType;
	typedef HASHINFO HashInfoType;
	typedef CHashTableIterator<CHashTable> TConstIterator;
	typedef TConstIterator TIterator;

	CHashTable();
	explicit CHashTable( int hashSize );
	CHashTable( const T* data, int dataSize );
	CHashTable( CHashTable&& );
	CHashTable( const std::initializer_list<T>& list );
	CHashTable( const CHashTable& ) = delete;

	~CHashTable();

	auto operator=( CHashTable&& ) -> CHashTable&;
	auto operator=( const std::initializer_list<T>& list ) -> CHashTable&;
	auto operator=( const CHashTable& ) -> CHashTable& = delete;

	void CopyTo( CHashTable& ) const;
	void AppendTo( CHashTable& ) const;
	void MoveTo( CHashTable& );

	int Size() const;
	void SetHashTableSize( int size );
	void SetBufferSize( int ) {}
	void Grow( int ) {}

	bool IsEmpty() const { return Size() == 0; }
	void Empty() { DeleteAll(); }
	void Set( const T& value );
	void Set( T&& value );
	void Add( const T& value ) { Set( value ); }
	void Add( T&& value ) { Set( std::move( value ) ); }
	void Add( const std::initializer_list<T>& list );
	template<class TARRAY, std::enable_if_t< std::is_lvalue_reference<TARRAY>::value, int > = 0 >
	void AddArray( TARRAY&& values );
	template<class TARRAY, std::enable_if_t< std::is_rvalue_reference<TARRAY&&>::value, int > = 0 >
	void AddArray( TARRAY&& values );
	void Add( const CHashTable& values );
	void Add( CHashTable&& values );

	bool Has( const T& ) const;

	bool Delete( const T& );
	void DeleteAt( THashTablePosition );
	void DeleteAll();
	void FreeBuffer();

	auto GetFirstPosition() const -> THashTablePosition;
	auto GetNextPosition( THashTablePosition pos ) const -> THashTablePosition;

	auto GetPosition( const T& ) const -> THashTablePosition;

	auto GetValue( THashTablePosition ) const -> const T&;
	auto operator [] ( THashTablePosition pos ) const -> const T& { return GetValue( pos ); }
	auto GetOrCreateValue( const T& ) -> const T&;
	auto GetOrCreateValue( T&& ) -> const T&;

	void Serialize( CArchive& ar );

	auto begin() const -> TConstIterator { return TConstIterator( this, GetFirstPosition() ); }
	auto end() const -> TConstIterator { return TConstIterator( this, NotFound ); }

	bool operator == ( const CHashTable& ) const;
	bool operator != ( const CHashTable& ) const;

private:
	class CIndexEntry {
	public:
		CIndexEntry() : data( 0 ) { PresumeFO( IsFree() ); }
		explicit CIndexEntry( unsigned int groupStart ) : data( ( groupStart << 1 ) | 1 )
		{ static_assert( sizeof( unsigned int ) == 4, "" ); PresumeFO( ( groupStart >> 31 ) == 0 ); PresumeFO( IsGroupPointer() ); }
		explicit CIndexEntry( T* dataPointer ) : data( reinterpret_cast<size_t>( dataPointer ) )
		{ PresumeFO( IsDataPointer() ); }

		bool IsFree() const { return data == 0; }
		bool IsDataPointer() const { return !IsFree() && !IsGroupPointer(); }
		bool IsGroupPointer() const { return ( data & 1 ) != 0; }

		T* DataPointer() const { PresumeFO( IsDataPointer() ); return reinterpret_cast<T*>( data ); }
		int NextGroupStart() const { PresumeFO( IsGroupPointer() ); return static_cast<int>( data >> 1 ); }

	private:
		size_t data = 0;
	};

	enum { AllocatorBlockSize = sizeof( T ) > MinHashTableAllocatorBlockSize
		? sizeof( T )
		: MinHashTableAllocatorBlockSize };

	CArray<CIndexEntry, ALLOCATOR> index;
	int valuesCount = 0;
	int hashTableSize = 0;
	const int initialHashTableSize;
	CHashTableAllocator<ALLOCATOR, AllocatorBlockSize> dataAllocator;

	void growIndex( int minSize );
	void init( int hashSize );
	template<class Arg>
	auto addValue( int hash, Arg&& value ) -> T*;
	auto findIndexFreePos( int hash, int hashTableSize, CArray<CIndexEntry, ALLOCATOR>& index ) const -> THashTablePosition;
	bool canRehash() const;
	auto findValueInIndex( const T&, THashTablePosition from ) const -> THashTablePosition;

	auto first( int hash, int hashTableSize, const CArray<CIndexEntry, ALLOCATOR>& index ) const -> THashTablePosition;
	auto first( int hash ) const -> THashTablePosition;
	auto next( THashTablePosition pos, int hashTableSize,
		const CArray<CIndexEntry, ALLOCATOR>& index ) const -> THashTablePosition;
	auto next( THashTablePosition ) const -> THashTablePosition;

	static int getIndexSize( int hashSize );
	template<class Arg>
	static auto constructAt( void* where, Arg&& value ) -> T*;

	template<class Arg>
	void setImpl( Arg&& arg );

	template<class Arg>
	auto getOrCreateValueImpl( Arg&& arg ) -> const T&;
};

//---------------------------------------------------------------------------------------------------------------------

// CHashTable methods

int UpperPrimeNumber( int number );

template<class T, class HASHINFO, class ALLOCATOR>
CHashTable<T, HASHINFO, ALLOCATOR>::CHashTable() :
	initialHashTableSize( DefHashTableSize )
{}

template<class T, class HASHINFO, class ALLOCATOR>
CHashTable<T, HASHINFO, ALLOCATOR>::CHashTable( int hashSize ) :
	initialHashTableSize( UpperPrimeNumber( hashSize - 1 ) )
{
	PresumeFO( hashSize >= 0 );
	dataAllocator.Reserve( hashSize );
}

template<class T, class HASHINFO, class ALLOCATOR>
CHashTable<T, HASHINFO, ALLOCATOR>::CHashTable( const T* data, int dataSize ) :
	initialHashTableSize( UpperPrimeNumber( dataSize - 1 ) )
{
	AssertFO( data != 0 );
	AssertFO( dataSize > 0 );

	dataAllocator.Reserve( dataSize );
	init( initialHashTableSize );

	for( int i = 0; i < dataSize; i++ ) {
		Add( *( data + i ) );
	}
}

template<class T, class HASHINFO, class ALLOCATOR>
CHashTable<T, HASHINFO, ALLOCATOR>::CHashTable( const std::initializer_list<T>& list ) :
	CHashTable()
{
	Add( list );
}

template<class T, class HASHINFO, class ALLOCATOR>
CHashTable<T, HASHINFO, ALLOCATOR>::CHashTable( CHashTable&& other ) :
	CHashTable()
{
	FObj::swap( *this, other );
}

template<class T, class HASHINFO, class ALLOCATOR>
CHashTable<T, HASHINFO, ALLOCATOR>::~CHashTable()
{
	FreeBuffer();
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::operator=( const std::initializer_list<T>& list ) -> CHashTable&
{
	DeleteAll();
	Add( list );
	return *this;
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::operator=( CHashTable&& other ) -> CHashTable&
{
	FObj::swap( *this, other );
	return *this;
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::CopyTo( CHashTable& dest ) const
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
			entry = CIndexEntry( constructAt( ptr, *entry.DataPointer() ) );
		}
	}
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::AppendTo( CHashTable& dest ) const
{
	if( &dest == this ) {
		return;
	}
	dest.SetHashTableSize( dest.Size() + Size() );

	for( const CIndexEntry& entry : index ) {
		if( !entry.IsDataPointer() ) {
			continue;
		}
		const T* dataPtr = entry.DataPointer();
		dest.Add( *dataPtr );
	}
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::MoveTo( CHashTable& dest )
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

template<class T, class HASHINFO, class ALLOCATOR>
int CHashTable<T, HASHINFO, ALLOCATOR>::Size() const
{
	return valuesCount;
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::SetHashTableSize( int size )
{
	dataAllocator.Reserve( size );
	if( size > hashTableSize ) {
		growIndex( size );
	}
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::init( int hashSize )
{
	hashTableSize = hashSize;
	index.DeleteAll();
	index.SetSize( hashTableSize );
}

template<class T, class HASHINFO, class ALLOCATOR>
int CHashTable<T, HASHINFO, ALLOCATOR>::getIndexSize( int hashSize )
{
	return hashSize + CeilTo( hashSize / 2, HashIndexGroupLength );
}

template<class T, class HASHINFO, class ALLOCATOR>
template<class Arg>
auto CHashTable<T, HASHINFO, ALLOCATOR>::constructAt( void* where, Arg&& value ) -> T*
{
	PresumeFO( where != AddressOfObject<T>( value ) );
	return ::new( where ) T( std::forward<Arg>( value ) );
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::Set( const T& value )
{
	setImpl( value );
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::Set( T&& value )
{
	setImpl( std::move( value ) );
}

template<class T, class HASHINFO, class ALLOCATOR>
template<class Arg>
void CHashTable<T, HASHINFO, ALLOCATOR>::setImpl( Arg&& value )
{
	if( index.Size() == 0 ) {
		init( initialHashTableSize );
	}

	int hash = HASHINFO::HashKey( value );
	int position = findValueInIndex( value, first( hash ) );
	if( position == NotFound ) {
		addValue( hash, std::forward<Arg>( value ) );
	} else {
		T* ptr = index[position].DataPointer();
		PresumeFO( ptr != AddressOfObject( value ) );
		ptr->~T();
		constructAt( ptr, std::forward<Arg>( value ) );
	}
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::Add( const std::initializer_list<T>& list )
{
	SetHashTableSize( Size() + to<int>( list.size() ) );
	for( const T& value : list ) {
		Add( value );
	}
}

template<class T, class HASHINFO, class ALLOCATOR>
template<class TARRAY, std::enable_if_t< std::is_lvalue_reference<TARRAY>::value, int > >
void CHashTable<T, HASHINFO, ALLOCATOR>::AddArray( TARRAY&& values )
{
	for( const T& value : values ) {
		Add( value );
	}
}

template<class T, class HASHINFO, class ALLOCATOR>
template<class TARRAY, std::enable_if_t< std::is_rvalue_reference<TARRAY&&>::value, int > >
void CHashTable<T, HASHINFO, ALLOCATOR>::AddArray( TARRAY&& values )
{
	for( T& value : values ) {
		Add( std::move( value ) );
	}
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::Add( const CHashTable& values )
{
	values.AppendTo( *this );
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::Add( CHashTable&& values )
{
	values.AppendTo( *this );
}

template<class T, class HASHINFO, class ALLOCATOR>
bool CHashTable<T, HASHINFO, ALLOCATOR>::Delete( const T& value )
{
	THashTablePosition pos = GetPosition( value );
	if( pos != NotFound ) {
		DeleteAt( pos );
		return true;
	} else {
		return false;
	}
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::DeleteAt( THashTablePosition pos )
{
	T* data = index[pos].DataPointer();
	index[pos] = CIndexEntry();
	valuesCount--;
	data->~T();
	dataAllocator.Free( data );
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::DeleteAll()
{
	if( valuesCount == 0 ) {
		return;
	}
	for( CIndexEntry& entry : index ) {
		if( entry.IsDataPointer() ) {
			entry.DataPointer()->~T();
			dataAllocator.Free( entry.DataPointer() );
		}
		entry = CIndexEntry();
	}
	index.SetSize( hashTableSize );
	valuesCount = 0;
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::first( int hashCode, int tableSize,
	const CArray<CIndexEntry, ALLOCATOR>& indexEntry ) const -> THashTablePosition
{
	if( indexEntry.Size() == 0 ) {
		return NotFound;
	}

	AssertFO( tableSize > 0 );
	int hash = static_cast<DWORD>( hashCode ) % static_cast<DWORD>( tableSize );
	PresumeFO( 0 <= hash && hash < tableSize );
	if( indexEntry[hash].IsGroupPointer() ) {
		return indexEntry[hash].NextGroupStart();
	} else {
		return hash;
	}
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::first( int hashCode ) const -> THashTablePosition
{
	return first( hashCode, hashTableSize, index );
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::next( THashTablePosition pos, int tableSize,
	const CArray<CIndexEntry, ALLOCATOR>& indexEntry ) const -> THashTablePosition
{
	if( pos < tableSize || ( ( pos - tableSize + 1 ) % HashIndexGroupLength ) == 0 ) {
		return NotFound;
	}

	pos++;
	if( indexEntry[pos].IsGroupPointer() ) {
		return indexEntry[pos].NextGroupStart();
	} else {
		return pos;
	}
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::next( THashTablePosition pos ) const -> THashTablePosition
{
	return next( pos, hashTableSize, index );
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::FreeBuffer()
{
	if( valuesCount != 0 ) {
		for( CIndexEntry& entry : index ) {
			if( entry.IsDataPointer() ) {
				entry.DataPointer()->~T();
			}
		}
		valuesCount = 0;
	}

	index.FreeBuffer();
	dataAllocator.FreeBuffer();
	hashTableSize = 0;
}

template<class T, class HASHINFO, class ALLOCATOR>
bool CHashTable<T, HASHINFO, ALLOCATOR>::Has( const T& value ) const
{
	return GetPosition( value ) != NotFound;
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::GetFirstPosition() const -> THashTablePosition
{
	THashTablePosition result = 0;
	for( const CIndexEntry& entry : index ) {
		if( entry.IsDataPointer() ) {
			return result;
		}
		result++;
	}
	return NotFound;
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::GetPosition( const T& value ) const -> THashTablePosition
{
	int hash = HASHINFO::HashKey( value );
	return findValueInIndex( value, first( hash ) );
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::GetNextPosition( THashTablePosition pos ) const -> THashTablePosition
{
	for( THashTablePosition i = pos + 1; i < index.Size(); i++ ) {
		if( index[i].IsDataPointer() ) {
			return i;
		}
	}
	return NotFound;
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::GetValue( THashTablePosition pos ) const -> const T&
{
	return *index[pos].DataPointer();
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::GetOrCreateValue( const T& value ) -> const T&
{
	return getOrCreateValueImpl( value );
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::GetOrCreateValue( T&& value ) -> const T&
{
	return getOrCreateValueImpl( std::move( value ) );
}

template<class T, class HASHINFO, class ALLOCATOR>
template<class Arg>
auto CHashTable<T, HASHINFO, ALLOCATOR>::getOrCreateValueImpl( Arg&& value ) -> const T&
{
	if( index.Size() == 0 ) {
		init( initialHashTableSize );
	}

	int hash = HASHINFO::HashKey( value );
	int position = findValueInIndex( value, first( hash ) );
	if( position == NotFound ) {
		return *addValue( hash, std::forward<Arg>( value ) );
	}
	return *index[position].DataPointer();
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::findValueInIndex( const T& value, THashTablePosition from ) const -> THashTablePosition
{
	for( THashTablePosition pos = from; pos != NotFound; pos = next( pos ) ) {
		if( index[pos].IsDataPointer() && HASHINFO::IsEqual( *index[pos].DataPointer(), value ) ) {
			return pos;
		}
	}
	return NotFound;
}

template<class T, class HASHINFO, class ALLOCATOR>
template<class Arg>
auto CHashTable<T, HASHINFO, ALLOCATOR>::addValue( int hash, Arg&& value ) -> T*
{
	THashTablePosition freePos = findIndexFreePos( hash, hashTableSize, index );
	if( freePos == NotFound && canRehash() ) {
		growIndex( hashTableSize );
		freePos = findIndexFreePos( hash, hashTableSize, index );
	}
	while( freePos == NotFound ) {
		growIndex( UpperPrimeNumber( hashTableSize ) );
		freePos = findIndexFreePos( hash, hashTableSize, index );
	}

	T* ptr = constructAt( dataAllocator.Alloc(), std::forward<Arg>( value ) );
	index[freePos] = CIndexEntry( ptr );
	valuesCount++;
	return ptr;
}

template<class T, class HASHINFO, class ALLOCATOR>
auto CHashTable<T, HASHINFO, ALLOCATOR>::findIndexFreePos( int hash, int tableSize,
	CArray<CIndexEntry, ALLOCATOR>& indexEntry ) const -> THashTablePosition
{
	THashTablePosition lastDataPos = NotFound;
	THashTablePosition pos = first( hash, tableSize, indexEntry );
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
	PresumeFO( pos < tableSize || ( ( pos - tableSize + 1 ) % HashIndexGroupLength ) == 0 );

	if( indexEntry.Size() + HashIndexGroupLength > getIndexSize( tableSize ) ) {
		return NotFound;
	}
	THashTablePosition groupPos = indexEntry.Size();
	indexEntry.SetSize( groupPos + HashIndexGroupLength );

	indexEntry[groupPos] = indexEntry[lastDataPos];
	indexEntry[lastDataPos] = CIndexEntry( groupPos );
	return groupPos + 1;
}

template<class T, class HASHINFO, class ALLOCATOR>
bool CHashTable<T, HASHINFO, ALLOCATOR>::canRehash() const
{
	return ( valuesCount + 1 < hashTableSize / 4 );
}

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::growIndex( int minSize )
{
	CArray<CIndexEntry, ALLOCATOR> newIndex;
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
			T* data = index[i].DataPointer();
			int hash = HASHINFO::HashKey( *data );

			THashTablePosition freePos = findIndexFreePos( hash, newHashTableSize, newIndex );
			if( freePos == NotFound ) {
				break;
			}
			newIndex[freePos] = CIndexEntry( data );
		}
	} while( i < index.Size() );

	newIndex.MoveTo( index );
	hashTableSize = newHashTableSize;
}

template<class T, class HASHINFO, class ALLOCATOR>
bool CHashTable<T, HASHINFO, ALLOCATOR>::operator==( const CHashTable& values ) const
{
	if( &values == this ) {
		return true;
	}
	if( values.Size() != Size() ) {
		return false;
	}
	for( const T& value : *this ) {
		if( !values.Has( value ) ) {
			return false;
		}
	}
	return true;
}

template<class T, class HASHINFO, class ALLOCATOR>
bool CHashTable<T, HASHINFO, ALLOCATOR>::operator!=( const CHashTable& values ) const
{
	return !( *this == values );
}

//---------------------------------------------------------------------------------------------------------------------

// Serialize

template<class T, class HASHINFO, class ALLOCATOR>
void CHashTable<T, HASHINFO, ALLOCATOR>::Serialize( CArchive& ar )
{
	if( ar.IsStoring() ) {
		int count = Size();
		ar << count;
		for( THashTablePosition pos = GetFirstPosition(); pos != NotFound; pos = GetNextPosition( pos ) ) {
			ar << GetValue( pos );
			count--;
		}
	} else {
		DeleteAll();
		int count;
		ar >> count;
		init( UpperPrimeNumber( count - 1 ) );
		for( int i = 0; i < count; i++ ) {
			T value;
			ar >> value;
			Set( value );
		}
	}
}

template<class T, class HASHINFO, class ALLOCATOR>
CArchive& operator>>( CArchive& archive, CHashTable<T, HASHINFO, ALLOCATOR>& hashTable )
{
	hashTable.Serialize( archive );
	return archive;
}

template<class T, class HASHINFO, class ALLOCATOR>
CArchive& operator<<( CArchive& archive, const CHashTable<T, HASHINFO, ALLOCATOR>& hashTable )
{
	const_cast< CHashTable<T, HASHINFO, ALLOCATOR>& >( hashTable ).Serialize( archive );
	return archive;
}

} // namespace FObj
