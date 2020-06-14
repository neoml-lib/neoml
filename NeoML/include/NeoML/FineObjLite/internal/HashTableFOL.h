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

#include "../StringFOL.h"
#include "../ArrayFOL.h"
#include "HashTableAllocatorFOL.h"

namespace FObj {

typedef int THashTablePosition;

const int DefHashTableSize = 31;
const int HashIndexGroupLength = 4;

inline void AddToHashKey( int valueToAdd, int& result )
{
	result = ( result << 5 ) + result + valueToAdd;
}

inline int GetMBCStringHash( const char* string )
{
	if( *string == 0 ) {
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

//------------------------------------------------------------------------------------------------------------

template< typename KEY > 
struct CDefaultHash {
	static int HashKey( const KEY& key )
	{ 
		return key.HashKey();
	}

	static bool IsEqual( const KEY& first, const KEY& second )
	{
		return (first == second);
	}
};

template< class KEY >
struct CDefaultHash<KEY*> {
	static int HashKey( KEY* const& key )
	{
		return static_cast<int>( reinterpret_cast<size_t>( key ) );
	}
	static bool IsEqual( KEY* const& first, KEY* const& second )
	{
		return first == second;
	}
};

template<> 
inline int CDefaultHash<char>::HashKey( const char& key )
{
	return static_cast<int>( key );
}

template<> 
inline int CDefaultHash<unsigned char>::HashKey( const unsigned char& key )
{
	return static_cast<int>( key );
}

template<> 
inline int CDefaultHash<wchar_t>::HashKey( const wchar_t& key )
{
	return static_cast<int>( key );
}

template<> 
inline int CDefaultHash<short>::HashKey( const short& key )
{
	return static_cast<int>( key );
}

template<> 
inline int CDefaultHash<unsigned short>::HashKey( const unsigned short& key )
{
	return static_cast<int>( key );
}

template<> 
inline int CDefaultHash<int>::HashKey( const int& key )
{
	return key;
}

template<> 
inline int CDefaultHash<unsigned int>::HashKey( const unsigned int& key )
{
	return static_cast<int>( key );
}

template<>
inline int CDefaultHash<double>::HashKey( const double& key )
{
	if( key == 0.0 ) { // 0.0 == -0.0
		return 0;
	}
	return CDefaultHash<unsigned int>::HashKey( *reinterpret_cast<const unsigned int*>( &key ) );
}

template<>
inline int CDefaultHash<float>::HashKey( const float& key )
{
	if( key == 0.0f ) { // 0.0f == -0.0f
		return 0;
	}
	return CDefaultHash<unsigned int>::HashKey( *reinterpret_cast<const unsigned int*>( &key ) );
}

template <>
inline int CDefaultHash<CString>::HashKey( const CString& str )
{
	const char* ptr = str.data();
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

//------------------------------------------------------------------------------------------------------------

template<class VALUE, class HASHINFO = CDefaultHash<VALUE>, class ALLOCATOR = CurrentMemoryManager>
class CHashTable {
public:
	typedef VALUE TElement;
	typedef ALLOCATOR AllocatorType;
	typedef HASHINFO HashInfoType;

	CHashTable();
	explicit CHashTable( int hashSize );
	CHashTable( const VALUE* data, int dataSize );
	~CHashTable();

	void CopyTo( CHashTable& ) const;
	void MoveTo( CHashTable& );
	
	int Size() const;
	void SetHashTableSize( int size );

	bool IsEmpty() const { return Size() == 0; }
	void Empty() { DeleteAll(); }
	void Set( const VALUE& value );
	void Add( const VALUE& value ) { Set( value ); }
	template<class TARRAY> void AddArray( const TARRAY& values );

	bool Has( const VALUE& ) const;

	void Delete( const VALUE& );
	void DeleteAt( THashTablePosition );
	void DeleteAll();
	void FreeBuffer();

	THashTablePosition GetFirstPosition() const;
	THashTablePosition GetNextPosition( THashTablePosition pos ) const;

	THashTablePosition GetPosition( const VALUE& ) const;

	const VALUE& GetValue( THashTablePosition ) const;
	const VALUE& operator [] ( THashTablePosition pos ) const { return GetValue( pos ); }
	const VALUE& GetOrCreateValue( const VALUE& );

private:
	class CIndexEntry {
	public:
		CIndexEntry() : Data( 0 ) { PresumeFO( IsFree() ); }
		explicit CIndexEntry( unsigned int groupStart ) : Data( ( groupStart << 1 ) | 1 )
			{ static_assert( sizeof(unsigned int) == 4, "sizeof(unsigned int) != 4" ); PresumeFO( ( groupStart >> 31 ) == 0 ); PresumeFO( IsGroupPointer() ); }
		explicit CIndexEntry( VALUE* dataPointer ) : Data( reinterpret_cast<size_t>( dataPointer ) )
			{ PresumeFO( IsDataPointer() ); }

		bool IsFree() const { return Data == 0; }
		bool IsDataPointer() const { return !IsFree() && !IsGroupPointer(); }
		bool IsGroupPointer() const { return ( Data & 1 ) != 0; }

		VALUE* DataPointer() const { PresumeFO( IsDataPointer() ); return reinterpret_cast<VALUE*>( Data ); }
		int NextGroupStart() const { PresumeFO( IsGroupPointer() ); return static_cast<int>( Data >> 1 ); }
		
	private:
		size_t Data;
	};

	enum { AllocatorBlockSize = sizeof( VALUE ) > MinHashTableAllocatorBlockSize 
		? sizeof( VALUE ) 
		: MinHashTableAllocatorBlockSize };

	CArray<CIndexEntry, ALLOCATOR> index;
	int valuesCount;
	int hashTableSize;
	const int initialHashTableSize;
	CHashTableAllocator<ALLOCATOR, AllocatorBlockSize> dataAllocator;

	void growIndex( int minSize );
	void init( int hashSize );
	VALUE* addValue( int hash, const VALUE& value );
	THashTablePosition findIndexFreePos( int hash, int hashTableSize, CArray<CIndexEntry, ALLOCATOR>& index ) const;
	bool canRehash() const;
	THashTablePosition findValueInIndex( const VALUE&, THashTablePosition from ) const;

	THashTablePosition first( int hash, int hashTableSize, const CArray<CIndexEntry, ALLOCATOR>& index ) const;
	THashTablePosition first( int hash ) const;
	THashTablePosition next( THashTablePosition pos, int hashTableSize,
		const CArray<CIndexEntry, ALLOCATOR>& index ) const;
	THashTablePosition next( THashTablePosition ) const;
	
	static int getIndexSize( int hashSize );
	static VALUE* constructByCopy( void* where, const VALUE& copyFrom );

	CHashTable( const CHashTable& );
	void operator=( const CHashTable& );
};

//------------------------------------------------------------------------------------------------------------

int UpperPrimeNumber( int number );

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline CHashTable<VALUE, HASHINFO, ALLOCATOR>::CHashTable() :
	valuesCount( 0 ),
	hashTableSize( 0 ),
	initialHashTableSize( DefHashTableSize )
{
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline CHashTable<VALUE, HASHINFO, ALLOCATOR>::CHashTable( int hashSize ) :
	hashTableSize( 0 ),
	valuesCount( 0 ),
	initialHashTableSize( UpperPrimeNumber( hashSize - 1 ) )
{
	PresumeFO( hashSize >= 0 );
	dataAllocator.Reserve( hashSize );
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline CHashTable<VALUE, HASHINFO, ALLOCATOR>::CHashTable( const VALUE* data, int dataSize ) :
	hashTableSize( 0 ),
	valuesCount( 0 ),
	initialHashTableSize( UpperPrimeNumber( dataSize - 1 ) )
{
	AssertFO( data != 0 );
	AssertFO( dataSize > 0 );

	dataAllocator.Reserve( dataSize );
	init( initialHashTableSize );

	for( int i = 0; i < dataSize; i++ ) {
		Add( *(data + i) );
	}
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline CHashTable<VALUE, HASHINFO, ALLOCATOR>::~CHashTable()
{
	FreeBuffer();
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline void CHashTable<VALUE, HASHINFO, ALLOCATOR>::CopyTo( CHashTable<VALUE, HASHINFO, ALLOCATOR>& dest ) const
{
	if( &dest == this ) {
		return;
	}
	dest.DeleteAll();
	index.CopyTo( dest.index );
	dest.valuesCount = valuesCount;
	dest.hashTableSize = hashTableSize;

	for( int i = 0; i < dest.index.Size(); i++ ) {
		if( dest.index[i].IsDataPointer() ) {
			void* ptr = dest.dataAllocator.Alloc();
			dest.index[i] = CIndexEntry( constructByCopy( ptr, *dest.index[i].DataPointer() ) );
		}
	}
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline void CHashTable<VALUE, HASHINFO, ALLOCATOR>::MoveTo( CHashTable& dest )
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

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline int CHashTable<VALUE, HASHINFO, ALLOCATOR>::Size() const
{
	return valuesCount;
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline void CHashTable<VALUE, HASHINFO, ALLOCATOR>::SetHashTableSize( int size )
{
	dataAllocator.Reserve( size );
	if( size > hashTableSize ) {
		growIndex( size );
	}
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline void CHashTable<VALUE, HASHINFO, ALLOCATOR>::init( int hashSize )
{
	hashTableSize = hashSize;
	index.DeleteAll();
	index.SetSize( hashTableSize );
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline int CHashTable<VALUE, HASHINFO, ALLOCATOR>::getIndexSize( int hashSize )
{
	return hashSize + CeilTo( hashSize / 2, HashIndexGroupLength );
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline VALUE* CHashTable<VALUE, HASHINFO, ALLOCATOR>::constructByCopy( void* where, const VALUE& copyFrom )
{
	PresumeFO( where != AddressOfObject<VALUE>( copyFrom ) );
	return ::new( where ) VALUE( copyFrom );
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline void CHashTable<VALUE, HASHINFO, ALLOCATOR>::Set( const VALUE& value )
{
	if( index.Size() == 0 ) {
		init( initialHashTableSize );
	}

	int hash = HASHINFO::HashKey( value );
	int position = findValueInIndex( value, first( hash ) );
	if( position == NotFound ) {
		addValue( hash, value );
	} else {
		VALUE* ptr = index[position].DataPointer();
		PresumeFO( ptr != AddressOfObject( value ) );
		ptr->~VALUE();
		constructByCopy( ptr, value );
	}
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
template<class TARRAY>
inline void CHashTable<VALUE, HASHINFO, ALLOCATOR>::AddArray( const TARRAY& values )
{
	for( int i = 0; i < values.Size(); i++ ) {
		Add( values[i] );
	}
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline void CHashTable<VALUE, HASHINFO, ALLOCATOR>::Delete( const VALUE& value )
{
	THashTablePosition pos = GetPosition( value );
	AssertFO( pos != NotFound );
	DeleteAt( pos );
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline void CHashTable<VALUE, HASHINFO, ALLOCATOR>::DeleteAt( THashTablePosition pos )
{
	VALUE* data = index[pos].DataPointer();
	index[pos] = CIndexEntry();
	valuesCount--;
	data->~VALUE();
	dataAllocator.Free( data );
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline void CHashTable<VALUE, HASHINFO, ALLOCATOR>::DeleteAll()
{
	if( valuesCount != 0 ) {
		for( int i = 0; i < index.Size(); i++ ) {
			if( index[i].IsDataPointer() ) {
				index[i].DataPointer()->~VALUE();
				dataAllocator.Free( index[i].DataPointer() );
			}
			index[i] = CIndexEntry();
		}
		index.SetSize( hashTableSize );
		valuesCount = 0;
	}
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline THashTablePosition CHashTable<VALUE, HASHINFO, ALLOCATOR>::first( int hashCode, int tableSize,
	const CArray<CIndexEntry, ALLOCATOR>& indexEntry ) const
{
	if( indexEntry.Size() == 0 ) {
		return NotFound;
	}

	AssertFO( tableSize > 0);
	int hash = static_cast<DWORD>( hashCode ) % static_cast<DWORD>( tableSize );
	PresumeFO( 0 <= hash && hash < tableSize );
	if( indexEntry[hash].IsGroupPointer() ) {
		return indexEntry[hash].NextGroupStart();
	} else {
		return hash;
	}
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline THashTablePosition CHashTable<VALUE, HASHINFO, ALLOCATOR>::first( int hashCode ) const
{
	return first( hashCode, hashTableSize, index );
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline THashTablePosition CHashTable<VALUE, HASHINFO, ALLOCATOR>::next( THashTablePosition pos, int tableSize,
	const CArray<CIndexEntry, ALLOCATOR>& indexEntry ) const
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

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline THashTablePosition CHashTable<VALUE, HASHINFO, ALLOCATOR>::next( THashTablePosition pos ) const
{
	return next( pos, hashTableSize, index );
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline void CHashTable<VALUE, HASHINFO, ALLOCATOR>::FreeBuffer()
{
	if( valuesCount != 0 ) {
		for( int i = 0; i < index.Size(); i++ ) {
			if( index[i].IsDataPointer() ) {
				index[i].DataPointer()->~VALUE();
			}
		}
		valuesCount = 0;
	}

	index.FreeBuffer();
	dataAllocator.FreeBuffer();
	hashTableSize = 0;
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline bool CHashTable<VALUE, HASHINFO, ALLOCATOR>::Has( const VALUE& value ) const
{
	return GetPosition( value ) != NotFound;
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline THashTablePosition CHashTable<VALUE, HASHINFO, ALLOCATOR>::GetFirstPosition() const
{
	for( int i = 0; i < index.Size(); i++ ) {
		if( index[i].IsDataPointer() ) {
			return i;
		}
	}
	return NotFound;
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline THashTablePosition CHashTable<VALUE, HASHINFO, ALLOCATOR>::GetPosition( const VALUE& value ) const
{
	int hash = HASHINFO::HashKey( value );
	return findValueInIndex( value, first( hash ) );
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline THashTablePosition CHashTable<VALUE, HASHINFO, ALLOCATOR>::GetNextPosition( THashTablePosition pos ) const
{
	for( int i = pos + 1; i < index.Size(); i++ ) {
		if( index[i].IsDataPointer() ) {
			return i;
		}
	}
	return NotFound;
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline const VALUE& CHashTable<VALUE, HASHINFO, ALLOCATOR>::GetValue( THashTablePosition pos ) const
{
	return *index[pos].DataPointer();
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline const VALUE& CHashTable<VALUE, HASHINFO, ALLOCATOR>::GetOrCreateValue( const VALUE& value )
{
	if( index.Size() == 0 ) {
		init( initialHashTableSize );
	}

	int hash = HASHINFO::HashKey( value );
	int position = findValueInIndex( value, first( hash ) );
	if( position == NotFound ) {
		return *addValue( hash, value );
	}
	return *index[position].DataPointer();
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline THashTablePosition CHashTable<VALUE, HASHINFO, ALLOCATOR>::findValueInIndex( const VALUE& value, 
	THashTablePosition from ) const
{
	for( THashTablePosition pos = from; pos != NotFound; pos = next( pos ) ) {
		if( index[pos].IsDataPointer() && HASHINFO::IsEqual( *index[pos].DataPointer(), value ) ) {
			return pos;
		}
	}
	return NotFound;
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline VALUE* CHashTable<VALUE, HASHINFO, ALLOCATOR>::addValue( int hash, const VALUE& value )
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

	VALUE* ptr = constructByCopy( dataAllocator.Alloc(), value );
	index[freePos] = CIndexEntry( ptr );
	valuesCount++;
	return ptr;
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline THashTablePosition CHashTable<VALUE, HASHINFO, ALLOCATOR>::findIndexFreePos( int hash, int tableSize,
	CArray<CIndexEntry, ALLOCATOR>& indexEntry ) const
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

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline bool CHashTable<VALUE, HASHINFO, ALLOCATOR>::canRehash() const
{
	return ( valuesCount + 1 < hashTableSize / 4 );
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline void CHashTable<VALUE, HASHINFO, ALLOCATOR>::growIndex( int minSize )
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
			VALUE* data = index[i].DataPointer();
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

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline CArchive& operator>>( CArchive& archive, CHashTable<VALUE, HASHINFO, ALLOCATOR>& hashTable )
{
	hashTable.Serialize( archive );
	return archive;
}

template<class VALUE, class HASHINFO, class ALLOCATOR>
inline CArchive& operator<<( CArchive& archive, const CHashTable<VALUE, HASHINFO, ALLOCATOR>& hashTable )
{
	const_cast< CHashTable<VALUE, HASHINFO, ALLOCATOR>& >( hashTable ).Serialize( archive );
	return archive;
}

} // namespace FObj
