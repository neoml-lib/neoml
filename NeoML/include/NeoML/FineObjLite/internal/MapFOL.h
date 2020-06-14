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

namespace FObj {

typedef int TMapPosition;

template<class KEY, class VALUE>
struct CMapData {
	KEY Key;
	VALUE Value;

	CMapData( const KEY& key, const VALUE& val ) : Key( key ), Value( val ) {}
	explicit CMapData( const KEY& key ) : Key( key ), Value() {}
	CMapData() : Key(), Value() {}
};

const int DefMapHashTableSize = 31;
const int MapIndexGroupLength = 4;

//------------------------------------------------------------------------------------------------------------

template<class KEY, class VALUE, class KEYHASHINFO = CDefaultHash<KEY>, class ALLOCATOR = CurrentMemoryManager>
class CMap {
public:
	typedef KEY KeyType;
	typedef VALUE ValueType;
	typedef ALLOCATOR AllocatorType;
	typedef KEYHASHINFO KeyHashInfoType;
	typedef CMapData<const KEY, VALUE> TElement;

	CMap();
	explicit CMap( int hashSize );
	template<class MapInitStruct>
	CMap( const MapInitStruct* data, int dataSize );
	CMap( std::initializer_list<TElement> list );
	~CMap();

	CMap& operator=( std::initializer_list<TElement> list );

	void CopyTo( CMap& ) const;
	void MoveTo( CMap& );

	int Size() const;
	void SetHashTableSize( int hashSize );

	bool IsEmpty() const { return Size() == 0; }
	void Empty() { DeleteAll(); }
	void Set( const KEY&, const VALUE& );
	void Add( const KEY&, const VALUE& );
	void Add( std::initializer_list<TElement> list );
	VALUE& CreateValue( const KEY& );
	VALUE& AddValue( const KEY& );

	bool Has( const KEY& ) const;
	bool Lookup( const KEY& key, VALUE& resultValue ) const;
	template<class ARRAY>
	bool LookupAllValues( const KEY& key, ARRAY& values ) const;
	template<class ARRAY>
	bool LookupAllValues( const KEY& key, ARRAY& values );
	const VALUE& Get( const KEY& ) const;
	VALUE& Get( const KEY& );
	VALUE& GetOrCreateValue( const KEY& );
	VALUE& GetOrCreateValue( const KEY& key, const VALUE& defaultValue );

	const VALUE& operator [] ( const KEY& key ) const { return Get( key ); }
	VALUE& operator [] ( const KEY& key ) { return Get( key ); }

	void Delete( const KEY& );
	void DeleteAt( TMapPosition );
	void DeleteAll();
	void FreeBuffer();

	TMapPosition GetFirstPosition() const;
	TMapPosition GetNextPosition( TMapPosition pos ) const;

	TMapPosition GetFirstPosition( const KEY& ) const;
	TMapPosition GetNextPosition( const KEY&, TMapPosition pos ) const;

	const KEY& GetKey( TMapPosition ) const;
	const VALUE& GetValue( TMapPosition ) const;
	VALUE& GetValue( TMapPosition );

	const CMapData<const KEY, VALUE>& GetKeyValue( TMapPosition pos ) const;
	CMapData<const KEY, VALUE>& GetKeyValue( TMapPosition pos );

	void Serialize( CArchive& ar );

private:
	class CIndexEntry {
	public:
		CIndexEntry() : Data( 0 ) { PresumeFO( IsFree() ); }
		explicit CIndexEntry( unsigned int groupStart ) : Data( ( groupStart << 1 ) | 1 )
			{ static_assert( sizeof(unsigned int) == 4, "sizeof(unsigned int) != 4" ); PresumeFO( ( groupStart >> 31 ) == 0 ); PresumeFO( IsGroupPointer() ); }
		explicit CIndexEntry( CMapData<KEY, VALUE>* dataPointer ) : Data( reinterpret_cast<size_t>( dataPointer ) )
			{ PresumeFO( IsDataPointer() ); }

		bool IsFree() const { return Data == 0; }
		bool IsDataPointer() const { return !IsFree() && !IsGroupPointer(); }
		bool IsGroupPointer() const { return ( Data & 1 ) != 0; }

		CMapData<KEY, VALUE>* DataPointer() const { PresumeFO( IsDataPointer() ); return reinterpret_cast<CMapData<KEY, VALUE>*>( Data ); }
		int NextGroupStart() const { PresumeFO( IsGroupPointer() ); return static_cast<int>( Data >> 1 ); }
		
	private:
		size_t Data;
	};

	enum {
		AllocatorBlockSize = sizeof( CMapData<KEY, VALUE> ) > MinHashTableAllocatorBlockSize ?
			sizeof( CMapData<KEY, VALUE> ) : MinHashTableAllocatorBlockSize
	};

	CArray<CIndexEntry, ALLOCATOR> index;
	int valuesCount;
	int hashTableSize;
	const int initialHashTableSize;
	CHashTableAllocator<ALLOCATOR, AllocatorBlockSize> dataAllocator;

	void growIndex( int minSize );
	void init( int hashSize );
	VALUE& addValue( int hash, const KEY&, const VALUE& );
	VALUE& addValue( int hash, const KEY& );
	void deleteAllValues( int hash, const KEY& );
	TMapPosition findIndexFreePos( int hash, int hashTableSize, CArray<CIndexEntry, ALLOCATOR>& index ) const;
	bool canRehash() const;
	TMapPosition findIndexFreePosWithGrow( int hash );
	TMapPosition findKeyInIndex( const KEY&, TMapPosition from ) const;
	
	TMapPosition first( int hashCode, int hashTableSize, const CArray<CIndexEntry, ALLOCATOR>& index ) const;
	TMapPosition first( int hash ) const;
	TMapPosition next( TMapPosition pos, int hashTableSize, const CArray<CIndexEntry, ALLOCATOR>& index ) const;
	TMapPosition next( TMapPosition ) const;
	
	static int getIndexSize( int hashSize );

	CMap( const CMap& );
	void operator=( const CMap& );
};

//------------------------------------------------------------------------------------------------------------

extern int UpperPrimeNumber( int number );

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CMap() :
	valuesCount( 0 ),
	hashTableSize( 0 ),
	initialHashTableSize( DefMapHashTableSize )
{
}
	
template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CMap( int hashSize ) :
	hashTableSize( 0 ),
	valuesCount( 0 ),
	initialHashTableSize( UpperPrimeNumber( hashSize - 1 ) )
{
	PresumeFO( hashSize >= 0 );
	dataAllocator.Reserve( hashSize );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class MapInitStruct>
inline CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CMap( const MapInitStruct* data, int dataSize ) :
	hashTableSize( 0 ),
	valuesCount( 0 ),
	initialHashTableSize( UpperPrimeNumber( dataSize - 1 ) )
{
	AssertFO( data != 0 );
	AssertFO( dataSize > 0 );

	dataAllocator.Reserve( dataSize );
	init( initialHashTableSize );

	for( int i = 0; i < dataSize; i++ ) {
		const MapInitStruct& entry = *(data + i);
		Add( entry.Key, entry.Value );
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CMap( std::initializer_list<TElement> list ) :
	CMap()
{
	Add( list );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::~CMap()
{
	FreeBuffer();
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::operator=( 
	std::initializer_list<TElement> list )
{
	DeleteAll();
	Add( list );
	return *this;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline int CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Size() const
{
	return valuesCount;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::SetHashTableSize( int size )
{
	dataAllocator.Reserve( size );
	if( size > hashTableSize ) {
		growIndex( size );
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::init( int hashSize )
{
	hashTableSize = hashSize;
	index.DeleteAll();
	index.SetSize( hashTableSize );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline int CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::getIndexSize( int hashSize )
{
	return hashSize + CeilTo( hashSize / 2, MapIndexGroupLength );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Set( const KEY& key, const VALUE& value )
{
	int hash = KEYHASHINFO::HashKey( key );
	deleteAllValues( hash, key );
	addValue( hash, key, value );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline VALUE& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CreateValue( const KEY& key )
{
	int hash = KEYHASHINFO::HashKey( key );
	deleteAllValues( hash, key );
	return addValue( hash, key );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Add( const KEY& key, const VALUE& value )
{
	int hash = KEYHASHINFO::HashKey( key );
	addValue( hash, key, value );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Add( std::initializer_list<TElement> list )
{
	SetHashTableSize( Size() + to<int>( list.size() ) );
	for( const TElement& element : list ) {
		Add( element.Key, element.Value );
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline VALUE& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::AddValue( const KEY& key )
{
	int hash = KEYHASHINFO::HashKey( key );
	return addValue( hash, key );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline bool CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Lookup( const KEY& key, VALUE& value ) const
{
	const TMapPosition position = GetFirstPosition( key );
	if( position == NotFound ) {
		return false;
	}
	value = GetValue( position );
	return true;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class ARRAY>
inline bool CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::LookupAllValues( const KEY& key, ARRAY& values ) const
{
	values.DeleteAll();
	int hash = KEYHASHINFO::HashKey( key );

	for( TMapPosition pos = first( hash ); pos != NotFound; pos = next( pos ) ) {
		if( index[pos].IsDataPointer() && KEYHASHINFO::IsEqual( index[pos].DataPointer()->Key, key ) ) {
			values.Add( GetValue(pos) );
		}
	}
	return !values.IsEmpty();
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
template<class ARRAY>
inline bool CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::LookupAllValues( const KEY& key, ARRAY& values )
{
	values.DeleteAll();
	int hash = KEYHASHINFO::HashKey( key );

	for( TMapPosition pos = first( hash ); pos != NotFound; pos = next( pos ) ) {
		if( index[pos].IsDataPointer() && KEYHASHINFO::IsEqual( index[pos].DataPointer()->Key, key ) ) {
			values.Add( GetValue(pos) );
		}
	}
	return !values.IsEmpty();
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline VALUE& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Get( const KEY& key )
{
	TMapPosition position = GetFirstPosition( key );
	AssertFO( position != NotFound );
	return GetValue( position );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline const VALUE& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Get( const KEY& key ) const
{
	TMapPosition position = GetFirstPosition( key );
	AssertFO( position != NotFound );
	return GetValue( position );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline VALUE& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetOrCreateValue( const KEY& key )
{
	int hash = KEYHASHINFO::HashKey( key );
	TMapPosition position = findKeyInIndex( key, first( hash ) );
	if( position != NotFound ) {
		return GetValue( position );
	}
	return addValue( hash, key );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline VALUE& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetOrCreateValue( const KEY& key, const VALUE& value )
{
	int hash = KEYHASHINFO::HashKey( key );
	TMapPosition position = findKeyInIndex( key, first( hash ) );
	if( position != NotFound ) {
		return GetValue( position );
	}
	return addValue( hash, key, value );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline VALUE& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::addValue( int hash, const KEY& key, const VALUE& value )
{
	TMapPosition freePos = findIndexFreePosWithGrow( hash );
	CMapData<KEY, VALUE>* ptr = new( dataAllocator.Alloc() ) CMapData<KEY, VALUE>( key, value );
	valuesCount++;
	index[freePos] = CIndexEntry( ptr );
	return ptr->Value;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline VALUE& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::addValue( int hash, const KEY& key )
{
	TMapPosition freePos = findIndexFreePosWithGrow( hash );
	CMapData<KEY, VALUE>* ptr = new( dataAllocator.Alloc() ) CMapData<KEY, VALUE>( key );
	valuesCount++;
	index[freePos] = CIndexEntry( ptr );
	return ptr->Value;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Delete( const KEY& key )
{
	int hash = KEYHASHINFO::HashKey( key );
	deleteAllValues( hash, key );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::deleteAllValues( int hash, const KEY& key )
{
	for( TMapPosition pos = first( hash ); pos != NotFound; pos = next( pos ) ) {
		if( index[pos].IsDataPointer() && KEYHASHINFO::IsEqual( index[pos].DataPointer()->Key, key ) ) {
			DeleteAt( pos );
		}
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::DeleteAt( TMapPosition pos )
{
	CMapData<KEY, VALUE>* data = index[pos].DataPointer();
	index[pos] = CIndexEntry();
	valuesCount--;
	data->~CMapData<KEY, VALUE>();
	dataAllocator.Free( data );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::DeleteAll()
{
	if( valuesCount != 0 ) {
		for( int i = 0; i < index.Size(); i++ ) {
			if( index[i].IsDataPointer() ) {
				index[i].DataPointer()->~CMapData<KEY, VALUE>();
				dataAllocator.Free( index[i].DataPointer() );
			}
			index[i] = CIndexEntry();
		}
		index.SetSize( hashTableSize );
		valuesCount = 0;
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::CopyTo( CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>& dest ) const
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
			dest.index[i] = CIndexEntry( new( ptr ) CMapData<KEY, VALUE>( *dest.index[i].DataPointer() ) );
		}
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::MoveTo( CMap& dest )
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
inline TMapPosition CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::first( int hashCode, int tableSize,
	const CArray<CIndexEntry, ALLOCATOR>& indexEntry ) const
{
	if( indexEntry.Size() == 0 ) {
		return NotFound;
	}

	AssertFO( tableSize > 0 );
	int hash = (unsigned int)( hashCode ) % (unsigned int)( tableSize );
	PresumeFO( 0 <= hash && hash < tableSize );
	if( indexEntry[hash].IsGroupPointer() ) {
		return indexEntry[hash].NextGroupStart();
	} else {
		return hash;
	}
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline TMapPosition CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::first( int hashCode ) const
{
	return first( hashCode, hashTableSize, index );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline TMapPosition CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::next( TMapPosition pos, int tableSize,
	const CArray<CIndexEntry, ALLOCATOR>& indexEntry ) const
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
inline TMapPosition CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::next( TMapPosition pos ) const
{
	return next( pos, hashTableSize, index );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::FreeBuffer()
{
	if( valuesCount != 0 ) {
		for( int i = 0; i < index.Size(); i++ ) {
			if( index[i].IsDataPointer() ) {
				index[i].DataPointer()->~CMapData<KEY, VALUE>();
			}
		}
		valuesCount = 0;
	}

	index.FreeBuffer();
	dataAllocator.FreeBuffer();
	hashTableSize = 0;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline bool CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Has( const KEY& key ) const
{
	return GetFirstPosition( key ) != NotFound;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline TMapPosition CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetFirstPosition() const
{
	for( int i = 0; i < index.Size(); i++ ) {
		if( index[i].IsDataPointer() ) {
			return i;
		}
	}
	return NotFound;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline TMapPosition CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetFirstPosition( const KEY& key ) const
{
	int hash = KEYHASHINFO::HashKey( key );
	return findKeyInIndex( key, first( hash ) );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline TMapPosition CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetNextPosition( TMapPosition pos ) const
{
	for( int i = pos + 1; i < index.Size(); i++ ) {
		if( index[i].IsDataPointer() ) {
			return i;
		}
	}
	return NotFound;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline TMapPosition CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetNextPosition( const KEY& key, TMapPosition prev ) const
{
	return findKeyInIndex( key, next( prev ) );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline const VALUE& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetValue( TMapPosition pos ) const
{
	return index[pos].DataPointer()->Value;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline VALUE& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetValue( TMapPosition pos )
{
	return index[pos].DataPointer()->Value;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline const KEY& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetKey( TMapPosition pos ) const
{
	return index[pos].DataPointer()->Key;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline const CMapData<const KEY, VALUE>& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetKeyValue( TMapPosition pos ) const
{
	return *reinterpret_cast<CMapData<const KEY, VALUE>*>( index[pos].DataPointer() );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline CMapData<const KEY, VALUE>& CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::GetKeyValue( TMapPosition pos )
{
	return *reinterpret_cast<CMapData<const KEY, VALUE>*>( index[pos].DataPointer() );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline TMapPosition CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::findKeyInIndex( const KEY& key, TMapPosition from ) const
{
	for( TMapPosition pos = from; pos != NotFound; pos = next( pos ) ) {
		if( index[pos].IsDataPointer() && KEYHASHINFO::IsEqual( index[pos].DataPointer()->Key, key ) ) {
			return pos;
		}
	}
	return NotFound;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline TMapPosition CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::findIndexFreePos( int hash, int tableSize,
	CArray<CIndexEntry, ALLOCATOR>& indexEntry ) const
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
inline bool CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::canRehash() const
{
	return ( valuesCount + 1 < hashTableSize / 4 );
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline TMapPosition CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::findIndexFreePosWithGrow( int hash )
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
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::growIndex( int minSize )
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
			CMapData<KEY, VALUE>* data = index[i].DataPointer();
			int hash = KEYHASHINFO::HashKey( data->Key );
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

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline void CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>::Serialize( CArchive& ar )
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
inline CArchive& operator>>( CArchive& archive, CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>& map )
{
	map.Serialize( archive );
	return archive;
}

template<class KEY, class VALUE, class KEYHASHINFO, class ALLOCATOR>
inline CArchive& operator<<( CArchive& archive, const CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>& map )
{
	const_cast< CMap<KEY, VALUE, KEYHASHINFO, ALLOCATOR>& >( map ).Serialize( archive );
	return archive;
}

} // namespace FObj
