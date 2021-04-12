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

#include "../SortFOL.h"
#include "../ArchiveFOL.h"

namespace FObj {

template<class T, int initialBufferSize, class Allocator = CurrentMemoryManager>
class CFastArray {
public:
	typedef T TElement;
	typedef Allocator AllocatorType;

	CFastArray();
	CFastArray( std::initializer_list<T> list );
	~CFastArray();

	CFastArray& operator=( std::initializer_list<T> list );

	int Size() const;
	int BufferSize() const;
	bool IsEmpty() const { return Size() == 0; }
	void Empty() { DeleteAll(); }
	static int InitialBufferSize();
	void SetSize( int newSize );
	void SetBufferSize( int nElem );
	void Grow( int newSize ) { grow( newSize ); }

	void Add( const T& elem );
	void Add( const T& elem, int count );
	void Add( const CFastArray& );
	void Add( std::initializer_list<T> list );
	T& Append() { SetSize( Size() + 1 ); return Last(); }

	T& operator[]( int location );
	const T& operator[]( int location ) const;
	T& Last();
	const T& Last() const;
	T& First();
	const T& First() const;

	bool IsValidIndex( int index ) const;

	T* GetPtr();
	const T* GetPtr() const;
	T* GetBufferPtr() { return dataPtr; }
	const T* GetBufferPtr() const { return dataPtr; }

	void ReplaceAt( const T& elem, int location );
	void InsertAt( const T& elem, int location );
	void InsertAt( const T& elem, int location, int count );
	void InsertAt( const CFastArray& what, int location );
	void InsertAt( std::initializer_list<T> list, int location );
	void DeleteAt( int location, int count );
	void DeleteAt( int location );
	void DeleteLast();
	void DeleteAll();
	void FreeBuffer();

	void CopyTo( CFastArray& dest ) const;
	void MoveTo( CFastArray& dest );

	bool operator==( const CFastArray& other ) const;
	bool operator!=( const CFastArray& other ) const;

	int Find( const T& what, int startPos = 0 ) const;
	bool Has( const T& what ) const { return Find( what ) != NotFound; }

	template <class COMPARE>
	void QuickSort( COMPARE* param );
	template <class COMPARE>
	void QuickSort();

	template<class COMPARE>
	bool IsSorted( COMPARE* compare ) const;
	template<class COMPARE>
	bool IsSorted() const;

	template <class COMPARE, class SEARCHED_TYPE>
	int FindInsertionPoint( const SEARCHED_TYPE& what, COMPARE* param ) const;
	template <class COMPARE, class SEARCHED_TYPE>
	int FindInsertionPoint( const SEARCHED_TYPE& what ) const;

	void Serialize( CArchive& );

private:
	char buffer[initialBufferSize * sizeof( T )];
	T* dataPtr;
	int size;
	int bufferSize;

	void growAt( int pos, int newSize );
	void grow( int newSize );
	void reallocateBuffer( int newSize );

	CFastArray( const CFastArray& );
	CFastArray& operator=( const CFastArray& );
};

template<class T, int initialBufferSize, class Allocator>
inline CFastArray<T, initialBufferSize, Allocator>::CFastArray() : 
	size( 0 ), 
	bufferSize( initialBufferSize )
{
	dataPtr = reinterpret_cast<T*>( buffer );
}

template<class T, int initialBufferSize, class Allocator>
inline CFastArray<T, initialBufferSize, Allocator>::CFastArray( std::initializer_list<T> list ) :
	CFastArray()
{
	Add( list );
}

template<class T, int initialBufferSize, class Allocator>
inline CFastArray<T, initialBufferSize, Allocator>::~CFastArray()
{
	if( dataPtr != ( T* ) buffer ) {
		Allocator::Free( dataPtr );
	}
}

template<class T, int initialBufferSize, class Allocator>
inline CFastArray<T, initialBufferSize, Allocator>& CFastArray<T, initialBufferSize, Allocator>::operator=( 
	std::initializer_list<T> list )
{
	DeleteAll();
	Add( list );
	return *this;
}

template<class T, int initialBufferSize, class Allocator>
inline int CFastArray<T, initialBufferSize, Allocator>::Size() const
{
	return size;
}

template<class T, int initialBufferSize, class Allocator>
inline int CFastArray<T, initialBufferSize, Allocator>::BufferSize() const
{
	return bufferSize;
}

template<class T, int initialBufferSize, class Allocator>
inline int CFastArray<T, initialBufferSize, Allocator>::InitialBufferSize()
{
	return initialBufferSize;
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::SetSize( int newSize )
{
	PresumeFO( newSize >= 0 );
	if( newSize > bufferSize ) {
		grow( newSize );
	}
	size = newSize;
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::grow( int newSize )
{
	PresumeFO( newSize >= 0 );
	if( newSize > bufferSize ) {
		int delta = min( max( newSize - bufferSize, max( bufferSize / 2, initialBufferSize ) ), INT_MAX - bufferSize );
		reallocateBuffer( bufferSize + delta );
	}
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::SetBufferSize( int newSize )
{
	PresumeFO( newSize >= 0 );
	if( newSize > bufferSize ) {
		reallocateBuffer( newSize );
	}
}

template<class T, int initialBufferSize, class Allocator>
inline T* CFastArray<T, initialBufferSize, Allocator>::GetPtr()
{
	if( Size() == 0 ) {
		return 0;
	}
	return dataPtr;
}

template<class T, int initialBufferSize, class Allocator>
inline const T* CFastArray<T, initialBufferSize, Allocator>::GetPtr() const
{
	if( Size() == 0 ) {
		return 0;
	}
	return dataPtr;
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::Add( const T& elem )
{
	PresumeFO( &elem < dataPtr || &elem >= dataPtr + size );
	SetSize( size + 1 );
	dataPtr[size - 1] = elem;
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::Add( const T& elem, int count )
{
	InsertAt( elem, Size(), count );
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::Add( const CFastArray<T, initialBufferSize, Allocator>& ar )
{
	PresumeFO( &ar != this );

	if( ar.Size() > 0 ) {
		int location = size;
		SetSize( size + ar.Size() );
		::memcpy( dataPtr + location, ar.GetPtr(), ar.Size() * sizeof( T ) );
	}
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::Add( std::initializer_list<T> list )
{
	InsertAt( list, Size() );
}

template<class T, int initialBufferSize, class Allocator>
inline const T& CFastArray<T, initialBufferSize, Allocator>::operator [] ( int location ) const
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	return dataPtr[location];
}

template<class T, int initialBufferSize, class Allocator>
inline T& CFastArray<T, initialBufferSize, Allocator>::operator [] ( int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	return dataPtr[location];
}

template<class T, int initialBufferSize, class Allocator>
inline const T& CFastArray<T, initialBufferSize, Allocator>::Last() const
{
	PresumeFO( size > 0 );
	return dataPtr[size - 1];
}

template<class T, int initialBufferSize, class Allocator>
inline T& CFastArray<T, initialBufferSize, Allocator>::Last()
{
	PresumeFO( size > 0 );
	return dataPtr[size - 1];
}

template<class T, int initialBufferSize, class Allocator>
inline const T& CFastArray<T, initialBufferSize, Allocator>::First() const
{
	PresumeFO( size > 0 );
	return dataPtr[0];
}

template<class T, int initialBufferSize, class Allocator>
inline T& CFastArray<T, initialBufferSize, Allocator>::First()
{
	PresumeFO( size > 0 );
	return dataPtr[0];
}

template<class T, int initialBufferSize, class Allocator>
inline bool CFastArray<T, initialBufferSize, Allocator>::IsValidIndex( int index ) const
{
	return index >= 0 && index < size;
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::ReplaceAt( const T& elem, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	dataPtr[location] = elem;
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::growAt( int pos, int newSize )
{
	PresumeFO( newSize > size );

	int delta = newSize - size;
	SetSize( newSize );
	if( size != pos + delta ) {
		::memmove( dataPtr + pos + delta, dataPtr + pos, ( size - pos - delta ) * sizeof( T ) );
	}
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::InsertAt( const T& elem, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );
	PresumeFO( &elem < dataPtr || &elem >= dataPtr + size );

	growAt( location, size + 1 );
	dataPtr[location] = elem;
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::InsertAt( const T& elem, int location, int count )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );
	PresumeFO( count >= 0 );
	PresumeFO( &elem < dataPtr || &elem >= dataPtr + size );

	if( count > 0 ) {
		growAt( location, size + count );
		for( int i = location; i < location + count; i++ ) {
			dataPtr[i] = elem;
		}
	}
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::InsertAt( 
	const CFastArray<T, initialBufferSize, Allocator>& ar, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );
	PresumeFO( &ar != this );

	if( ar.Size() > 0 ) {
		growAt( location, size + ar.Size() );
		::memcpy( dataPtr + location, ar.GetPtr(), ar.Size() * sizeof( T ) );
	}
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::InsertAt( std::initializer_list<T> list, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );

	const int listSize = to<int>( list.size() );
	if( listSize > 0 ) {
		growAt( location, size + listSize );
		T* ptr = dataPtr + location;
		for( const T& element : list ) {
			*ptr = element;
			ptr++;
		}
	}
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::DeleteAt( int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	if( size != location + 1 ) {
		::memmove( dataPtr + location, dataPtr + location + 1, ( size - location - 1 ) * sizeof( T ) );
	}
	--size;
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::DeleteAt( int location, int count )
{
	PresumeFO( count >= 0 );
	PresumeFO( location >= 0 );
	PresumeFO( location + count <= size );
	if( count > 0 ) {
		if( size != location + count ) {
			::memmove( dataPtr + location, dataPtr + location + count, ( size - location - count ) * sizeof( T ) );
		}
		size -= count;
	}
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::DeleteLast()
{
	PresumeFO( size > 0 );
	size--;
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::DeleteAll()
{
	size = 0;
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::FreeBuffer()
{
	DeleteAll();
	reallocateBuffer( 0 );
}

template<class T, int initialBufferSize, class Allocator> template <class COMPARE>
inline void CFastArray<T, initialBufferSize, Allocator>::QuickSort( COMPARE* param )
{
	FObj::QuickSort<T, COMPARE>( GetPtr(), Size(), param );
}

template<class T, int initialBufferSize, class Allocator> template <class COMPARE>
inline void CFastArray<T, initialBufferSize, Allocator>::QuickSort()
{
	FObj::QuickSort<T, COMPARE>( GetPtr(), Size() );
}

template<class T, int initialBufferSize, class Allocator> template <class COMPARE>
inline bool CFastArray<T, initialBufferSize, Allocator>::IsSorted( COMPARE* compare ) const
{
	return FObj::IsSorted<T, COMPARE>( GetPtr(), Size(), compare );
}

template<class T, int initialBufferSize, class Allocator> template <class COMPARE>
inline bool CFastArray<T, initialBufferSize, Allocator>::IsSorted() const
{
	return FObj::IsSorted<T, COMPARE>( GetPtr(), Size() );
}

template<class T, int initialBufferSize, class Allocator> template <class COMPARE, class SEARCHED_TYPE>
inline int CFastArray<T, initialBufferSize, Allocator>::FindInsertionPoint( const SEARCHED_TYPE& what, 
	COMPARE* param ) const
{
	return FObj::FindInsertionPoint<T, COMPARE, SEARCHED_TYPE>( what, GetPtr(), Size(), param );
}

template<class T, int initialBufferSize, class Allocator> template <class COMPARE, class SEARCHED_TYPE>
inline int CFastArray<T, initialBufferSize, Allocator>::FindInsertionPoint( const SEARCHED_TYPE& what ) const
{
	return FObj::FindInsertionPoint<T, COMPARE, SEARCHED_TYPE>( what, GetPtr(), Size() );
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::Serialize( CArchive& arch )
{
	if( arch.IsLoading() ) {
		unsigned int nElems;
		arch >> nElems;
		check( static_cast<int>( nElems ) >= 0, ERR_BAD_ARCHIVE, arch.Name() );
		SetBufferSize( nElems );
		SetSize( nElems );
		arch.Read( dataPtr, nElems * sizeof( T ) );
	} else {
		arch << static_cast<unsigned int>( Size() );
		arch.Write( dataPtr, Size() * sizeof( T ) );
	}
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::reallocateBuffer( int newSize )
{
	PresumeFO( newSize >= size );

	if( newSize > InitialBufferSize() ) {
		T* oldDataPtr = dataPtr;
		AssertFO( newSize <= UINTPTR_MAX / sizeof( T ) );
		dataPtr = static_cast<T*>( ALLOCATE_MEMORY( Allocator, newSize * sizeof( T ) ) );
		if( size > 0 ) {
			::memcpy( reinterpret_cast<char*>( dataPtr ), reinterpret_cast<char*>( oldDataPtr ), size * sizeof( T ) );
		}
		if( oldDataPtr != ( T* ) buffer ) {
			Allocator::Free( oldDataPtr );
		}
		bufferSize = newSize;
	} else if( dataPtr != ( T* ) buffer ) {
		if( size > 0 ) {
			::memcpy( buffer, reinterpret_cast<char*>( dataPtr ), size * sizeof( T ) );
		}
		Allocator::Free( dataPtr );
		dataPtr = ( T* ) buffer;
		bufferSize = InitialBufferSize();
	}
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::CopyTo( 
	CFastArray<T, initialBufferSize, Allocator>& dest ) const
{
	if( &dest == this ) {
		return;
	}

	dest.DeleteAll();
	dest.SetBufferSize( size );

	dest.size = size;
	::memcpy( dest.dataPtr, dataPtr, size * sizeof( T ) );
}

template<class T, int initialBufferSize, class Allocator>
inline void CFastArray<T, initialBufferSize, Allocator>::MoveTo( CFastArray<T, initialBufferSize, Allocator>& dest )
{
	if( &dest == this ) {
		return;
	}

	if( dataPtr == ( T* ) buffer ) {
		::memcpy( dest.dataPtr, dataPtr, size * sizeof( T ) );
		dest.size = size;
	} else {
		dest.FreeBuffer();
		dest.dataPtr = dataPtr;
		dest.bufferSize = bufferSize;
		dest.size = size;
		dataPtr = ( T* ) buffer;
	}

	size = 0;
	bufferSize = initialBufferSize;
}

template<class T, int initialBufferSize, class Allocator>
inline bool CFastArray<T, initialBufferSize, Allocator>::operator==( const CFastArray& other ) const
{
	if( this == &other ) {
		return true;
	}
	if( Size() != other.Size() ) {
		return false;
	}
	for( int i = 0; i < Size(); i++ ) {
		if( !( (*this)[i] == other[i] ) ) {
			return false;
		}
	}
	return true;
}

template<class T, int initialBufferSize, class Allocator>
inline bool CFastArray<T, initialBufferSize, Allocator>::operator!=( const CFastArray& other ) const
{
	return !( *this == other );
}

template<class T, int initialBufferSize, class Allocator>
inline int CFastArray<T, initialBufferSize, Allocator>::Find( const T& what, int startPos ) const
{
	PresumeFO( startPos >= 0 );
	for( int i = startPos; i < Size(); i++ ) {
		if( what == ( *this )[i] ) {
			return i;
		}
	}
	return NotFound;
}

//--------------------------------------------------------------------------------------------------------------

template<class T, int InitialBufferSize, class Allocator>
inline void ArrayMemMoveElement( CFastArray<T, InitialBufferSize, Allocator>* dest, 
	CFastArray<T, InitialBufferSize, Allocator>* source )
{
	PresumeFO( dest != source );
	::new( dest ) CFastArray<T, InitialBufferSize, Allocator>;
	source->MoveTo( *dest );
	source->~CFastArray<T, InitialBufferSize, Allocator>();
}

} // namespace FObj
