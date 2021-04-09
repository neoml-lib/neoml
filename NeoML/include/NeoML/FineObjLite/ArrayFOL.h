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

#include "SortFOL.h"
#include "ArchiveFOL.h"
#include "ObjectFOL.h"

namespace FObj {

// Check if the type may be bitwise moved in memory
template<typename T>
struct IsMemmoveable {
	static const bool Value =
		sizeof(T) <= sizeof( void* ) || std::is_trivially_copyable<T>::value;
};

template<class T>
inline void ArrayMemMoveElement( T* dest, T* source );

// The template function used by CArray to move elements in the buffer
// It uses the copy constructor
template<class T>
inline void ArrayMemMove( T* dest, T* source, int count )
{
	if( dest > source ) {
		for( int i = count - 1; i >= 0; i-- ) {
			ArrayMemMoveElement( dest + i, source + i );
		}
	} else {
		for( int i = 0; i < count; i++ ) {
			ArrayMemMoveElement( dest + i, source + i );
		}
	}
}

template<class T>
inline void ArrayMemMoveElement( T* dest, T* source )
{
	PresumeFO( dest != source );
	::new( dest ) T( *source );
	source->~T();
}

template<class T>
inline void ArrayMemMoveBitwize( T* dest, T* source, int count )
{
	::memmove( reinterpret_cast<char*>( dest ), reinterpret_cast<char*>( source ), count * sizeof( T ) );
}

/*
// Specialized IsMemmoveable to indicate that CArray can be moved in memory
template<class T, class Allocator>
struct IsMemmoveable< CArray<T, Allocator> > {
	static const bool Value = true;
};
*/

// Array template. Elements are added with the help of copy constructor
// When elements are deleted their destructors are called
template<class T, class Allocator = CurrentMemoryManager>
class CArray {
	struct CDataHolder {
		CDataHolder( const T& data ) : Data( data ) {}
		~CDataHolder() {}
		T Data;
	};
	struct CDataHolderExt {
		CDataHolderExt() {}
		T Data;
	};

public:
	typedef T TElement;
	typedef Allocator AllocatorType;

	CArray();
	CArray( std::initializer_list<T> list );
	~CArray() { FreeBuffer(); }

	CArray& operator=( std::initializer_list<T> list );

	// The number of elements in the array
	int Size() const;
	// Gets the size of the current memory buffer for the array, in terms of elements
	// Buffer size may be greater than array size
	int BufferSize() const;
	// Returns true if there are no elements in the array
	bool IsEmpty() const { return Size() == 0; }
	void Empty() { DeleteAll(); }
	// Returns the pointer to the first element in the array
	// If the array is empty, 0 is returned
	T* GetPtr();
	const T* GetPtr() const;
	// Accessing elements by index
	const T& operator [] ( int location ) const;
	T& operator [] ( int location );
	// Gets the last element in the array
	const T& Last() const;
	T& Last();
	// Gets the first element in the array
	const T& First() const;
	T& First();

	// Checks if an index is valid for this array
	bool IsValidIndex( int index ) const;

	// Sets the buffer size, in terms of elements 
	// You may only increase the size with this method
	void SetBufferSize( int nElements );
	// Grows the buffer size according to the growth policy
	void Grow( int newSize ) { grow( newSize ); }
	// Sets the array size
	void SetSize( int newSize );

	// Adds elements to the end of the array. Copy constructor is used
	void Add( const T& anElem );
	void Add( const T& anElem, int count ) { InsertAt( anElem, Size(), count ); }
	void Add( const CArray& ar ) { InsertAt( ar, Size() ); }
	void Add( std::initializer_list<T> list ) { InsertAt( list, Size() ); }
	// Adds an "empty" element to the end of the array
	T& Append() { SetSize( Size() + 1 ); return Last(); }
	// Inserts an element into the given position in the array (including the last)
	// Copy constructor is used
	void InsertAt( const T& what, int location );
	void InsertAt( const T& what, int location, int count );
	void InsertAt( const CArray& what, int location );
	void InsertAt( std::initializer_list<T> list, int location );
	// Replaces an element in the array. The old element is deleted using its destructor
	// The new element is copied into the array using the copy constructor
	void ReplaceAt( const T& newElem, int location );
	// Deletes elements from the array. Their destructors will be called
	// The buffer size does not decrease when deleting
	void DeleteAt( int location, int num = 1 );
	// Deletes the last element in the array
	// presume is used to check if the element is there
	void DeleteLast();
	// Deletes all elements from the array. Their destructors will be called
	// The array buffer is not cleared when deleting
	void DeleteAll();
	// Deletes all element from the array and clears the array buffer
	// The destructors will be called for all elements
	void FreeBuffer();

	// Copies the array into another array. The original elements of the target array are deleted
	void CopyTo( CArray& dest ) const;
	// Moves the array into another array. The original elements of the target array are deleted
	void MoveTo( CArray& dest );

	// Checking if the arrays are equal
	bool operator==( const CArray& other ) const;
	bool operator!=( const CArray& other ) const;

	// Linear search in the array. Operator == is used for comparison
	int Find( const T& what, int startPos = 0 ) const;
	// Checks if the array contains the specified element. Uses linear search and comparison operator
	bool Has( const T& what ) const { return Find( what ) != NotFound; }

	// Sorts the array with the help of comparison class (see AscendingFOL.h, DescendingFOL.h for examples)
	template<class COMPARE>
	void QuickSort( COMPARE* param );
	template<class COMPARE>
	void QuickSort();

	// Checks if the array is sorted
	template<class COMPARE>
	bool IsSorted( COMPARE* compare ) const;
	template<class COMPARE>
	bool IsSorted() const;

	// Binary searches for the correct position to insert an element into the sorted array
	template<class COMPARE, class SEARCHED_TYPE>
	int FindInsertionPoint( const SEARCHED_TYPE& what, COMPARE* param ) const;
	template<class COMPARE, class SEARCHED_TYPE>
	int FindInsertionPoint( const SEARCHED_TYPE& what ) const;

	void Serialize( CArchive& );

private:
	int size;
	int bufferSize;
	CDataHolder* dataPtr;

	void growAt( int location, int newSize );
	void grow( int newSize );
	void reallocateBuffer( int newSize );
	static void moveData( CDataHolder* destDataPtr, int destIndex, CDataHolder* srcDataPtr, int srcIndex, int count );

	CArray( const CArray& );
	CArray& operator=( const CArray& );
};

template<class T, class Allocator>
inline CArray<T, Allocator>::CArray() :
	size( 0 ),
	bufferSize( 0 ),
	dataPtr( 0 )
{
	static_assert( sizeof( T ) == sizeof( CDataHolder ), "sizeof( T ) != sizeof( CDataHolder )" );
}

template<class T, class Allocator>
inline CArray<T, Allocator>::CArray( std::initializer_list<T> list ) :
	CArray()
{
	Add( list );
}

template<class T, class Allocator>
inline CArray<T, Allocator>& CArray<T, Allocator>::operator=( std::initializer_list<T> list )
{
	DeleteAll();
	Add( list );
	return *this;
}

template<class T, class Allocator>
inline int CArray<T, Allocator>::Size() const
{
	return size;
}

template<class T, class Allocator>
inline int CArray<T, Allocator>::BufferSize() const
{
	return bufferSize;
}

template<class T, class Allocator>
inline T* CArray<T, Allocator>::GetPtr()
{
	if( Size() == 0 ) {
		return 0;
	}
	return ( T* )dataPtr;
}

template<class T, class Allocator>
inline const T* CArray<T, Allocator>::GetPtr() const
{
	if( Size() == 0 ) {
		return 0;
	}
	return ( const T* )dataPtr;
}

template<class T, class Allocator>
inline const T& CArray<T, Allocator>::operator [] ( int location ) const
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	return dataPtr[location].Data;
}

template<class T, class Allocator>
inline T& CArray<T, Allocator>::operator [] ( int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	return dataPtr[location].Data;
}

template<class T, class Allocator>
inline const T& CArray<T, Allocator>::Last() const
{
	PresumeFO( size > 0 );
	return dataPtr[size - 1].Data;
}

template<class T, class Allocator>
inline T& CArray<T, Allocator>::Last()
{
	PresumeFO( size > 0 );
	return dataPtr[size - 1].Data;
}

template<class T, class Allocator>
inline const T& CArray<T, Allocator>::First() const
{
	PresumeFO( size > 0 );
	return dataPtr[0].Data;
}

template<class T, class Allocator>
inline T& CArray<T, Allocator>::First()
{
	PresumeFO( size > 0 );
	return dataPtr[0].Data;
}

template<class T, class Allocator>
inline bool CArray<T, Allocator>::IsValidIndex( int index ) const
{
	return index >= 0 && index < size;
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::SetBufferSize( int newSize )
{
	PresumeFO( newSize >= 0 );
	if( newSize > bufferSize ) {
		reallocateBuffer( newSize );
	}
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::SetSize( int newSize )
{
	PresumeFO( newSize >= 0 );

	for( int index = newSize; index < size; index++ ) {
		dataPtr[index].~CDataHolder();
	}

	grow( newSize );

	for( int i = size; i < newSize; i++ ) {
		::new( &dataPtr[i] ) CDataHolderExt();
	}
	size = newSize;
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::Add( const T& what )
{
	// The elements from the same array may not be inserted
	PresumeFO( dataPtr == 0 || AddressOfObject( what ) < ( T* )dataPtr || AddressOfObject( what ) >= ( T* )( dataPtr + size ) );
	PresumeFO( size <= bufferSize );
	if( size + 1 > bufferSize ) {
		grow( size + 1 );
	}
	::new( ( void* )&dataPtr[size] ) CDataHolder( what );
	size++;
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::InsertAt( const T& what, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );
	// The elements from the same array may not be inserted
	PresumeFO( dataPtr == 0 || AddressOfObject( what ) < ( T* )dataPtr || AddressOfObject( what ) >= ( T* )( dataPtr + size ) );

	growAt( location, size + 1 );
	::new( ( void* )&dataPtr[location] ) CDataHolder( what );
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::InsertAt( const T& what, int location, int count )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );
	PresumeFO( count >= 0 );
	// The elements from the same array may not be inserted
	PresumeFO( dataPtr == 0 || AddressOfObject( what ) < ( T* )dataPtr || AddressOfObject( what ) >= ( T* )( dataPtr + size ) );

	if( count > 0 ) {
		growAt( location, size + count );
		for( int i = 0; i < count; i++ ) {
			::new( ( void* )&dataPtr[location + i] ) CDataHolder( what );
		}
	}
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::InsertAt( const CArray<T, Allocator>& what, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );
	PresumeFO( &what != this );

	if( what.Size() > 0 ) {
		growAt( location, size + what.Size() );
		for( int i = 0; i < what.Size(); i++ ) {
			::new( &dataPtr[location + i] ) CDataHolder( what[i] );
		}
	}
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::InsertAt( std::initializer_list<T> list, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );

	const int listSize = to<int>( list.size() );
	if( listSize > 0 ) {
		growAt( location, size + listSize );
		int pos = location;
		for( const T& element : list ) {
			::new( &dataPtr[pos] ) CDataHolder( element );
			pos++;
		}
	}
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::ReplaceAt( const T& newElem, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	PresumeFO( AddressOfObject( newElem ) != AddressOfObject( dataPtr[location].Data ) );

	dataPtr[location].~CDataHolder();
	::new( &dataPtr[location] ) CDataHolder( newElem );
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::DeleteAt( int location, int num )
{
	PresumeFO( num >= 0 );
	PresumeFO( num <= size );
	PresumeFO( location >= 0 );
	PresumeFO( location <= size - num );
	if( num == 0 ) {
		return;
	}

	for( int index = location + num - 1; index >= location; index-- ) {
		dataPtr[index].~CDataHolder();
	}

	moveData( dataPtr, location, dataPtr, location + num, size - location - num );

	size -= num;
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::DeleteLast()
{
	PresumeFO( size > 0 );

	dataPtr[size - 1].~CDataHolder();
	size--;
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::DeleteAll()
{
	for( int index = size - 1; index >= 0; index-- ) {
		dataPtr[index].~CDataHolder();
	}
	size = 0;
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::FreeBuffer()
{
	DeleteAll();

	CDataHolder* oldDataPtr = dataPtr;
	dataPtr = 0;
	if( oldDataPtr != 0 ) {
		Allocator::Free( oldDataPtr );
	}
	bufferSize = 0;
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::CopyTo( CArray<T, Allocator>& dest ) const
{
	if( &dest == this ) {
		return;
	}

	dest.DeleteAll();
	dest.SetBufferSize( size );
	
	dest.size = size;
	for( int i = 0; i < size; i++ ) {
		::new( &dest.dataPtr[i] ) CDataHolder( dataPtr[i] );
	}

}

template<class T, class Allocator>
inline void CArray<T, Allocator>::MoveTo( CArray<T, Allocator>& dest )
{
	if( &dest == this ) {
		return;
	}

	dest.FreeBuffer();
	dest.dataPtr = dataPtr;
	dest.bufferSize = bufferSize;
	dest.size = size;
	dataPtr = 0;
	bufferSize = 0;
	size = 0;
}

template<class T, class Allocator>
inline bool CArray<T, Allocator>::operator==( const CArray& other ) const
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

template<class T, class Allocator>
inline bool CArray<T, Allocator>::operator!=( const CArray& other ) const
{
	return !( *this == other );
}

template<class T, class Allocator>
inline int CArray<T, Allocator>::Find( const T& what, int startPos ) const
{
	PresumeFO( startPos >= 0 );
	for( int i = startPos; i < Size(); i++ ) {
		if( what == ( *this )[i] ) {
			return i;
		}
	}
	return NotFound; 
}

template<class T, class Allocator> template<class COMPARE>
inline void CArray<T, Allocator>::QuickSort( COMPARE* param )
{
	FObj::QuickSort<T, COMPARE>( GetPtr(), Size(), param );
}

template<class T, class Allocator> template<class COMPARE>
inline void CArray<T, Allocator>::QuickSort()
{
	FObj::QuickSort<T, COMPARE>( GetPtr(), Size() );
}

template<class T, class Allocator> template<class COMPARE>
inline bool CArray<T, Allocator>::IsSorted( COMPARE* compare ) const
{
	return FObj::IsSorted<T, COMPARE>( GetPtr(), Size(), compare );
}

template<class T, class Allocator> template<class COMPARE>
inline bool CArray<T, Allocator>::IsSorted() const
{
	return FObj::IsSorted<T, COMPARE>( GetPtr(), Size() );
}

template<class T, class Allocator> template<class COMPARE, class SEARCHED_TYPE>
inline int CArray<T, Allocator>::FindInsertionPoint( const SEARCHED_TYPE& what, COMPARE* param ) const
{
	return FObj::FindInsertionPoint<T, COMPARE, SEARCHED_TYPE>( what, GetPtr(), Size(), param );
}

template<class T, class Allocator> template<class COMPARE, class SEARCHED_TYPE>
inline int CArray<T, Allocator>::FindInsertionPoint( const SEARCHED_TYPE& what ) const
{
	return FObj::FindInsertionPoint<T, COMPARE, SEARCHED_TYPE>( what, GetPtr(), Size() );
}

//----------------------------------------------------------------------------------------------

static const int MinBufferGrowSize = 8;

template<class T, class Allocator>
inline void CArray<T, Allocator>::moveData( typename CArray<T, Allocator>::CDataHolder* destDataPtr, int destIndex,
	typename CArray<T, Allocator>::CDataHolder* srcDataPtr, int srcIndex, int count )
{
	if( count > 0 ) {
		if( IsMemmoveable<T>::Value ) {
			ArrayMemMoveBitwize( AddressOfObject( destDataPtr[destIndex].Data ),
				AddressOfObject( srcDataPtr[srcIndex].Data ), count );
		} else {
			ArrayMemMove( AddressOfObject( destDataPtr[destIndex].Data ),
				AddressOfObject( srcDataPtr[srcIndex].Data ), count );
		}
	}
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::growAt( int location, int newSize )
{
	PresumeFO( newSize > size );
	PresumeFO( location <= size );
	if( newSize > bufferSize ) {
		grow( newSize );
	}

	if( location != size ) {
		moveData( dataPtr, location + newSize - size, dataPtr, location, size - location );
	}
	size = newSize;
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::grow( int newSize )
{
	PresumeFO( newSize >= 0 );
	if( newSize > bufferSize ) {
		int delta = max( newSize - bufferSize, max( bufferSize / 2, MinBufferGrowSize ) );
		reallocateBuffer( bufferSize + delta );
	}
}

template<class T, class Allocator>
inline void CArray<T, Allocator>::reallocateBuffer( int newSize )
{
	PresumeFO( newSize > 0 );
	PresumeFO( newSize >= size );
	CDataHolder* oldDataPtr = dataPtr;

	dataPtr = static_cast<CDataHolder*>( ALLOCATE_MEMORY( Allocator, newSize * sizeof( CDataHolder ) ) );
	moveData( dataPtr, 0, oldDataPtr, 0, size );

	if( oldDataPtr != 0 ) {
		Allocator::Free( oldDataPtr );
	}
	bufferSize = newSize;
}

//------------------------------------------------------------------------------------------------------------
// Serialization

template<class T, class Allocator>
inline void CArray<T, Allocator>::Serialize( CArchive& arch )
{
	if( arch.IsLoading() ) {
		DeleteAll();
		unsigned int nElems;
		arch >> nElems;
		check( static_cast<int>( nElems ) >= 0, ERR_BAD_ARCHIVE, arch.Name() );
		SetBufferSize( nElems );
		SetSize( nElems );
		for( int i = 0; i < static_cast<int>( nElems ); i++ ) {
			arch >> ( *this )[i];
		}
	} else {
		arch << static_cast<unsigned int>( Size() );
		for( int i = 0; i < Size(); i++ ) {
			arch << ( *this )[i];
		}
	}
}

template<class T, class Allocator>
inline CArchive& operator>>( CArchive& archive, CArray<T, Allocator>& arr )
{
	arr.Serialize( archive );
	return archive;
}

template<class T, class Allocator>
inline CArchive& operator<<( CArchive& archive, const CArray<T, Allocator>& arr )
{
	const_cast<CArray<T, Allocator>&>( arr ).Serialize( archive );
	return archive;
}

//------------------------------------------------------------------------------------------------------
// Objects array

template<class T, class Allocator = CurrentMemoryManager>
class CObjectArray : public CArray< CPtr<T>, Allocator > {
public:
	CObjectArray() {}
};

//---------------------------------------------------------------------------------------------------
// Specialized ArrayMemMove for the types that may be bitwise moved in memory

template<class T, class Allocator>
inline void ArrayMemMove( CArray<T, Allocator>* dest, CArray<T, Allocator>* source, int count )
{
	ArrayMemMoveBitwize( dest, source, count );
}

inline void ArrayMemMove( double* dest, double* source, int count )
{
	ArrayMemMoveBitwize( dest, source, count );
}

inline void ArrayMemMove( __int64* dest, __int64* source, int count )
{
	ArrayMemMoveBitwize( dest, source, count );
}

} // namespace FineObjects
