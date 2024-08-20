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

#include <ArrayIteratorFOL.h>
#include <ArchiveFOL.h>
#include <ObjectFOL.h>
#include <MathFOL.h>
#include <SortFOL.h>

namespace FObj {

template<class T>
void ArrayMemMoveElement( T* dest, T* source );

// The template function used by CArray to move elements in the buffer
template<class T>
void ArrayMemMove( T* dest, T* source, int count )
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
void ArrayMemMoveElement( T* dest, T* source )
{
	PresumeFO( dest != source );
	::new( dest ) T( std::move( *source ) );
	source->~T();
}

template<class T>
void ArrayMemMoveBitwize( T* dest, T* source, int count )
{
	::memmove( reinterpret_cast<char*>( dest ), reinterpret_cast<char*>( source ), count * sizeof( T ) );
}

//---------------------------------------------------------------------------------------------------------------------

template<class T, class Allocator>
class CArray;

// Specialized IsMemmoveable to indicate that CArray can be moved in memory
template<template<class T, class TAllocator> class TArray, class T, class TAllocator>
struct IsMemmoveable<TArray<T, TAllocator>,
	std::enable_if_t<
		std::is_base_of<CArray<typename TArray<T, TAllocator>::TElement, TAllocator>, TArray<T, TAllocator>>::value &&
		sizeof( CArray<typename TArray<T, TAllocator>::TElement, TAllocator> ) == sizeof( TArray<T, TAllocator> )>
	>
{
	static constexpr bool Value = true;
};

//---------------------------------------------------------------------------------------------------------------------

// Array template. Elements are added with the help of copy constructor
// When elements are deleted their destructors are called
template<class T, class Allocator = CurrentMemoryManager>
class CArray {
	struct CDataHolder {
		struct fromArgsToken {};

		CDataHolder() = default;
		CDataHolder( const T& data ) : Data( data ) {}
		CDataHolder( T&& data ) : Data( std::move( data ) ) {}

		template<class... Args>
		CDataHolder( fromArgsToken, Args&&... args ) : Data( std::forward<Args>( args )... ) {}

		~CDataHolder() = default;
		T Data;
	};

public:
	typedef T TElement;
	typedef Allocator AllocatorType;
	typedef CConstArrayIterator<CArray> TConstIterator;
	typedef CArrayIterator<CArray> TIterator;
	using CompFunc = int ( * )( const T*, const T* );

	CArray();
	CArray( const std::initializer_list<T>& list );
	CArray( CArray&& );
	CArray( const CArray& ) = delete;

	~CArray() { FreeBuffer(); }

	CArray& operator=( const std::initializer_list<T>& list );
	CArray& operator=( CArray&& );
	CArray& operator=( const CArray& ) = delete;

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
	// Returns the pointer to the first element in the array
	T* GetBufferPtr() { return reinterpret_cast<T*>( dataPtr ); }
	const T* GetBufferPtr() const { return reinterpret_cast<T*>( dataPtr ); }
	// Accessing elements by index
	T& operator [] ( int location );
	const T& operator [] ( int location ) const ;
	T& Last();
	const T& Last() const;
	T& First();
	const T& First() const;

	// Checks if an index is valid for this array
	bool IsValidIndex( int index ) const;

	// Sets the buffer size, in terms of elements 
	// You may only increase the size with this method
	void SetBufferSize( int nElements );
	// Grows the buffer size according to the growth policy
	void Grow( int newSize ) { grow( newSize ); }
	// Sets the array size
	void SetSize( int newSize );
	void ShrinkBuffer() { reallocateBuffer( size ); }

	// Adds elements to the end of the array. Copy constructor is used
	void Add( const T& anElem );
	void Add( const T& anElem, int count ) { InsertAt( anElem, Size(), count ); }
	void Add( const CArray& ar ) { InsertAt( ar, Size() ); }
	void Add( const std::initializer_list<T>& list ) { InsertAt( list, Size() ); }
	void Add( T&& anElem );
	void Add( CArray&& ar ) { InsertAt( std::move( ar ), Size() ); }

	template<class... Args>
	void EmplaceBack( Args&&... args );

	template<class... TElements>
	void AddElements( TElements&&... elements );

	// Adds an "empty" element to the end of the array
	T& Append() { SetSize( Size() + 1 ); return Last(); }
	// Inserts an element into the given position in the array (including the last)
	// Copy constructor is used
	void InsertAt( const T& what, int location );
	void InsertAt( const T& what, int location, int count );
	void InsertAt( const CArray& what, int location );
	void InsertAt( const std::initializer_list<T>& list, int location );

	void InsertAt( T&& what, int location );
	void InsertAt( CArray&& what, int location );

	// Replaces an element in the array. The old element is deleted using its destructor
	// The new element is copied into the array using the copy constructor
	void ReplaceAt( const T& newElem, int location );
	void ReplaceAt( T&& newElem, int location );
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

	void MoveElement( int from, int to );

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

	// range-based loop
	TConstIterator begin() const { return TConstIterator( GetPtr(), this ); }
	TConstIterator end() const { return TConstIterator( GetPtr() + Size(), this ); }
	TIterator begin() { return TIterator( GetPtr(), this ); }
	TIterator end() { return TIterator( GetPtr() + Size(), this ); }

private:
	int size = 0;
	int bufferSize = 0;
	CDataHolder* dataPtr = nullptr;

	void growAt( int location, int count );
	void destruct( int begin, int end );
	void grow( int newSize );
	void reallocateBuffer( int newSize );

	template<class... Args>
	auto constructAt( int location, Args&&... args ) -> CDataHolder*;

	template<class Arr, class FuncOp>
	void insertAtImplArray( Arr&& what, int location, FuncOp op );

	template<class El>
	void insertAtImplEl( El&& what, int location );

	template<class... Args>
	void addImplEl( Args&&... args );
	void addImplEl();

	template<class El>
	void replaceAtImplEl( El&& what, int location );

	template<class U>
	using IsMemmoveableWrapperTrue = std::enable_if_t< IsMemmoveable<U>::Value && std::is_same<U, T>::value, void>;
	template<class U>
	using IsMemmoveableWrapperFalse = std::enable_if_t< !IsMemmoveable<U>::Value && std::is_same<U, T>::value, void>;

	void addElementsImpl() {} // stop recursion of bypass variable template parameters

	template<class TOther, class... TElements>
	void addElementsImpl( TOther&& element, TElements&&... elements );

	template<class U = T, class CArrayDataHolder = typename CArray<T, Allocator>::CDataHolder>
	static auto moveData( CArrayDataHolder* destDataPtr, int destIndex,
		CArrayDataHolder* srcDataPtr, int srcIndex, int count ) -> IsMemmoveableWrapperTrue<U>;

	template<class U = T, class CArrayDataHolder = typename CArray<T, Allocator>::CDataHolder>
	static auto moveData( CArrayDataHolder* destDataPtr, int destIndex,
		CArrayDataHolder* srcDataPtr, int srcIndex, int count ) -> IsMemmoveableWrapperFalse<U>;
};

template<class T, class Allocator>
CArray<T, Allocator>::CArray()
{
	static_assert( sizeof( T ) == sizeof( CDataHolder ), "sizeof( T ) != sizeof( CDataHolder )" );
}

template<class T, class Allocator>
CArray<T, Allocator>::CArray( const std::initializer_list<T>& list ) :
	CArray()
{
	Add( list );
}

template<class T, class Allocator>
CArray<T, Allocator>& CArray<T, Allocator>::operator=( const std::initializer_list<T>& list )
{
	DeleteAll();
	Add( list );
	return *this;
}

template<class T, class Allocator>
CArray<T, Allocator>::CArray( CArray&& rhs ) :
	CArray()
{
	FObj::swap( *this, rhs );
}

template<class T, class Allocator>
CArray<T, Allocator>& CArray<T, Allocator>::operator=( CArray&& rhs )
{
	FObj::swap( *this, rhs );
	return *this;
}

template<class T, class Allocator>
int CArray<T, Allocator>::Size() const
{
	return size;
}

template<class T, class Allocator>
int CArray<T, Allocator>::BufferSize() const
{
	return bufferSize;
}

template<class T, class Allocator>
T* CArray<T, Allocator>::GetPtr()
{
	if( Size() == 0 ) {
		return 0;
	}
	return ( T* )dataPtr;
}

template<class T, class Allocator>
const T* CArray<T, Allocator>::GetPtr() const
{
	if( Size() == 0 ) {
		return 0;
	}
	return ( const T* )dataPtr;
}

template<class T, class Allocator>
const T& CArray<T, Allocator>::operator [] ( int location ) const
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	return dataPtr[location].Data;
}

template<class T, class Allocator>
T& CArray<T, Allocator>::operator [] ( int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	return dataPtr[location].Data;
}

template<class T, class Allocator>
const T& CArray<T, Allocator>::Last() const
{
	PresumeFO( size > 0 );
	return dataPtr[size - 1].Data;
}

template<class T, class Allocator>
T& CArray<T, Allocator>::Last()
{
	PresumeFO( size > 0 );
	return dataPtr[size - 1].Data;
}

template<class T, class Allocator>
const T& CArray<T, Allocator>::First() const
{
	PresumeFO( size > 0 );
	return dataPtr[0].Data;
}

template<class T, class Allocator>
T& CArray<T, Allocator>::First()
{
	PresumeFO( size > 0 );
	return dataPtr[0].Data;
}

template<class T, class Allocator>
bool CArray<T, Allocator>::IsValidIndex( int index ) const
{
	return index >= 0 && index < size;
}

template<class T, class Allocator>
void CArray<T, Allocator>::SetBufferSize( int newSize )
{
	PresumeFO( newSize >= 0 );
	if( newSize > bufferSize ) {
		reallocateBuffer( newSize );
	}
}

template<class T, class Allocator>
void CArray<T, Allocator>::SetSize( int newSize )
{
	PresumeFO( newSize >= 0 );
	if( size < newSize ) {
		grow( newSize );
		while( size < newSize ) {
			constructAt( size );
			size++;
		}
	} else if( size > newSize ) {
		DeleteAt( newSize, size - newSize );
	}
}

template<class T, class Allocator>
void CArray<T, Allocator>::Add( const T& what )
{
	addImplEl( what );
}

template<class T, class Allocator>
void CArray<T, Allocator>::Add( T&& what )
{
	addImplEl( std::move( what ) );
}

template<class T, class Allocator>
template<class... TElements>
void CArray<T, Allocator>::AddElements( TElements&&... elements )
{
	grow( Size() + sizeof...( elements ) );
	addElementsImpl( std::forward<TElements>( elements )... );
}

// perfect-forward
template<class T, class Allocator>
void CArray<T, Allocator>::addImplEl()
{
	PresumeFO( size <= bufferSize );
	if( size + 1 > bufferSize ) {
		grow( size + 1 );
	}
	constructAt( size );
	size++;
}

// perfect-forward
template<class T, class Allocator>
template<class... Args>
void CArray<T, Allocator>::addImplEl( Args&&... args )
{
	PresumeFO( size <= bufferSize );
	if( size + 1 > bufferSize ) {
		grow( size + 1 );
	}
	constructAt( size, std::forward<Args>( args )... );
	size++;
}

template<class T, class Allocator>
template<class... Args>
void CArray<T, Allocator>::EmplaceBack( Args&&... args )
{
	using fromArgsToken = typename CDataHolder::fromArgsToken;
	addImplEl( fromArgsToken(), std::forward<Args>( args )... );
}

template<class T, class Allocator>
void CArray<T, Allocator>::InsertAt( const T& what, int location )
{
	insertAtImplEl( what, location );
}

template<class T, class Allocator>
void CArray<T, Allocator>::InsertAt( T&& what, int location )
{
	insertAtImplEl( std::move( what ), location );
}

// perfect-forward
template<class T, class Allocator>
template<class El>
void CArray<T, Allocator>::insertAtImplEl( El&& what, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );
	// The elements from the same array may not be inserted
	PresumeFO( dataPtr == 0 || AddressOfObject( what ) < ( T* )dataPtr || AddressOfObject( what ) >= ( T* )( dataPtr + size ) );

	growAt( location, 1 );
	try {
		constructAt( location, std::forward<El>( what ) );
		size++;
	} catch( ... ) {
		destruct( location + 1, size + 1 );
		size = location;
		throw;
	}
}

template<class T, class Allocator>
void CArray<T, Allocator>::InsertAt( const T& what, int location, int count )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );
	PresumeFO( count >= 0 );
	// The elements from the same array may not be inserted
	PresumeFO( dataPtr == 0 || AddressOfObject( what ) < ( T* )dataPtr || AddressOfObject( what ) >= ( T* )( dataPtr + size ) );

	if( count > 0 ) {
		growAt( location, count );
		int pos = location;
		try {
			for( ; pos < location + count; pos++ ) {
				constructAt( pos, what );
			}
		} catch( ... ) {
			destruct( location + count, size + count );
			size = pos;
			throw;
		}
		size += count;
	}
}

template<class T, class Allocator>
void CArray<T, Allocator>::InsertAt( const CArray& what, int location )
{
	const auto op = [this]( const T& el, int pos ) { constructAt( pos, el ); };
	insertAtImplArray( what, location, op );
}

template<class T, class Allocator>
void CArray<T, Allocator>::InsertAt( CArray&& what, int location )
{
	const auto op = [this]( T& el, int pos ) { constructAt( pos, std::move( el ) ); };
	insertAtImplArray( std::move( what ), location, op );
}

// perfect-forward
template<class T, class Allocator>
template<class Arr, class FuncOp>
void CArray<T, Allocator>::insertAtImplArray( Arr&& what, int location, FuncOp op )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );
	PresumeFO( &what != this );

	const int count = what.Size();
	if( count > 0 ) {
		growAt( location, count );
		int pos = location;
		try {
			for( decltype( auto ) element : what ) {
				op( element, pos );
				pos++;
			}
		} catch( ... ) {
			destruct( location + count, size + count );
			size = pos;
			throw;
		}
		size += count;
	}
}

template<class T, class Allocator>
void CArray<T, Allocator>::InsertAt( const std::initializer_list<T>& list, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );

	const int count = to<int>( list.size() );
	if( count > 0 ) {
		growAt( location, count );
		int pos = location;
		try {
			for( const T& element : list ) {
				constructAt( pos, element );
				++pos;
			}
		} catch( ... ) {
			destruct( location + count, size + count );
			size = pos;
			throw;
		}
		size += count;
	}
}

template<class T, class Allocator>
void CArray<T, Allocator>::ReplaceAt( const T& newElem, int location )
{
	replaceAtImplEl( newElem, location );
}

template<class T, class Allocator>
void CArray<T, Allocator>::ReplaceAt( T&& newElem, int location )
{
	replaceAtImplEl( std::move( newElem ), location );
}

// perfect-forward
template<class T, class Allocator>
template<class El>
void CArray<T, Allocator>::replaceAtImplEl( El&& newElem, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	PresumeFO( AddressOfObject( newElem ) != AddressOfObject( dataPtr[location].Data ) );

	dataPtr[location].~CDataHolder();
	try {
		constructAt( location, std::forward<El>( newElem ) );
	} catch( ... ) {
		destruct( location + 1, size );
		size = location;
		throw;
	}
}

template<class T, class Allocator>
void CArray<T, Allocator>::DeleteAt( int location, int num )
{
	PresumeFO( num >= 0 );
	PresumeFO( num <= size );
	PresumeFO( location >= 0 );
	PresumeFO( location <= size - num );
	if( num == 0 ) {
		return;
	}

	destruct( location, location + num );
	moveData( dataPtr, location, dataPtr, location + num, size - location - num );

	size -= num;
}

template<class T, class Allocator>
void CArray<T, Allocator>::DeleteLast()
{
	PresumeFO( size > 0 );

	destruct( size - 1, size );
	size--;
}

template<class T, class Allocator>
void CArray<T, Allocator>::DeleteAll()
{
	destruct( 0, size );
	size = 0;
}

template<class T, class Allocator>
void CArray<T, Allocator>::FreeBuffer()
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
void CArray<T, Allocator>::MoveElement( int from, int to )
{
	PresumeFO( from >= 0 && from < size );
	PresumeFO( to >= 0 && to < size );
	if( from != to ) {
		CDataHolder tempElement;
		ArrayMemMoveElement( &tempElement, dataPtr + from );
		if( from < to ) {
			moveData( dataPtr, from, dataPtr, from + 1, to - from );
		} else {
			moveData( dataPtr, to + 1, dataPtr, to, from - to );
		}
		ArrayMemMoveElement( dataPtr + to, &tempElement );
	}
}

template<class T, class Allocator>
void CArray<T, Allocator>::CopyTo( CArray& dest ) const
{
	if( &dest == this ) {
		return;
	}

	dest.DeleteAll();
	dest.SetBufferSize( size );

	for( int i = 0; i < size; i++ ) {
		dest.constructAt( i, dataPtr[i] );
		dest.size++;
	}
}

template<class T, class Allocator>
void CArray<T, Allocator>::MoveTo( CArray& dest )
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
bool CArray<T, Allocator>::operator==( const CArray& other ) const
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
bool CArray<T, Allocator>::operator!=( const CArray& other ) const
{
	return !( *this == other );
}

template<class T, class Allocator>
int CArray<T, Allocator>::Find( const T& what, int startPos ) const
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
void CArray<T, Allocator>::QuickSort( COMPARE* param )
{
	FObj::QuickSort<T, COMPARE>( GetPtr(), Size(), param );
}

template<class T, class Allocator> template<class COMPARE>
void CArray<T, Allocator>::QuickSort()
{
	FObj::QuickSort<T, COMPARE>( GetPtr(), Size() );
}

template<class T, class Allocator> template<class COMPARE>
bool CArray<T, Allocator>::IsSorted( COMPARE* compare ) const
{
	return FObj::IsSorted<T, COMPARE>( GetPtr(), Size(), compare );
}

template<class T, class Allocator> template<class COMPARE>
bool CArray<T, Allocator>::IsSorted() const
{
	return FObj::IsSorted<T, COMPARE>( GetPtr(), Size() );
}

template<class T, class Allocator> template<class COMPARE, class SEARCHED_TYPE>
inline int CArray<T, Allocator>::FindInsertionPoint( const SEARCHED_TYPE& what, COMPARE* param ) const
{
	return FObj::FindInsertionPoint<T, COMPARE, SEARCHED_TYPE>( what, GetPtr(), Size(), param );
}

template<class T, class Allocator> template<class COMPARE, class SEARCHED_TYPE>
int CArray<T, Allocator>::FindInsertionPoint( const SEARCHED_TYPE& what ) const
{
	return FObj::FindInsertionPoint<T, COMPARE, SEARCHED_TYPE>( what, GetPtr(), Size() );
}

//----------------------------------------------------------------------------------------------

static const int MinBufferGrowSize = 8;

template<class T, class Allocator>
template<class U, class CArrayDataHolder>
auto CArray<T, Allocator>::moveData( CArrayDataHolder* destDataPtr, int destIndex,
	CArrayDataHolder* srcDataPtr, int srcIndex, int count ) -> IsMemmoveableWrapperTrue<U>
{
	if( count <= 0 ) {
		return;
	}
	ArrayMemMoveBitwize( AddressOfObject( destDataPtr[destIndex].Data ),
		AddressOfObject( srcDataPtr[srcIndex].Data ), count );
}

template<class T, class Allocator>
template<class U, class CArrayDataHolder>
auto CArray<T, Allocator>::moveData( CArrayDataHolder* destDataPtr, int destIndex,
	CArrayDataHolder* srcDataPtr, int srcIndex, int count ) -> IsMemmoveableWrapperFalse<U>
{
	if( count <= 0 ) {
		return;
	}
	ArrayMemMove( AddressOfObject( destDataPtr[destIndex].Data ),
		AddressOfObject( srcDataPtr[srcIndex].Data ), count );
}

template<class T, class Allocator>
void CArray<T, Allocator>::growAt( int location, int count )
{
	PresumeFO( count > 0 );
	PresumeFO( location <= size );
	const int newSize = size + count;
	if( newSize > bufferSize ) {
		grow( newSize );
	}

	if( location != size ) {
		moveData( dataPtr, location + count, dataPtr, location, size - location );
	}
}

template<class T, class Allocator>
void CArray<T, Allocator>::destruct( int begin, int end )
{
	PresumeFO( begin >= 0 );
	for( int index = end - 1; index >= begin; index-- ) {
		dataPtr[index].~CDataHolder();
	}
}

template<class T, class Allocator>
void CArray<T, Allocator>::grow( int newSize )
{
	PresumeFO( newSize >= 0 );
	if( newSize > bufferSize ) {
		int delta = min( max( newSize - bufferSize, max( bufferSize / 2, MinBufferGrowSize ) ), INT_MAX - bufferSize );
		reallocateBuffer( bufferSize + delta );
	}
}

template<class T, class Allocator>
template<class... Args>
auto CArray<T, Allocator>::constructAt( int location, Args&&... args ) -> CDataHolder*
{
	void* addr = ( void* )&dataPtr[location];
	::new( addr ) CDataHolder( std::forward<Args>( args )... );
	return reinterpret_cast<CDataHolder*>( addr );
}

template<class T, class Allocator>
void CArray<T, Allocator>::reallocateBuffer( int newSize )
{
	if( newSize == bufferSize ) {
		return;
	}

	PresumeFO( newSize > 0 );
	PresumeFO( newSize >= size );
	CDataHolder* oldDataPtr = dataPtr;

	AssertFO( static_cast<size_t>( newSize ) <= UINTPTR_MAX / sizeof( CDataHolder ) );
	dataPtr = static_cast<CDataHolder*>( Allocator::Alloc( newSize * sizeof( CDataHolder ) ) );
	moveData( dataPtr, 0, oldDataPtr, 0, size );

	if( oldDataPtr != 0 ) {
		Allocator::Free( oldDataPtr );
	}
	bufferSize = newSize;
}

template<class T, class Allocator>
template<class TOther, class... TElements>
void CArray<T, Allocator>::addElementsImpl( TOther&& element, TElements&&... elements )
{
	EmplaceBack( std::forward<TOther>( element ) );
	addElementsImpl( std::forward<TElements>( elements )... );
}

//---------------------------------------------------------------------------------------------------------------------

// Serialization

template<class T, class Allocator>
void CArray<T, Allocator>::Serialize( CArchive& archive )
{
	unsigned size = static_cast<unsigned>( Size() );
	archive.Serialize( size );

	if( archive.IsLoading() ) {
		check( static_cast<int>( size ) >= 0, ERR_BAD_ARCHIVE, archive.Name() );
		DeleteAll();
		SetSize( size );
	}

	for( auto& item : *this ) {
		archive.Serialize( item );
	}
}

template<class T, class Allocator>
CArchive& operator>>( CArchive& archive, CArray<T, Allocator>& arr )
{
	arr.Serialize( archive );
	return archive;
}

template<class T, class Allocator>
CArchive& operator<<( CArchive& archive, const CArray<T, Allocator>& arr )
{
	const_cast<CArray<T, Allocator>&>( arr ).Serialize( archive );
	return archive;
}

//---------------------------------------------------------------------------------------------------------------------

// Objects array

template<typename T, typename Allocator = CurrentMemoryManager>
using CObjectArray = CArray<CPtr<T>, Allocator>;

//---------------------------------------------------------------------------------------------------------------------

// Specialized ArrayMemMove for the types that may be bitwise moved in memory

template<class T, class Allocator>
void ArrayMemMove( CArray<T, Allocator>* dest, CArray<T, Allocator>* source, int count )
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

//---------------------------------------------------------------------------------------------------------------------

// Cast const CArray<T*>& into const CArray<const T*>&
template<class T, class Allocator>
const CArray<const T*, Allocator>& ToConst( const CArray<T*, Allocator>& arr )
{
	return *( reinterpret_cast<const CArray<const T*, Allocator>*>( &arr ) );
}

// Cast CArray<T*>& into CArray<const T*>&
template<class T, class Allocator>
CArray<const T*, Allocator>& ToConst( CArray<T*, Allocator>& arr )
{
	return *( reinterpret_cast<CArray<const T*, Allocator>*>( &arr ) );
}

// Cast CArray< CPtr<T> >& into CArray< CPtr<const T> >&
template<class T, class Allocator>
CArray< CPtr<const T>, Allocator >& ToConst( CArray< CPtr<T>, Allocator >& arr )
{
	return *( reinterpret_cast<CArray<CPtr<const T>, Allocator>*>( &arr ) );
}

// Cast const CArray< CPtr<T> >& into const CArray< CPtr<const T> >&
template<class T, class Allocator>
const CArray< CPtr<const T>, Allocator >& ToConst( const CArray< CPtr<T>, Allocator >& arr )
{
	return *( reinterpret_cast<const CArray<CPtr<const T>, Allocator>*>( &arr ) );
}

//---------------------------------------------------------------------------------------------------------------------

template<class TSrcArray, class TDstArray>
void CopyTo( const TSrcArray& srcArray, TDstArray& dstArray )
{
	using DstElType = typename TDstArray::TElement;
	PresumeFO( static_cast<const void*>( &srcArray ) != &dstArray );

	dstArray.DeleteAll();
	dstArray.SetBufferSize( srcArray.Size() );
	for( const auto& element : srcArray ) {
		dstArray.Add( static_cast<DstElType>( element ) );
	}
}

template<class TSrcArray, class TDstArray,
	std::enable_if_t< !std::is_same<TSrcArray, TDstArray>::value, void* > = nullptr>
void MoveTo( TSrcArray&& srcArray, TDstArray& dstArray )
{
	using DstElType = typename TDstArray::TElement;
	PresumeFO( static_cast<const void*>( &srcArray ) != &dstArray );

	dstArray.DeleteAll();
	dstArray.SetBufferSize( srcArray.Size() );
	for( auto&& element : srcArray ) {
		dstArray.Add( static_cast<DstElType&&>( std::move( element ) ) );
	}
	srcArray.DeleteAll();
}

template<class TSrcArray, class TDstArray,
	std::enable_if_t< std::is_same<TSrcArray, TDstArray>::value, void*> = nullptr>
void MoveTo( TSrcArray&& srcArray, TDstArray& dstArray )
{
	PresumeFO( static_cast<const void*>( &srcArray ) != &dstArray );
	srcArray.MoveTo( dstArray );
	srcArray.DeleteAll();
}

template<class TSrcArray, class TDstArray,
	std::enable_if_t< std::is_lvalue_reference<TSrcArray>::value, void* > = nullptr>
void AddTo( TSrcArray&& srcArray, TDstArray& dstArray )
{
	using DstElType = typename TDstArray::TElement;
	PresumeFO( static_cast<const void*>( &srcArray ) != &dstArray );
	dstArray.Grow( dstArray.Size() + srcArray.Size() );

	for( const auto& element : srcArray ) {
		dstArray.Add( static_cast<DstElType>( element ) );
	}
}

template<class TSrcArray, class TDstArray,
	std::enable_if_t< std::is_rvalue_reference<TSrcArray&&>::value, void* > = nullptr>
void AddTo( TSrcArray&& srcArray, TDstArray& dstArray )
{
	using DstElType = typename TDstArray::TElement;
	PresumeFO( static_cast<const void*>( &srcArray ) != &dstArray );
	dstArray.Grow( dstArray.Size() + srcArray.Size() );

	for( auto&& element : srcArray ) {
		dstArray.Add( static_cast<DstElType&&>( std::move( element ) ) );
	}
}

template<class TArray1, class TArray2>
bool AreEqual( const TArray1& array1, const TArray2& array2 )
{
	if( array1.Size() != array2.Size() ) {
		return false;
	}
	for( int i = 0; i < array1.Size(); ++i ) {
		if( !( array1[i] == array2[i] ) ) {
			return false;
		}
	}
	return true;
}

template<class TArray>
bool AreEqual( const TArray& array, std::initializer_list<typename TArray::TElement> list )
{
	const int listSize = to<int>( list.size() );
	if( array.Size() != listSize ) {
		return false;
	}
	int i = 0;
	for( const typename TArray::TElement& element : list ) {
		if( !( array[i] == element ) ) {
			return false;
		}
		i++;
	}
	return true;
}

template<class TArray>
bool AreEqual( std::initializer_list<typename TArray::TElement> list, const TArray& array )
{
	return AreEqual( array, list );
}

// creates CArray from variadic elements
template<class T, class... TElements>
CArray<T> MakeCArray( TElements&&... elements )
{
	CArray<T> res;
	res.AddElements( std::forward<TElements>( elements )... );
	return res;
}

} // namespace FObj
