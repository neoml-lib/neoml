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

#include <ArchiveFOL.h>
#include <ArrayIteratorFOL.h>
#include <SortFOL.h>

namespace FObj {

template<class T, int initBufSize, class Allocator>
class CFastArray;

template<class T, int initBufSize, class Allocator>
struct IsMemmoveable<CFastArray<T, initBufSize, Allocator>> {
	static constexpr bool Value = false;
};

//---------------------------------------------------------------------------------------------------------------------

template<class T, int initialBufferSize, class Allocator = CurrentMemoryManager>
class CFastArray {
	template<int rhsInitBufSize>
	using TFastArray = CFastArray<T, rhsInitBufSize, Allocator>;
public:
	typedef T TElement;
	typedef Allocator AllocatorType;
	typedef CConstArrayIterator<CFastArray> TConstIterator;
	typedef CArrayIterator<CFastArray> TIterator;
	using CompFunc = int ( * )( const T*, const T* );

	CFastArray();
	CFastArray( const std::initializer_list<T>& list );
	template<int rhsInitBufSize>
	CFastArray( TFastArray<rhsInitBufSize>&& );
	~CFastArray();

	CFastArray( const CFastArray& ) = delete;
	template<int rhsInitBufSize>
	CFastArray& operator=( const TFastArray<rhsInitBufSize>& ) = delete;

	template<int rhsInitBufSize>
	CFastArray& operator=( TFastArray<rhsInitBufSize>&& );
	CFastArray& operator=( const std::initializer_list<T>& list );

	int Size() const;
	int BufferSize() const;
	bool IsEmpty() const { return Size() == 0; }
	void Empty() { DeleteAll(); }
	static constexpr int InitialBufferSize();
	void SetSize( int newSize );
	void SetBufferSize( int nElem );
	void Grow( int newSize ) { grow( newSize ); }
	void ShrinkBuffer() { reallocateBuffer( size ); }

	void Add( const T& elem );
	void Add( const T& elem, int count );
	template<int rhsInitBufSize>
	void Add( const TFastArray<rhsInitBufSize>& );
	void Add( const std::initializer_list<T>& list );
	void Add( T&& elem );
	template<int rhsInitBufSize>
	void Add( TFastArray<rhsInitBufSize>&& );
	auto Append() -> T& { SetSize( Size() + 1 ); return Last(); }

	template<class... Args>
	void EmplaceBack( Args&&... args );

	auto operator[]( int location ) -> T&;
	auto operator[]( int location ) const -> const T&;
	auto Last() -> T&;
	auto Last() const -> const T&;
	auto First() -> T&;
	auto First() const -> const T&;

	bool IsValidIndex( int index ) const;

	auto GetPtr() -> T*;
	auto GetPtr() const -> const T*;
	auto GetBufferPtr() -> T* { return dataPtr; }
	auto GetBufferPtr() const -> const T* { return dataPtr; }

	void ReplaceAt( const T& elem, int location );
	void ReplaceAt( T&& elem, int location );
	void InsertAt( const T& elem, int location );
	void InsertAt( T&& elem, int location );
	void InsertAt( const T& elem, int location, int count );
	template<int rhsInitBufSize>
	void InsertAt( const TFastArray<rhsInitBufSize>& what, int location );
	template<int rhsInitBufSize>
	void InsertAt( TFastArray<rhsInitBufSize>&& what, int location );
	void InsertAt( const std::initializer_list<T>& list, int location );
	void DeleteAt( int location, int count );
	void DeleteAt( int location );
	void DeleteLast();
	void DeleteAll();
	void FreeBuffer();

	void MoveElement( int from, int to );

	template<int rhsInitBufSize>
	void CopyTo( TFastArray<rhsInitBufSize>& dest ) const;
	template<int rhsInitBufSize>
	void MoveTo( TFastArray<rhsInitBufSize>& dest );

	template<int rhsInitBufSize>
	bool operator==( const TFastArray<rhsInitBufSize>& other ) const;
	template<int rhsInitBufSize>
	bool operator!=( const TFastArray<rhsInitBufSize>& other ) const;

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

	// range-based loop
	auto begin() const -> TConstIterator { return TConstIterator( GetPtr(), this ); }
	auto end() const -> TConstIterator { return TConstIterator( GetPtr() + Size(), this ); }
	auto begin() -> TIterator { return TIterator( GetPtr(), this ); }
	auto end() -> TIterator { return TIterator( GetPtr() + Size(), this ); }

private:
	T* dataPtr = nullptr;
	BYTE buffer[initialBufferSize * sizeof( T )];
	int size = 0;
	int bufferSize = initialBufferSize;

	template<class Arg1, class Arg2>
	static bool addrsEq( const Arg1&, const Arg2& );

	template<class Arg>
	void addImplEl( Arg&& arg );
	template<class Arg, class Func>
	void addImplArr( Arg&& arg, Func func );

	template<class Arg>
	void replaceAtImplEl( Arg&& arg, int location );

	template<class Arg>
	void insertAtImplEl( Arg&& arg, int location );
	template<class Arg, class Func>
	void insertAtImplArr( Arg&& arg, int location, Func func );

	void growAt( int pos, int newSize );
	void grow( int newSize );
	void reallocateBuffer( int newSize );

	template<class OtherT, int otherInitBufSize, class OtherAlloc>
	friend class CFastArray;
};

//---------------------------------------------------------------------------------------------------------------------

template<class T, int initialBufferSize, class Allocator>
CFastArray<T, initialBufferSize, Allocator>::CFastArray() :
	size( 0 ),
	bufferSize( initialBufferSize )
{
	static_assert( initialBufferSize >= 0, "" );
	dataPtr = reinterpret_cast<T*>( buffer );
}

template<class T, int initialBufferSize, class Allocator>
CFastArray<T, initialBufferSize, Allocator>::CFastArray( const std::initializer_list<T>& list ) :
	CFastArray()
{
	Add( list );
}

template<class T, int initialBufferSize, class Allocator>
template<int rhsInitBufSize>
CFastArray<T, initialBufferSize, Allocator>::CFastArray( TFastArray<rhsInitBufSize>&& other ) :
	CFastArray()
{
	*this = std::move( other );
}

template<class T, int initialBufferSize, class Allocator>
CFastArray<T, initialBufferSize, Allocator>::~CFastArray()
{
	if( dataPtr != ( T* ) buffer ) {
		Allocator::Free( dataPtr );
	}
}

template<class T, int initialBufferSize, class Allocator>
auto CFastArray<T, initialBufferSize, Allocator>::operator=( const std::initializer_list<T>& list ) -> CFastArray&
{
	DeleteAll();
	Add( list );
	return *this;
}

template<class T, int initialBufferSize, class Allocator>
template<int rhsInitBufSize>
auto CFastArray<T, initialBufferSize, Allocator>::operator=( TFastArray<rhsInitBufSize>&& other ) -> CFastArray&
{
	other.MoveTo( *this );
	return *this;
}

template<class T, int initialBufferSize, class Allocator>
int CFastArray<T, initialBufferSize, Allocator>::Size() const
{
	return size;
}

template<class T, int initialBufferSize, class Allocator>
int CFastArray<T, initialBufferSize, Allocator>::BufferSize() const
{
	return bufferSize;
}

template<class T, int initialBufferSize, class Allocator>
constexpr int CFastArray<T, initialBufferSize, Allocator>::InitialBufferSize()
{
	return initialBufferSize;
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::SetSize( int newSize )
{
	PresumeFO( newSize >= 0 );
	if( newSize > bufferSize ) {
		grow( newSize );
	}
	size = newSize;
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::grow( int newSize )
{
	PresumeFO( newSize >= 0 );
	if( newSize > bufferSize ) {
		int delta = min( max( newSize - bufferSize, max( bufferSize / 2, initialBufferSize ) ), INT_MAX - bufferSize );
		reallocateBuffer( bufferSize + delta );
	}
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::SetBufferSize( int newSize )
{
	PresumeFO( newSize >= 0 );
	if( newSize > bufferSize ) {
		reallocateBuffer( newSize );
	}
}

template<class T, int initialBufferSize, class Allocator>
auto CFastArray<T, initialBufferSize, Allocator>::GetPtr() -> T*
{
	if( Size() == 0 ) {
		return 0;
	}
	return dataPtr;
}

template<class T, int initialBufferSize, class Allocator>
auto CFastArray<T, initialBufferSize, Allocator>::GetPtr() const -> const T*
{
	if( Size() == 0 ) {
		return 0;
	}
	return dataPtr;
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::Add( const T& elem )
{
	addImplEl( elem );
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::Add( T&& elem )
{
	addImplEl( std::move( elem ) );
}

template<class T, int initialBufferSize, class Allocator>
template<class Arg>
void CFastArray<T, initialBufferSize, Allocator>::addImplEl( Arg&& arg )
{
	PresumeFO( AddressOfObject( arg ) < dataPtr || AddressOfObject( arg ) >= dataPtr + size );
	SetSize( size + 1 );
	dataPtr[size - 1] = std::forward<Arg>( arg );
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::Add( const T& elem, int count )
{
	InsertAt( elem, Size(), count );
}

template<class T, int initialBufferSize, class Allocator>
template<int rhsInitBufSize>
void CFastArray<T, initialBufferSize, Allocator>::Add( const TFastArray<rhsInitBufSize>& ar )
{
	addImplArr( ar, &::memcpy );
}

template<class T, int initialBufferSize, class Allocator>
template<int rhsInitBufSize>
void CFastArray<T, initialBufferSize, Allocator>::Add( TFastArray<rhsInitBufSize>&& ar )
{
	addImplArr( std::move( ar ), &::memmove );
}

template<class T, int initialBufferSize, class Allocator>
template<class Arg, class Func>
void CFastArray<T, initialBufferSize, Allocator>::addImplArr( Arg&& arg, Func func )
{
	insertAtImplArr( std::forward<Arg>( arg ), size, func );
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::Add( const std::initializer_list<T>& list )
{
	InsertAt( list, Size() );
}

template<class T, int initialBufferSize, class Allocator>
template<class... Args>
void CFastArray<T, initialBufferSize, Allocator>::EmplaceBack( Args&&... args )
{
	T tmp{ std::forward<Args>( args )... };
	Add( std::move( tmp ) );
}

template<class T, int initialBufferSize, class Allocator>
auto CFastArray<T, initialBufferSize, Allocator>::operator [] ( int location ) const -> const T&
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	return dataPtr[location];
}

template<class T, int initialBufferSize, class Allocator>
auto CFastArray<T, initialBufferSize, Allocator>::operator [] ( int location ) -> T&
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	return dataPtr[location];
}

template<class T, int initialBufferSize, class Allocator>
auto CFastArray<T, initialBufferSize, Allocator>::Last() const -> const T&
{
	PresumeFO( size > 0 );
	return dataPtr[size - 1];
}

template<class T, int initialBufferSize, class Allocator>
auto CFastArray<T, initialBufferSize, Allocator>::Last() -> T&
{
	PresumeFO( size > 0 );
	return dataPtr[size - 1];
}

template<class T, int initialBufferSize, class Allocator>
auto CFastArray<T, initialBufferSize, Allocator>::First() const -> const T&
{
	PresumeFO( size > 0 );
	return dataPtr[0];
}

template<class T, int initialBufferSize, class Allocator>
auto CFastArray<T, initialBufferSize, Allocator>::First() -> T&
{
	PresumeFO( size > 0 );
	return dataPtr[0];
}

template<class T, int initialBufferSize, class Allocator>
bool CFastArray<T, initialBufferSize, Allocator>::IsValidIndex( int index ) const
{
	return index >= 0 && index < size;
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::ReplaceAt( const T& elem, int location )
{
	replaceAtImplEl( elem, location );
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::ReplaceAt( T&& elem, int location )
{
	replaceAtImplEl( std::move( elem ), location );
}

template<class T, int initialBufferSize, class Allocator>
template<class Arg>
void CFastArray<T, initialBufferSize, Allocator>::replaceAtImplEl( Arg&& arg, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	dataPtr[location] = std::forward<Arg>( arg );
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::growAt( int pos, int newSize )
{
	PresumeFO( newSize > size );

	int delta = newSize - size;
	SetSize( newSize );
	if( size != pos + delta ) {
		::memmove( reinterpret_cast<char*>( dataPtr + pos + delta ),
			reinterpret_cast<char*>( dataPtr + pos ),
			( size - pos - delta ) * sizeof( T ) );
	}
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::InsertAt( const T& elem, int location )
{
	insertAtImplEl( elem, location );
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::InsertAt( T&& elem, int location )
{
	insertAtImplEl( std::move( elem ), location );
}

template<class T, int initialBufferSize, class Allocator>
template<class Arg>
void CFastArray<T, initialBufferSize, Allocator>::insertAtImplEl( Arg&& arg, int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );
	PresumeFO( AddressOfObject( arg ) < dataPtr || AddressOfObject( arg ) >= dataPtr + size );

	growAt( location, size + 1 );
	dataPtr[location] = std::forward<Arg>( arg );
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::InsertAt( const T& elem, int location, int count )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );
	PresumeFO( count >= 0 );
	PresumeFO( AddressOfObject( elem ) < dataPtr || AddressOfObject( elem ) >= dataPtr + size );

	if( count > 0 ) {
		growAt( location, size + count );
		for( int i = location; i < location + count; i++ ) {
			dataPtr[i] = elem;
		}
	}
}

template<class T, int initialBufferSize, class Allocator>
template<int rhsInitBufSize>
void CFastArray<T, initialBufferSize, Allocator>::InsertAt( const TFastArray<rhsInitBufSize>& ar, int location )
{
	insertAtImplArr( ar, location, &::memcpy );
}

template<class T, int initialBufferSize, class Allocator>
template<int rhsInitBufSize>
void CFastArray<T, initialBufferSize, Allocator>::InsertAt( TFastArray<rhsInitBufSize>&& ar, int location )
{
	insertAtImplArr( std::move( ar ), location, &::memmove );
}

template<class T, int initialBufferSize, class Allocator>
template<class Arg, class Func>
void CFastArray<T, initialBufferSize, Allocator>::insertAtImplArr( Arg&& arg, int location, Func func )
{
	PresumeFO( location >= 0 );
	PresumeFO( location <= size );
	PresumeFO( !addrsEq( arg, *this ) );

	if( arg.Size() > 0 ) {
		growAt( location, size + arg.Size() );
		func( reinterpret_cast<char*>( dataPtr + location ),
			reinterpret_cast<const char*>( arg.GetPtr() ),
			arg.Size() * sizeof( T ) );
	}
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::InsertAt( const std::initializer_list<T>& list, int location )
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
void CFastArray<T, initialBufferSize, Allocator>::DeleteAt( int location )
{
	PresumeFO( location >= 0 );
	PresumeFO( location < size );
	if( size != location + 1 ) {
		::memmove( reinterpret_cast<char*>( dataPtr + location ),
			reinterpret_cast<const char*>( dataPtr + location + 1 ),
			( size - location - 1 ) * sizeof( T ) );
	}
	--size;
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::DeleteAt( int location, int count )
{
	PresumeFO( count >= 0 );
	PresumeFO( location >= 0 );
	PresumeFO( location + count <= size );
	if( count > 0 ) {
		if( size != location + count ) {
			::memmove( reinterpret_cast<char*>( dataPtr + location ),
				reinterpret_cast<char*>( dataPtr + location + count ),
				( size - location - count ) * sizeof( T ) );
		}
		size -= count;
	}
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::DeleteLast()
{
	PresumeFO( size > 0 );
	size--;
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::DeleteAll()
{
	size = 0;
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::FreeBuffer()
{
	DeleteAll();
	reallocateBuffer( 0 );
}

template<class T, int initialBufferSize, class Allocator> template <class COMPARE>
void CFastArray<T, initialBufferSize, Allocator>::QuickSort( COMPARE* param )
{
	FObj::QuickSort<T, COMPARE>( GetPtr(), Size(), param );
}

template<class T, int initialBufferSize, class Allocator> template <class COMPARE>
void CFastArray<T, initialBufferSize, Allocator>::QuickSort()
{
	FObj::QuickSort<T, COMPARE>( GetPtr(), Size() );
}

template<class T, int initialBufferSize, class Allocator> template <class COMPARE>
bool CFastArray<T, initialBufferSize, Allocator>::IsSorted( COMPARE* compare ) const
{
	return FObj::IsSorted<T, COMPARE>( GetPtr(), Size(), compare );
}

template<class T, int initialBufferSize, class Allocator> template <class COMPARE>
bool CFastArray<T, initialBufferSize, Allocator>::IsSorted() const
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
int CFastArray<T, initialBufferSize, Allocator>::FindInsertionPoint( const SEARCHED_TYPE& what ) const
{
	return FObj::FindInsertionPoint<T, COMPARE, SEARCHED_TYPE>( what, GetPtr(), Size() );
}

//---------------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------------

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::reallocateBuffer( int newSize )
{
	if( newSize == bufferSize ) {
		return;
	}

	PresumeFO( newSize >= size );

	if( newSize > InitialBufferSize() ) {
		T* oldDataPtr = dataPtr;
		AssertFO( static_cast<size_t>( newSize ) <= UINTPTR_MAX / sizeof( T ) );
		dataPtr = static_cast<T*>( Allocator::Alloc( newSize * sizeof( T ) ) );
		if( size > 0 ) {
			::memmove( reinterpret_cast<char*>( dataPtr ), reinterpret_cast<const char*>( oldDataPtr ), size * sizeof( T ) );
		}
		if( oldDataPtr != ( T* ) buffer ) {
			Allocator::Free( oldDataPtr );
		}
		bufferSize = newSize;
	} else if( dataPtr != ( T* ) buffer ) {
		if( size > 0 ) {
			::memmove( reinterpret_cast<char*>( buffer ), reinterpret_cast<const char*>( dataPtr ), size * sizeof( T ) );
		}
		Allocator::Free( dataPtr );
		dataPtr = ( T* ) buffer;
		bufferSize = InitialBufferSize();
	}
}

template<class T, int initialBufferSize, class Allocator>
void CFastArray<T, initialBufferSize, Allocator>::MoveElement( int from, int to )
{
	PresumeFO( from >= 0 && from < size );
	PresumeFO( to >= 0 && to < size );
	if( from != to ) {
		alignas( T ) BYTE tmp[sizeof( T )];
		ArrayMemMoveElement( reinterpret_cast<T*>( tmp ), dataPtr + from ); // Uses function from Array.h
		if( from < to ) {
			ArrayMemMove( dataPtr + from, dataPtr + from + 1, to - from );
		} else {
			ArrayMemMove( dataPtr + to + 1, dataPtr + to, from - to );
		}
		ArrayMemMoveElement( dataPtr + to, reinterpret_cast<T*>( tmp ) ); // Uses function from Array.h
	}
}

template<class T, int initialBufferSize, class Allocator>
template<int rhsInitBufSize>
void CFastArray<T, initialBufferSize, Allocator>::CopyTo( TFastArray<rhsInitBufSize>& dest ) const
{
	if( addrsEq( dest, *this ) ) {
		return;
	}

	dest.DeleteAll();
	dest.SetBufferSize( size );

	dest.size = size;
	::memcpy( reinterpret_cast<char*>( dest.dataPtr ), reinterpret_cast<const char*>( dataPtr ), size * sizeof( T ) );
}

template<class T, int initialBufferSize, class Allocator>
template<int destInitBufSize>
void CFastArray<T, initialBufferSize, Allocator>::MoveTo( CFastArray<T, destInitBufSize, Allocator>& dest )
{
	if( addrsEq( dest, *this ) ) {
		return;
	}

	const int bytesSize = size * sizeof( T );
	const int rhsInitBufSize = dest.InitialBufferSize() * sizeof( T );
	dest.DeleteAll();
	if( dataPtr != ( T* ) buffer &&
		bytesSize > rhsInitBufSize )
	{
		dest.FreeBuffer();
		dest.dataPtr = dataPtr;
		dest.bufferSize = bufferSize;
		dest.size = size;
		dataPtr = ( T* ) buffer;
	} else {
		dest.reallocateBuffer( size );
		::memmove( reinterpret_cast<char*>( dest.dataPtr ), reinterpret_cast<const char*>( dataPtr ), bytesSize );
		dest.size = size;
	}

	size = 0;
	bufferSize = initialBufferSize;
}

template<class T, int initialBufferSize, class Allocator>
template<int rhsInitBufSize>
bool CFastArray<T, initialBufferSize, Allocator>::operator==( const TFastArray<rhsInitBufSize>& other ) const
{
	if( addrsEq( *this, other ) ) {
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
template<int rhsInitBufSize>
bool CFastArray<T, initialBufferSize, Allocator>::operator!=( const TFastArray<rhsInitBufSize>& other ) const
{
	return !( *this == other );
}

template<class T, int initialBufferSize, class Allocator>
int CFastArray<T, initialBufferSize, Allocator>::Find( const T& what, int startPos ) const
{
	PresumeFO( startPos >= 0 );
	for( int i = startPos; i < Size(); i++ ) {
		if( what == ( *this )[i] ) {
			return i;
		}
	}
	return NotFound;
}

template<class T, int initialBufferSize, class Allocator>
template<class Arg1, class Arg2>
bool CFastArray<T, initialBufferSize, Allocator>::addrsEq( const Arg1& arg1, const Arg2& arg2 )
{
	return reinterpret_cast<const void*>( &arg1 ) == reinterpret_cast<const void*>( &arg2 );
}

//---------------------------------------------------------------------------------------------------------------------

template<class T, int InitialBufferSize, class Allocator,
	class TFastArray = CFastArray<T, InitialBufferSize, Allocator>>
void ArrayMemMoveElement( TFastArray* dest, TFastArray* source )
{
	PresumeFO( dest != source );
	::new( dest ) TFastArray;
	source->MoveTo( *dest );
	source->~TFastArray();
}

//---------------------------------------------------------------------------------------------------------------------

template<class T, int initialBufferSize, class Allocator>
auto ToConst( const CFastArray<T*, initialBufferSize, Allocator>& arr ) -> const CFastArray<const T*, initialBufferSize, Allocator>&
{
	return *( reinterpret_cast< const CFastArray<const T*, initialBufferSize, Allocator>* >( &arr ) );
}

} // namespace FObj
