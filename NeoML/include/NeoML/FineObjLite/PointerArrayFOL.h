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

#include <PtrOwnerFOL.h>

namespace FObj {

template<class T, class Allocator, class Deleter>
class CPointerArray;

namespace DetailsCPointerArray {
template<class T>
struct DefaultDeleter {
	void operator() ( T*& value ) 
	{
		delete value;
		value = nullptr;
	}
};
} // namespace DetailsCPointerArray

template<class T, class TAllocator, class TDeleter>
struct IsMemmoveable< CPointerArray<T, TAllocator, TDeleter> > {
	static constexpr bool Value = true;
};

//---------------------------------------------------------------------------------------------------------------------

template <class T, class Allocator = CurrentMemoryManager, class Deleter = DetailsCPointerArray::DefaultDeleter<T> >
class CPointerArray final : private Deleter {
public:
	using TElement = T*;
	using AllocatorType = Allocator;
	using TConstIterator = CConstArrayIterator<CArray<const T*, Allocator>>;
	using TIterator = CConstArrayIterator<CArray<T*, Allocator>>;
	using TDeleter = Deleter;
	using FuncComp = int ( * )( T* const*, T* const* );

	CPointerArray() = default;
	CPointerArray( CPointerArray&& ) = default;
	CPointerArray( const CPointerArray& ) = delete;
	~CPointerArray();

	CPointerArray& operator=( CPointerArray&& ) = default;
	CPointerArray& operator=( const CPointerArray& ) = delete;

	void MoveTo( CPointerArray& other );
	void AppendTo( CPointerArray& other );

	int Size() const;
	int BufferSize() const { return body.BufferSize(); }
	void SetSize( int newSize );
	void SetBufferSize( int newSize );
	void Grow( int newSize );
	void ShrinkBuffer() { body.ShrinkBuffer(); }

	bool IsEmpty() const;
	void Empty() { DeleteAll(); }
	const T* operator [] ( int index ) const;
	T* operator [] ( int index );
	const T* Last() const;
	T* Last();
	const T* First() const;
	T* First();

	bool IsValidIndex( int index ) const { return body.IsValidIndex( index ); }

	const CArray<const T*, Allocator>& GetArrayOfPointers() const;
	const CArray<T*, Allocator>& GetAllPointers() const { return body; }

	void ReplaceAt( T* element, int location );
	void ReplaceAt( CPtrOwner<T>& elementPtr, int location ) = delete;
	void ReplaceAt( CPtrOwner<T>&& elementPtr, int location ) { ReplaceAt( elementPtr.Detach(), location ); }
	void Add( T* element );
	void Add( CPtrOwner<T>& elementPtr ) = delete;
	void Add( CPtrOwner<T>&& elementPtr );
	void InsertAt( T* element, int location );
	void InsertAt( CPtrOwner<T>& elementPtr, int location ) = delete;
	void InsertAt( CPtrOwner<T>&& elementPtr, int location );
	T* DetachAndReplaceAt( T* element, int location );
	T* DetachAndReplaceAt( CPtrOwner<T>& elementPtr, int location ) = delete;
	T* DetachAndReplaceAt( CPtrOwner<T>&& elementPtr, int location )
	{ return DetachAndReplaceAt( elementPtr.Detach(), location ); }

	T* DetachAt( int location );
	void DetachAt( int location, int count );
	bool Detach( const T* element );
	void DetachAll();
	void DeleteAt( int location, int count = 1 );
	bool Delete( T* element );
	void DeleteLast();
	void DeleteAll();
	void FreeBuffer();

	int Find( const T* element, int from = 0 ) const;
	bool Has( const T* what ) const { return Find( what ) != NotFound; }

	void MoveElement( int from, int to );

	template <class COMPARE>
	void QuickSort( COMPARE* param );
	template <class COMPARE>
	void QuickSort();

	template<class COMPARE>
	bool IsSorted( COMPARE* compare ) const;
	template<class COMPARE>
	bool IsSorted() const;

	template<class COMPARE, class SEARCHED_TYPE>
	int FindInsertionPoint( const SEARCHED_TYPE& what, COMPARE* param ) const;
	template<class COMPARE, class SEARCHED_TYPE>
	int FindInsertionPoint( const SEARCHED_TYPE& what ) const;

	void Serialize( CArchive& archive );

	TConstIterator begin() const { return GetArrayOfPointers().begin(); }
	TConstIterator end() const { return GetArrayOfPointers().end(); }
	TIterator begin() { return GetAllPointers().begin(); }
	TIterator end() { return GetAllPointers().end(); }

private:
	CArray<T*, Allocator> body = {};
};

//---------------------------------------------------------------------------------------------------------------------

template<class T, class Allocator, class Deleter>
CPointerArray<T, Allocator, Deleter>::~CPointerArray()
{
	DeleteAll();
}

template<class T, class Allocator, class Deleter>
int CPointerArray<T, Allocator, Deleter>::Size() const
{
	return body.Size();
}

template<class T, class Allocator, class Deleter>
bool CPointerArray<T, Allocator, Deleter>::IsEmpty() const
{
	return Size() == 0;
}

template<class T, class Allocator, class Deleter>
inline const T* CPointerArray<T, Allocator, Deleter>::operator[]( int index ) const
{
	PresumeFO( 0 <= index );
	PresumeFO( index < Size() );
	return body[index];
}

template<class T, class Allocator, class Deleter>
inline T* CPointerArray<T, Allocator, Deleter>::operator[]( int index )
{
	PresumeFO( 0 <= index );
	PresumeFO( index < Size() );
	return body[index];
}

template<class T, class Allocator, class Deleter>
inline const T* CPointerArray<T, Allocator, Deleter>::Last() const
{
	PresumeFO( Size() > 0 );
	return body.Last();
}

template<class T, class Allocator, class Deleter>
inline T* CPointerArray<T, Allocator, Deleter>::Last()
{
	PresumeFO( Size() > 0 );
	return body.Last();
}

template<class T, class Allocator, class Deleter>
inline const T* CPointerArray<T, Allocator, Deleter>::First() const
{
	PresumeFO( Size() > 0 );
	return body.First();
}

template<class T, class Allocator, class Deleter>
inline T* CPointerArray<T, Allocator, Deleter>::First()
{
	PresumeFO( Size() > 0 );
	return body.First();
}

template<class T, class Allocator, class Deleter>
const CArray<const T*, Allocator>& CPointerArray<T, Allocator, Deleter>::GetArrayOfPointers() const
{
	return ToConst( body );
}

template<class T, class Allocator, class Deleter>
inline void CPointerArray<T, Allocator, Deleter>::SetSize( int newSize )
{
	PresumeFO( newSize >= 0 );

	int oldSize = Size();
	if( newSize < oldSize ) {
		DeleteAt( newSize, oldSize - newSize );
	} else {
		body.Add( nullptr, newSize - oldSize );
	}
}

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::SetBufferSize( int newSize )
{
	body.SetBufferSize( newSize );
}

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::Grow( int newSize )
{
	body.Grow( newSize );
}

template<class T, class Allocator, class Deleter>
inline void CPointerArray<T, Allocator, Deleter>::ReplaceAt( T* obj, int location )
{
	T* current = body[location];
	if( current != obj ) {
		body[location] = obj;
		Deleter::operator() ( current );
	}
}

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::Add( T* obj )
{
	body.Add( obj );
}

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::Add( CPtrOwner<T>&& objPtr )
{
	body.Grow( body.Size() + 1 );
	body.Add( objPtr.Detach() );
}

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::InsertAt( T* obj, int location )
{
	body.InsertAt( obj, location );
}

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::InsertAt( CPtrOwner<T>&& objPtr, int location )
{
	body.Grow( body.Size() + 1 );
	body.InsertAt( objPtr.Detach(), location );
}

template<class T, class Allocator, class Deleter>
inline T* CPointerArray<T, Allocator, Deleter>::DetachAndReplaceAt( T* obj, int location )
{
	T* ret = body[location];
	body[location] = obj;
	return ret;
}

template<class T, class Allocator, class Deleter>
inline T* CPointerArray<T, Allocator, Deleter>::DetachAt( int location )
{
	T* ret = body[location];
	body.DeleteAt( location );
	return ret;
}

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::DetachAt( int location, int num )
{
	body.DeleteAt( location, num );
}

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::DetachAll()
{ 
	body.DeleteAll();
}

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::DeleteAll()
{
	DeleteAt( 0, Size() );
}

template<class T, class Allocator, class Deleter>
inline void CPointerArray<T, Allocator, Deleter>::FreeBuffer()
{
	DeleteAll();
	body.FreeBuffer();
}

template<class T, class Allocator, class Deleter>
inline void CPointerArray<T, Allocator, Deleter>::MoveTo( CPointerArray& other )
{
	if( &other == this ) {
		return;
	}

	other.DeleteAll();
	body.MoveTo( other.body );
}

template<class T, class Allocator, class Deleter>
inline void CPointerArray<T, Allocator, Deleter>::AppendTo( CPointerArray& other )
{
	AssertFO( &other != this );

	other.body.Add( body );
	body.DeleteAll();
}

template<class T, class Allocator, class Deleter> template <class COMPARE>
void CPointerArray<T, Allocator, Deleter>::QuickSort( COMPARE* param )
{
	body.QuickSort( param );
}

template<class T, class Allocator, class Deleter> template <class COMPARE>
void CPointerArray<T, Allocator, Deleter>::QuickSort()
{
	body.template QuickSort<COMPARE>();
}

template<class T, class Allocator, class Deleter> template<class COMPARE>
bool CPointerArray<T, Allocator, Deleter>::IsSorted( COMPARE* compare ) const
{
	return body.IsSorted( compare );
}

template<class T, class Allocator, class Deleter> template<class COMPARE>
bool CPointerArray<T, Allocator, Deleter>::IsSorted() const
{
	return body.template IsSorted<COMPARE>();
}

template<class T, class Allocator, class Deleter> template<class COMPARE, class SEARCHED_TYPE>
int CPointerArray<T, Allocator, Deleter>::FindInsertionPoint( const SEARCHED_TYPE& what, COMPARE* param ) const
{
	return body.FindInsertionPoint( what, param );
}

template<class T, class Allocator, class Deleter> template<class COMPARE, class SEARCHED_TYPE>
int CPointerArray<T, Allocator, Deleter>::FindInsertionPoint( const SEARCHED_TYPE& what ) const
{
	return body.template FindInsertionPoint<COMPARE>( what );
}

template<class T, class Allocator, class Deleter>
inline int CPointerArray<T, Allocator, Deleter>::Find( const T* obj, int start ) const
{
	AssertFO( start >= 0 );
	AssertFO( start <= Size() );

	for( int pos = start; pos < Size(); pos++ ) {
		if( body[pos] == obj ) {
			return pos;
		}
	}
	return NotFound;
}

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::MoveElement( int from, int to )
{
	body.MoveElement( from, to );
}

template<class T, class Allocator, class Deleter>
bool CPointerArray<T, Allocator, Deleter>::Detach( const T* obj )
{
	int pos = Find( obj );
	if( pos != NotFound ) {
		DetachAt( pos );
		return true;
	} else {
		return false;
	}
}

template<class T, class Allocator, class Deleter>
bool CPointerArray<T, Allocator, Deleter>::Delete( T* obj )
{
	int pos = Find( obj );
	if( pos != NotFound ) {
	 	DeleteAt( pos );
		return true;
	} else {
		return false;
	}
}

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::DeleteAt( int location, int num )
{
	AssertFO( location >= 0 );
	AssertFO( location <= Size() );
	AssertFO( num >= 0 );
	AssertFO( location <= Size() - num );

	if( num == 0 ) {
		return;
	}
	for( int pos = location; pos < location + num; pos++ ) {
		T* current = body[pos];
		body[pos] = nullptr;
		Deleter::operator() ( current );
	}
	body.DeleteAt( location, num );
}

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::DeleteLast()
{
	T* current = body.Last();
	body.DeleteLast();
	Deleter::operator() ( current );
}

//---------------------------------------------------------------------------------------------------------------------

template<class T, class Allocator, class Deleter>
void CPointerArray<T, Allocator, Deleter>::Serialize( CArchive& ar )
{
	if( ar.IsStoring() ) {
		ar << static_cast<unsigned int>( Size() );
		for( int i = 0; i < Size(); i++ ) {
			if( ( *this )[i] != nullptr ) {
				ar << static_cast<unsigned int>( i );
				ar << *( *this )[i];
			}
		}
		ar << static_cast<unsigned int>( Size() );
	} else {
		DeleteAll();
		unsigned int size;
		ar >> size;
		SetBufferSize( size );
		while( true ) {
			unsigned int index;
			ar >> index;
			SetSize( static_cast<int>( index ) );
			if( index == size ) {
				break;
			}
			T* obj = FINE_DEBUG_NEW T;
			Add( obj );
			ar >> *obj;
		}
	}
}

//---------------------------------------------------------------------------------------------------------------------

template<class T, class Allocator, class Deleter>
inline void ArrayMemMoveElement( CPointerArray<T, Allocator, Deleter>* dest, CPointerArray<T, Allocator, Deleter>* src )
{
	PresumeFO( dest != src );
	::new( dest ) CPointerArray<T, Allocator, Deleter>;
	src->MoveTo( *dest );
	src->~CPointerArray<T, Allocator, Deleter>();
}

} // namespace FObj
