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

template<class T, class Allocator = CurrentMemoryManager>
class CPointerArray {
public:
	typedef T* TElement;
	typedef Allocator AllocatorType;

	CPointerArray();
	~CPointerArray();

	void MoveTo( CPointerArray& other );
	void AppendTo( CPointerArray& other );

	int Size() const;
	int BufferSize() const { return body.BufferSize(); }
	void SetSize( int newSize );
	void SetBufferSize( int newSize );
	void Grow( int newSize );

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
	void Add( T* element );
	void InsertAt( T* element, int location );
	T* DetachAndReplaceAt( T* element, int location );
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

private:
	CArray<T*, Allocator> body;

	CPointerArray( const CPointerArray& );
	CPointerArray& operator=( const CPointerArray& );
};

template<class T, class Allocator>
inline CPointerArray<T, Allocator>::CPointerArray()
{
}

template<class T, class Allocator>
inline CPointerArray<T, Allocator>::~CPointerArray()
{
	DeleteAll();
}

template<class T, class Allocator>
inline int CPointerArray<T, Allocator>::Size() const
{
	return body.Size();
}

template<class T, class Allocator>
inline bool CPointerArray<T, Allocator>::IsEmpty() const
{
	return Size() == 0;
}

template<class T, class Allocator>
inline const T* CPointerArray<T, Allocator>::operator[]( int index ) const
{
	PresumeFO( 0 <= index );
	PresumeFO( index < Size() );
	return body[index];
}

template<class T, class Allocator>
inline T* CPointerArray<T, Allocator>::operator[]( int index )
{
	PresumeFO( 0 <= index );
	PresumeFO( index < Size() );
	return body[index];
}

template<class T, class Allocator>
inline const T* CPointerArray<T, Allocator>::Last() const
{
	PresumeFO( Size() > 0 );
	return body.Last();
}

template<class T, class Allocator>
inline T* CPointerArray<T, Allocator>::Last()
{
	PresumeFO( Size() > 0 );
	return body.Last();
}

template<class T, class Allocator>
inline const T* CPointerArray<T, Allocator>::First() const
{
	PresumeFO( Size() > 0 );
	return body.First();
}

template<class T, class Allocator>
inline T* CPointerArray<T, Allocator>::First()
{
	PresumeFO( Size() > 0 );
	return body.First();
}

template<class T, class Allocator>
const CArray<const T*, Allocator>& CPointerArray<T, Allocator>::GetArrayOfPointers() const
{
	return ToConst( body );
}

template<class T, class Allocator>
inline void CPointerArray<T, Allocator>::SetSize( int newSize )
{
	PresumeFO( newSize >= 0 );

	int oldSize = Size();
	if( newSize < oldSize ) {
		DeleteAt( newSize, oldSize - newSize );
	} else {
		body.Add( 0, newSize - oldSize );
	}
}

template<class T, class Allocator>
inline void CPointerArray<T, Allocator>::SetBufferSize( int newSize )
{
	body.SetBufferSize( newSize );
}

template<class T, class Allocator>
inline void CPointerArray<T, Allocator>::Grow( int newSize )
{
	body.Grow( newSize );
}

template<class T, class Allocator>
inline void CPointerArray<T, Allocator>::ReplaceAt( T* obj, int location )
{
	T* current = body[location];
	if( current != obj ) {
		body[location] = obj;
		delete current;
	}
}

template<class T, class Allocator>
inline void CPointerArray<T, Allocator>::Add( T* obj )
{
	body.Add( obj );
}

template<class T, class Allocator>
inline void CPointerArray<T, Allocator>::InsertAt( T* obj, int location )
{
	body.InsertAt( obj, location );
}

template<class T, class Allocator>
inline T* CPointerArray<T, Allocator>::DetachAndReplaceAt( T* obj, int location )
{
	T* ret = body[location];
	body[location] = obj;
	return ret;
}

template<class T, class Allocator>
inline T* CPointerArray<T, Allocator>::DetachAt( int location )
{
	T* ret = body[location];
	body.DeleteAt( location );
	return ret;
}

template<class T, class Allocator>
inline void CPointerArray<T, Allocator>::DetachAt( int location, int num )
{
	body.DeleteAt( location, num );
}

template<class T, class Allocator>
inline void CPointerArray<T, Allocator>::DetachAll()
{ 
	body.DeleteAll();
}

template<class T, class Allocator>
inline void CPointerArray<T, Allocator>::DeleteAll()
{
	DeleteAt( 0, Size() );
}

template<class T, class Allocator>
inline void CPointerArray<T, Allocator>::FreeBuffer()
{
	DeleteAll();
	body.FreeBuffer();
}

template<class T, class Allocator>
inline void CPointerArray<T, Allocator>::MoveTo( CPointerArray<T, Allocator>& other )
{
	if( &other == this ) {
		return;
	}

	other.DeleteAll();
	body.MoveTo( other.body );
}

template<class T, class Allocator>
inline void CPointerArray<T, Allocator>::AppendTo( CPointerArray<T, Allocator>& other )
{
	AssertFO( &other != this );

	other.body.Add( body );
	body.DeleteAll();
}

template<class T, class Allocator> template <class COMPARE>
inline void CPointerArray<T, Allocator>::QuickSort( COMPARE* param )
{
	body.QuickSort( param );
}

template<class T, class Allocator> template <class COMPARE>
inline void CPointerArray<T, Allocator>::QuickSort()
{
	body.template QuickSort<COMPARE>();
}

template<class T, class Allocator> template<class COMPARE>
inline bool CPointerArray<T, Allocator>::IsSorted( COMPARE* compare ) const
{
	return body.IsSorted( compare );
}

template<class T, class Allocator> template<class COMPARE>
inline bool CPointerArray<T, Allocator>::IsSorted() const
{
	return body.template IsSorted<COMPARE>();
}

template<class T, class Allocator> template<class COMPARE, class SEARCHED_TYPE>
inline int CPointerArray<T, Allocator>::FindInsertionPoint( const SEARCHED_TYPE& what, COMPARE* param ) const
{
	return body.template FindInsertionPoint<COMPARE, SEARCHED_TYPE>( what, param );
}

template<class T, class Allocator> template<class COMPARE, class SEARCHED_TYPE>
inline int CPointerArray<T, Allocator>::FindInsertionPoint( const SEARCHED_TYPE& what ) const
{
	return body.template FindInsertionPoint<COMPARE, SEARCHED_TYPE>( what );
}

template<class T, class Allocator>
int CPointerArray<T, Allocator>::Find( const T* obj, int start ) const
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

template<class T, class Allocator>
bool CPointerArray<T, Allocator>::Detach( const T* obj )
{
	int pos = Find( obj );
	if( pos != NotFound ) {
		DetachAt( pos );
		return true;
	} else {
		return false;
	}
}

template<class T, class Allocator>
bool CPointerArray<T, Allocator>::Delete( T* obj )
{
	int pos = Find( obj );
	if( pos != NotFound ) {
	 	DeleteAt( pos );
		return true;
	} else {
		return false;
	}
}

template<class T, class Allocator>
void CPointerArray<T, Allocator>::DeleteAt( int location, int num )
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
		body[pos] = 0;
		delete current;
	}
	body.DeleteAt( location, num );
}

template<class T, class Allocator>
void CPointerArray<T, Allocator>::DeleteLast()
{
	T* current = body.Last();
	body.DeleteLast();
	delete current;
}

template<class T, class Allocator>
void CPointerArray<T, Allocator>::Serialize( CArchive& ar )
{
	if( ar.IsStoring() ) {
		ar << static_cast<unsigned int>( Size() );
		for( int i = 0; i < Size(); i++ ) {
			if( ( *this )[i] != 0 ) {
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

//-------------------------------------------------------------------------------------------

template<class T, class Allocator>
inline void ArrayMemMoveElement( CPointerArray<T, Allocator>* dest, CPointerArray<T, Allocator>* src )
{
	PresumeFO( dest != src );
	::new( dest ) CPointerArray<T, Allocator>;
	src->MoveTo( *dest );
	src->~CPointerArray<T, Allocator>();
}

} // namespace FObj
