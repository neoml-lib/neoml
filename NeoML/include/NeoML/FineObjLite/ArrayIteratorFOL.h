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

#include <ErrorsFOL.h>

namespace FObj {

template<class TArray>
class CArrayIterator;

//---------------------------------------------------------------------------------------------------------------------

// Iterator over constant CArray and CFastArray
// Usage:
//		CArray<CMyValue> array; // ...
//		for( const CMyValue& value : array ) { ... }
template<class TArray>
class CConstArrayIterator {
public:
	typedef typename TArray::TElement TElement;

	CConstArrayIterator();
	CConstArrayIterator( const TElement* ptr, const TArray* arr );

	// Member access
	const TElement& operator*() const { return *GetSafePtr(); }
	const TElement* operator->() const { return GetSafePtr(); }
	const TElement& operator[]( int index ) const { return GetSafePtr()[index]; }

	// Comparison
	bool operator==( const CConstArrayIterator& other ) const;
	bool operator!=( const CConstArrayIterator& other ) const { return !( *this == other ); }
	bool operator<( const CConstArrayIterator& other ) const;
	bool operator>( const CConstArrayIterator& other ) const { return other < *this; }
	bool operator<=( const CConstArrayIterator& other ) const { return !( *this > other ); }
	bool operator>=( const CConstArrayIterator& other ) const { return !( *this < other ); }

	// Pointer arithmetics
	CConstArrayIterator& operator++();
	CConstArrayIterator operator++( int );
	CConstArrayIterator& operator--();
	CConstArrayIterator operator--( int );
	CConstArrayIterator& operator+=( int offset );
	CConstArrayIterator operator+( int offset ) const;
	CConstArrayIterator& operator-=( int offset );
	CConstArrayIterator operator-( int offset ) const;
	int operator-( const CConstArrayIterator& other ) const;

protected:
	// Returns pointer with validity check
	const TElement* GetSafePtr() const { PresumeFO( isValidPtr() ); return ptr; }

private:
	// Pointer to the current element of array
	const TElement* ptr;

	// Pointer to array which is being iterated
	// Used for checking out-of-bound access and buffer reallocatoin
#ifdef _DEBUG
	const TArray* arr;
#endif // _DEBUG

	bool isValidPtr() const;
};

//---------------------------------------------------------------------------------------------------------------------

// CConstArrayIterator methods

template<class TArray>
inline CConstArrayIterator<TArray>::CConstArrayIterator() : 
	ptr( 0 )
{
#ifdef _DEBUG
	arr = 0;
#endif // _DEBUG
}

template<class TArray>
inline CConstArrayIterator<TArray>::CConstArrayIterator( const TElement* _ptr, const TArray* _arr ) : 
	ptr( _ptr )
{
#ifdef _DEBUG
	arr = _arr;
#else // !_DEBUG
	( void ) _arr;
#endif // _DEBUG
}

template<class TArray>
inline bool CConstArrayIterator<TArray>::operator==( const CConstArrayIterator& other ) const
{
	PresumeFO( arr == other.arr );
	return ptr == other.ptr;
}

template<class TArray>
inline bool CConstArrayIterator<TArray>::operator<( const CConstArrayIterator& other ) const
{
	PresumeFO( arr == other.arr );
	return ptr < other.ptr;
}

template<class TArray>
inline CConstArrayIterator<TArray>& CConstArrayIterator<TArray>::operator++()
{
	ptr++;
	return *this;
}

template<class TArray>
inline CConstArrayIterator<TArray> CConstArrayIterator<TArray>::operator++( int )
{
	const CConstArrayIterator old( *this );
	ptr++;
	return old;
}

template<class TArray>
inline CConstArrayIterator<TArray>& CConstArrayIterator<TArray>::operator+=( int offset )
{
	ptr += offset;
	return *this;
}

template<class TArray>
inline CConstArrayIterator<TArray> CConstArrayIterator<TArray>::operator+( int offset ) const
{
	CConstArrayIterator tmp( *this );
	tmp += offset;
	return tmp;
}

template<class TArray>
inline CConstArrayIterator<TArray>& CConstArrayIterator<TArray>::operator--()
{
	ptr--;
	return *this;
}

template<class TArray>
inline CConstArrayIterator<TArray> CConstArrayIterator<TArray>::operator--( int )
{
	const CConstArrayIterator old( *this );
	ptr--;
	return old;
}

template<class TArray>
inline CConstArrayIterator<TArray>& CConstArrayIterator<TArray>::operator-=( int offset )
{
	ptr -= offset;
	return *this;
}

template<class TArray>
inline CConstArrayIterator<TArray> CConstArrayIterator<TArray>::operator-( int offset ) const
{
	CConstArrayIterator tmp( *this );
	tmp -= offset;
	return tmp;
}

template<class TArray>
inline int CConstArrayIterator<TArray>::operator-( const CConstArrayIterator& other ) const
{
	PresumeFO( arr == other.arr );
	return to<int>( ptr - other.ptr );
}

// Checks that pointer is valid
template<class TArray>
inline bool CConstArrayIterator<TArray>::isValidPtr() const
{
	// Check that ptr is actually inside of array buffer during Debug
#ifdef _DEBUG
	if( arr == 0 ) {
		return false;
	}
	const TElement* buffer = arr->GetPtr();
	return buffer <= ptr && ptr < buffer + arr->Size();
#else // _DEBUG
	return ptr != 0;
#endif // _DEBUG
}

template<class TArray>
inline CConstArrayIterator<TArray> operator+( int offset, const CConstArrayIterator<TArray>& it )
{
	return it + offset;
}

//---------------------------------------------------------------------------------------------------------------------

// Iterator over CArray and CFastArray
// Usage:
//		CArray<CMyValue> array; // ...
//		for( CMyValue& value : array ) { ... }
// It's derived from CConstArrayIterator, which allows us to skip all the implementation of comparison between
// const and non-const iterators (at the cost of const_cast in the CArrayIterator::GetSafePtr)
template<class TArray>
class CArrayIterator : public CConstArrayIterator<TArray> {
public:
	CArrayIterator() {}
	CArrayIterator( typename TArray::TElement* _ptr, const TArray* _arr ) : CConstArrayIterator<TArray>( _ptr, _arr ) {}

	// Member access
	typename TArray::TElement& operator*() const { return *GetSafePtr(); }
	typename TArray::TElement* operator->() const { return GetSafePtr(); }
	typename TArray::TElement& operator[]( int index ) const { return GetSafePtr()[index]; }

	// Pointer arithmetics
	CArrayIterator& operator++();
	CArrayIterator operator++( int );
	CArrayIterator& operator+=( int offset );
	CArrayIterator operator+( int offset ) const;
	CArrayIterator& operator--();
	CArrayIterator operator--( int );
	CArrayIterator& operator-=( int offset );
	CArrayIterator operator-( int offset ) const;
	int operator-( const CConstArrayIterator<TArray>& other ) const;

protected:
	// Returns pointer with validity check
	typename TArray::TElement* GetSafePtr() const { return const_cast<typename TArray::TElement*>( CConstArrayIterator<TArray>::GetSafePtr() ); }

private:
	// Conversion to reference to CConstArrayIterator
	// Allows us to shorten the calls to CConstArrayIterator methods
	const CConstArrayIterator<TArray>& myBase() const { return *this; }
	CConstArrayIterator<TArray>& myBase() { return *this; }
};

//---------------------------------------------------------------------------------------------------------------------

// CArrayIterator methods

template<class TArray>
inline CArrayIterator<TArray>& CArrayIterator<TArray>::operator++()
{
	++myBase();
	return *this;
}

template<class TArray>
inline CArrayIterator<TArray> CArrayIterator<TArray>::operator++( int )
{
	const CArrayIterator old( *this );
	++myBase();
	return old;
}

template<class TArray>
inline CArrayIterator<TArray>& CArrayIterator<TArray>::operator+=( int offset )
{
	myBase() += offset;
	return *this;
}

template<class TArray>
inline CArrayIterator<TArray> CArrayIterator<TArray>::operator+( int offset ) const
{
	CArrayIterator tmp( *this );
	tmp += offset;
	return tmp;
}

template<class TArray>
inline CArrayIterator<TArray>& CArrayIterator<TArray>::operator--()
{
	--myBase();
	return *this;
}

template<class TArray>
inline CArrayIterator<TArray> CArrayIterator<TArray>::operator--( int )
{
	const CArrayIterator old( *this );
	--myBase();
	return old;
}

template<class TArray>
inline CArrayIterator<TArray>& CArrayIterator<TArray>::operator-=( int offset )
{
	myBase() -= offset;
	return *this;
}

template<class TArray>
inline CArrayIterator<TArray> CArrayIterator<TArray>::operator-( int offset ) const
{
	CArrayIterator tmp( *this );
	tmp -= offset;
	return tmp;
}

template<class TArray>
inline int CArrayIterator<TArray>::operator-( const CConstArrayIterator<TArray>& other ) const
{
	return myBase() - other;
}

template<class TArray>
inline CArrayIterator<TArray> operator+( int offset, const CArrayIterator<TArray>& it )
{
	return it + offset;
}

} // namespace FObj
