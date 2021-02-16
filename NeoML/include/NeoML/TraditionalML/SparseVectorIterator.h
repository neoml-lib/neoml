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

#include <NeoML/NeoMLDefs.h>

namespace NeoML {

// A sparse vector iterator
template<class TElement>
class CSparseVectorIterator {
public:
	typedef typename TElement::TIndex TIndex;
	typedef typename TElement::TValue TValue;

	CSparseVectorIterator() : Indexes( nullptr ), Values( nullptr ) {}
	CSparseVectorIterator( TIndex* indexes, TValue* values ) : Indexes( indexes ), Values( values ) {}

	// to allow the creation of const iterator from a non-const
	template<class TAnotherElement>
	CSparseVectorIterator( const CSparseVectorIterator<TAnotherElement>& it ) :
		Indexes( it.Indexes ), Values( it.Values ) {}

	TElement operator*() const { return TElement( { *Indexes, *Values } ); }

	template<class TAnotherElement>
	bool operator==( const CSparseVectorIterator<TAnotherElement>& other ) const;
	template<class TAnotherElement>
	bool operator!=( const CSparseVectorIterator<TAnotherElement>& other ) const { return !( *this == other ); }
	template<class TAnotherElement>
	bool operator<( const CSparseVectorIterator<TAnotherElement>& other ) const;
	template<class TAnotherElement>
	bool operator>( const CSparseVectorIterator<TAnotherElement>& other ) const { return other < *this; }
	template<class TAnotherElement>
	bool operator<=( const CSparseVectorIterator<TAnotherElement>& other ) const { return !( *this > other ); }
	template<class TAnotherElement>
	bool operator>=( const CSparseVectorIterator<TAnotherElement>& other ) const { return !( *this < other ); }

	CSparseVectorIterator& operator++();
	CSparseVectorIterator operator++( int );
	CSparseVectorIterator& operator--();
	CSparseVectorIterator operator--( int );
	CSparseVectorIterator& operator+=( int offset );
	CSparseVectorIterator operator+( int offset ) const;
	CSparseVectorIterator& operator-=( int offset );
	CSparseVectorIterator operator-( int offset ) const;
	int operator-( const CSparseVectorIterator& other ) const;

private:
	TIndex* Indexes;
	TValue* Values;

	template<class TAnotherElement> friend class CSparseVectorIterator;
};

template<class TElement>
template<class TAnotherElement>
inline bool CSparseVectorIterator<TElement>::operator==( const CSparseVectorIterator<TAnotherElement>& other ) const
{
	return other.Indexes == Indexes && other.Values == Values;
}

template<class TElement>
template<class TAnotherElement>
inline bool CSparseVectorIterator<TElement>::operator<( const CSparseVectorIterator<TAnotherElement>& other ) const
{
	return Indexes < other.Indexes && Values < other.Values;
}

template<class TElement>
inline CSparseVectorIterator<TElement>& CSparseVectorIterator<TElement>::operator++()
{
	Indexes++;
	Values++;
	return *this;
}

template<class TElement>
inline CSparseVectorIterator<TElement> CSparseVectorIterator<TElement>::operator++( int )
{
	CSparseVectorIterator old( *this );
	Indexes++;
	Values++;
	return old;
}

template<class TElement>
inline CSparseVectorIterator<TElement>& CSparseVectorIterator<TElement>::operator--()
{
	Indexes--;
	Values--;
	return *this;
}

template<class TElement>
inline CSparseVectorIterator<TElement> CSparseVectorIterator<TElement>::operator--( int )
{
	CSparseVectorIterator old( *this );
	Indexes--;
	Values--;
	return old;
}

template<class TElement>
inline CSparseVectorIterator<TElement>& CSparseVectorIterator<TElement>::operator+=( int offset )
{
	Indexes += offset;
	Values += offset;
	return *this;
}

template<class TElement>
inline CSparseVectorIterator<TElement> CSparseVectorIterator<TElement>::operator+( int offset ) const
{
	CSparseVectorIterator tmp( *this );
	tmp += offset;
	return tmp;
}

template<class TElement>
inline CSparseVectorIterator<TElement>& CSparseVectorIterator<TElement>::operator-=( int offset )
{
	Indexes -= offset;
	Values -= offset;
	return *this;
}

template<class TElement>
inline CSparseVectorIterator<TElement> CSparseVectorIterator<TElement>::operator-( int offset ) const
{
	CSparseVectorIterator tmp( *this );
	tmp -= offset;
	return tmp;
}

template<class TElement>
inline int CSparseVectorIterator<TElement>::operator-( const CSparseVectorIterator<TElement>& other ) const
{
	NeoPresume( Indexes - other.Indexes == Values - other.Values );
	return static_cast<int>( Indexes - other.Indexes );
}

} // namespace NeoML
