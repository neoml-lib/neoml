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

#include <NeoML/TraditionalML/VectorIterator.h>

namespace NeoML {

// A vector iterator
template<typename TValue>
class CVectorIterator {
public:
	CVectorIterator() : Values( nullptr ) {}
	CVectorIterator( TValue* values ) : Values( values ) {}

	// to allow the creation of const iterator from a non-const
	template<typename TAnotherValue>
	CVectorIterator( const CVectorIterator<TAnotherValue>& it ) : Values( it.Values ) {}

	TValue& operator*() const { return *Values; }

	template<typename TAnotherValue>
	bool operator==( const CVectorIterator<TAnotherValue>& other ) const { return other.Values == Values; }
	template<typename TAnotherValue>
	bool operator!=( const CVectorIterator<TAnotherValue>& other ) const { return !( *this == other ); }
	template<typename TAnotherValue>
	bool operator<( const CVectorIterator<TAnotherValue>& other ) const { return Values < other.Values; }
	template<typename TAnotherValue>
	bool operator>( const CVectorIterator<TAnotherValue>& other ) const { return other < *this; }
	template<typename TAnotherValue>
	bool operator<=( const CVectorIterator<TAnotherValue>& other ) const { return !( *this > other ); }
	template<typename TAnotherValue>
	bool operator>=( const CVectorIterator<TAnotherValue>& other ) const { return !( *this < other ); }

	CVectorIterator& operator++();
	CVectorIterator operator++( int );
	CVectorIterator& operator--();
	CVectorIterator operator--( int );
	CVectorIterator& operator+=( int offset );
	CVectorIterator operator+( int offset ) const;
	CVectorIterator& operator-=( int offset );
	CVectorIterator operator-( int offset ) const;
	int operator-( const CVectorIterator& other ) const;

private:
	TValue* Values;

	template<typename TAnotherValue> friend class CVectorIterator;
};

template<typename TValue>
inline CVectorIterator<TValue>& CVectorIterator<TValue>::operator++()
{
	Values++;
	return *this;
}

template<typename TValue>
inline CVectorIterator<TValue> CVectorIterator<TValue>::operator++( int )
{
	CVectorIterator old( *this );
	Values++;
	return old;
}

template<typename TValue>
inline CVectorIterator<TValue>& CVectorIterator<TValue>::operator--()
{
	Values--;
	return *this;
}

template<typename TValue>
inline CVectorIterator<TValue> CVectorIterator<TValue>::operator--( int )
{
	CVectorIterator old( *this );
	Values--;
	return old;
}

template<typename TValue>
inline CVectorIterator<TValue>& CVectorIterator<TValue>::operator+=( int offset )
{
	Values += offset;
	return *this;
}

template<typename TValue>
inline CVectorIterator<TValue> CVectorIterator<TValue>::operator+( int offset ) const
{
	CVectorIterator tmp( *this );
	tmp += offset;
	return tmp;
}

template<typename TValue>
inline CVectorIterator<TValue>& CVectorIterator<TValue>::operator-=( int offset )
{
	Values -= offset;
	return *this;
}

template<typename TValue>
inline CVectorIterator<TValue> CVectorIterator<TValue>::operator-( int offset ) const
{
	CVectorIterator tmp( *this );
	tmp -= offset;
	return tmp;
}

template<typename TValue>
inline int CVectorIterator<TValue>::operator-( const CVectorIterator<TValue>& other ) const
{
	return static_cast<int>( Values - other.Values );
}

} // namespace NeoML
