/* Copyright © 2024 ABBYY

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

#include <FineObjLiteDefs.h>

namespace FObj {

template<class TBitSet>
class CDynamicBitSetIterator {
public:
	CDynamicBitSetIterator() : bitSet( 0 ), element( NotFound ) {}
	CDynamicBitSetIterator( const TBitSet* _bitSet, int _element ) : bitSet( _bitSet ), element( _element ) {}

	bool operator==( const CDynamicBitSetIterator& other ) const;
	bool operator!=( const CDynamicBitSetIterator& other ) const { return !( *this == other ); }

	CDynamicBitSetIterator& operator++();
	CDynamicBitSetIterator operator++( int );

	int operator*() const { return element; }

private:
	const TBitSet* bitSet;
	int element;

	const TBitSet& safeBitSet() const { PresumeFO( bitSet != 0 ); return *bitSet; }
};

//---------------------------------------------------------------------------------------------------------------------

template<class TBitSet>
inline bool CDynamicBitSetIterator<TBitSet>::operator==( const CDynamicBitSetIterator& other ) const
{
	PresumeFO( bitSet == other.bitSet );
	return element == other.element;
}

template<class TBitSet>
inline CDynamicBitSetIterator<TBitSet>& CDynamicBitSetIterator<TBitSet>::operator++()
{
	element = safeBitSet().FindNextElement( element );
	return *this;
}

template<class TBitSet>
inline CDynamicBitSetIterator<TBitSet> CDynamicBitSetIterator<TBitSet>::operator++( int )
{
	const CDynamicBitSetIterator old( *this );
	element = safeBitSet().FindNextElement( element );
	return old;
}

} // namespace FObj
