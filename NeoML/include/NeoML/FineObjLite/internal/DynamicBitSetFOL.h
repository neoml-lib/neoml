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

template<int InitialSize, class Allocator>
class CDynamicBitSet; 

template<int InitialSize, class Allocator>
CArchive& operator<< ( CArchive&, const CDynamicBitSet<InitialSize, Allocator>& );

template<int InitialSize, class Allocator>
CArchive& operator>> ( CArchive&, CDynamicBitSet<InitialSize, Allocator>& );

template<int InitialSize = 1, class Allocator = CurrentMemoryManager>
class CDynamicBitSet {
public:
	typedef int TElement;

	CDynamicBitSet() {}
	explicit CDynamicBitSet( int element );
	CDynamicBitSet( const CDynamicBitSet& other );
	CDynamicBitSet( const int* elements, int elementsCount );

	CDynamicBitSet& operator=( const CDynamicBitSet& other );
	void MoveTo( CDynamicBitSet& other );
	
	void* GetPtr() { return body.GetPtr(); }
	const void* GetPtr() const { return body.GetPtr(); }
	int BufferByteSize() const { return body.Size() * sizeof( bodyType ); }

	void SetBufferSize( int elementsCount );
	int GetBufferSize() const { return body.Size() * BitsPerElement; }

	bool IsEmpty() const { return isTailEmpty( 0 ); }
	bool IsEmpty( int from, int count ) const;
	void Clean() { cleanTail( 0 ); }
	void Empty();
	void FreeBuffer();
	bool Has( const CDynamicBitSet& subset ) const;
	bool Has( int element ) const;
	bool operator[]( int element ) const { return Has( element ); }

	CDynamicBitSet operator | ( const CDynamicBitSet& set ) const;
	CDynamicBitSet operator | ( int element ) const;
	CDynamicBitSet operator & ( const CDynamicBitSet& set ) const;
	CDynamicBitSet operator & ( int element ) const;
	CDynamicBitSet operator - ( const CDynamicBitSet& set ) const;
	CDynamicBitSet operator - ( int element ) const;
	CDynamicBitSet operator ^ ( const CDynamicBitSet& set ) const;
	CDynamicBitSet operator ^ ( int element ) const;
	
	const CDynamicBitSet& operator |= ( const CDynamicBitSet& set );
	const CDynamicBitSet& operator |= ( int element );
	const CDynamicBitSet& operator &= ( const CDynamicBitSet& set );
	const CDynamicBitSet& operator &= ( int element );
	const CDynamicBitSet& operator -= ( const CDynamicBitSet& set );
	const CDynamicBitSet& operator -= ( int element );
	const CDynamicBitSet& operator ^= ( const CDynamicBitSet& set );
	const CDynamicBitSet& operator ^= ( int element );

	void Set( int element ) { *this |= element; }
	void Set( const CDynamicBitSet& set ) { *this |= set; }
	void Set( int from, int count );
	void Reset( int element ) { *this -= element; }
	void Reset( const CDynamicBitSet& set ) { *this -= set; }
	void Reset( int from, int count );
	void Invert( int element ) { *this ^= element; }
	void Invert( const CDynamicBitSet& set ) { *this ^= set; }
	void Invert( int from, int count );

	bool operator == ( const CDynamicBitSet& set ) const;
	bool operator != ( const CDynamicBitSet& set ) const { return !( *this == set ); }

	bool Intersects( const CDynamicBitSet& set ) const;

	void ShiftForward();
	void ShiftBackward();
	
	int FindFirstElement() const;
	int FindLastElement() const;
	int FindNextElement( int from ) const;
	int FindPrevElement( int from ) const;
	int ElementsCount() const;

	int Compare( const CDynamicBitSet& ) const;

	int HashKey() const;

	friend CArchive& operator<< <InitialSize, Allocator>( CArchive&, const CDynamicBitSet<InitialSize, Allocator>& );
	friend CArchive& operator>> <InitialSize, Allocator>( CArchive&, CDynamicBitSet<InitialSize, Allocator>& );

private:
	typedef unsigned int bodyType;
	static const int BitsPerElement = CHAR_BIT * sizeof( bodyType );

	CFastArray<bodyType, ( InitialSize + BitsPerElement - 1 ) / BitsPerElement, Allocator> body;

	void grow( int newSize );
	void cleanTail( int from );
	bool isTailEmpty( int from ) const;
	static int index( int bit );
	static bodyType mask( int bit );
	static bodyType maskFrom( int from );
	static bodyType maskTo( int to );
};

//-------------------------------------------------------------------------------------

static const unsigned char BitSetElementsTable[256] = {
	0, 1, 1, 2, 1, 2, 2, 3,
	1, 2, 2, 3, 2, 3, 3, 4,
	1, 2, 2, 3, 2, 3, 3, 4,
	2, 3, 3, 4, 3, 4, 4, 5,
	1, 2, 2, 3, 2, 3, 3, 4,
	2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5,
	3, 4, 4, 5, 4, 5, 5, 6,
	1, 2, 2, 3, 2, 3, 3, 4,
	2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5,
	3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5,
	3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6,
	4, 5, 5, 6, 5, 6, 6, 7,
	1, 2, 2, 3, 2, 3, 3, 4,
	2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5,
	3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5,
	3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6,
	4, 5, 5, 6, 5, 6, 6, 7,
	2, 3, 3, 4, 3, 4, 4, 5,
	3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6,
	4, 5, 5, 6, 5, 6, 6, 7,
	3, 4, 4, 5, 4, 5, 5, 6,
	4, 5, 5, 6, 5, 6, 6, 7,
	4, 5, 5, 6, 5, 6, 6, 7,
	5, 6, 6, 7, 6, 7, 7, 8
};

static const unsigned char BitSetFirstElementsTable[256] = {
	8, 0, 1, 0, 2, 0, 1, 0, // 00000xxx 8 - undefined
	3, 0, 1, 0, 2, 0, 1, 0, // 00001xxx
	4, 0, 1, 0, 2, 0, 1, 0, // 00010xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 00011xxx
	5, 0, 1, 0, 2, 0, 1, 0, // 00100xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 00101xxx
	4, 0, 1, 0, 2, 0, 1, 0, // 00110xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 00111xxx
	6, 0, 1, 0, 2, 0, 1, 0, // 01000xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 01001xxx
	4, 0, 1, 0, 2, 0, 1, 0, // 01010xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 01011xxx
	5, 0, 1, 0, 2, 0, 1, 0, // 01100xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 01101xxx
	4, 0, 1, 0, 2, 0, 1, 0, // 01110xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 01111xxx
	7, 0, 1, 0, 2, 0, 1, 0, // 10000xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 10001xxx
	4, 0, 1, 0, 2, 0, 1, 0, // 10010xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 10011xxx
	5, 0, 1, 0, 2, 0, 1, 0, // 10100xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 10101xxx
	4, 0, 1, 0, 2, 0, 1, 0, // 10110xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 10111xxx
	6, 0, 1, 0, 2, 0, 1, 0, // 11000xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 11001xxx
	4, 0, 1, 0, 2, 0, 1, 0, // 11010xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 11011xxx
	5, 0, 1, 0, 2, 0, 1, 0, // 11100xxx
	3, 0, 1, 0, 2, 0, 1, 0, // 11101xxx
	4, 0, 1, 0, 2, 0, 1, 0, // 11110xxx
	3, 0, 1, 0, 2, 0, 1, 0  // 11111xxx
};

static const unsigned char BitSetNextMaskTable[CHAR_BIT] = {
	0xfe, // 11111110
	0xfc, // 11111100
	0xf8, // 11111000
	0xf0, // 11110000
	0xe0, // 11100000
	0xc0, // 11000000
	0x80, // 10000000
	0xff, // 11111111
};

static const unsigned char BitSetLastElementsTable[256] = {
	8,						// 00000000 undefined
	0,						// 00000001
	1, 1,					// 0000001x
	2, 2, 2, 2,				// 000001xx
	3, 3, 3, 3, 3, 3, 3, 3, // 00001xxx
	4, 4, 4, 4, 4, 4, 4, 4, // 00010xxx
	4, 4, 4, 4, 4, 4, 4, 4, // 00011xxx
	5, 5, 5, 5, 5, 5, 5, 5, // 00100xxx
	5, 5, 5, 5, 5, 5, 5, 5, // 00101xxx
	5, 5, 5, 5, 5, 5, 5, 5, // 00110xxx
	5, 5, 5, 5, 5, 5, 5, 5, // 00111xxx
	6, 6, 6, 6, 6, 6, 6, 6, // 01000xxx
	6, 6, 6, 6, 6, 6, 6, 6, // 01001xxx
	6, 6, 6, 6, 6, 6, 6, 6, // 01010xxx
	6, 6, 6, 6, 6, 6, 6, 6, // 01011xxx
	6, 6, 6, 6, 6, 6, 6, 6, // 01100xxx
	6, 6, 6, 6, 6, 6, 6, 6, // 01101xxx
	6, 6, 6, 6, 6, 6, 6, 6, // 01110xxx
	6, 6, 6, 6, 6, 6, 6, 6, // 01111xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 10000xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 10001xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 10010xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 10011xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 10100xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 10101xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 10110xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 10111xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 11000xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 11001xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 11010xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 11011xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 11100xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 11101xxx
	7, 7, 7, 7, 7, 7, 7, 7, // 11110xxx
	7, 7, 7, 7, 7, 7, 7, 7  // 11111xxx
};

static const unsigned char BitSetPrevMaskTable[CHAR_BIT] = {
	0xff, // 11111111
	0x01, // 00000001
	0x03, // 00000011
	0x07, // 00000111
	0x0f, // 00001111
	0x1f, // 00011111
	0x3f, // 00111111
	0x7f, // 01111111
};

//------------------------------------------------------------------------------------------------------------

template<int InitialSize, class Allocator>
inline typename CDynamicBitSet<InitialSize, Allocator>::bodyType 
	CDynamicBitSet<InitialSize, Allocator>::mask( int bit )
{
	PresumeFO( bit >= 0 );
	return 1 << ( ( static_cast<unsigned int>(bit) ) % BitsPerElement );
}

template<int InitialSize, class Allocator>
inline typename CDynamicBitSet<InitialSize, Allocator>::bodyType 
	CDynamicBitSet<InitialSize, Allocator>::maskFrom( int from )
{
	return ~bodyType( 0 ) & ~( mask( from ) - 1U );
}

template<int InitialSize, class Allocator>
inline typename CDynamicBitSet<InitialSize, Allocator>::bodyType 
	CDynamicBitSet<InitialSize, Allocator>::maskTo( int to )
{
	bodyType bitMask = mask( to );
	return ( bitMask - 1U ) | bitMask;
}

template<int InitialSize, class Allocator>
inline int CDynamicBitSet<InitialSize, Allocator>::index( int bit )
{
	PresumeFO( bit >= 0 );
	return ( static_cast<unsigned int>(bit) ) / BitsPerElement;
}

template<int InitialSize, class Allocator>
inline void CDynamicBitSet<InitialSize, Allocator>::cleanTail( int from )
{
	for( ; from < body.Size(); from++ ) {
		body[from] = 0;
	}
}

template<int InitialSize, class Allocator>
inline bool CDynamicBitSet<InitialSize, Allocator>::isTailEmpty( int from ) const
{
	for( ; from < body.Size(); from++ ) {
		if( body[from] != 0 ) {
			return false;
		}
	}
	return true;
}

template<int InitialSize, class Allocator>
inline CDynamicBitSet<InitialSize, Allocator>::CDynamicBitSet( int element )
{
	Set( element );
}

template<int InitialSize, class Allocator>
CDynamicBitSet<InitialSize, Allocator>::CDynamicBitSet( const int* elements, int elementsCount )
{
	PresumeFO( elementsCount >= 0 );
	PresumeFO( elements != 0 || elementsCount == 0 );

	for( int i = 0; i < elementsCount; i++ ) {
		Set( elements[i] );
	}
}

template<int InitialSize, class Allocator>
inline CDynamicBitSet<InitialSize, Allocator>::CDynamicBitSet( const CDynamicBitSet<InitialSize, Allocator>& other )
{
	other.body.CopyTo( body );
}

template<int InitialSize, class Allocator>
inline void CDynamicBitSet<InitialSize, Allocator>::MoveTo( CDynamicBitSet<InitialSize, Allocator>& other )
{
	body.MoveTo( other.body );
}

template<int InitialSize, class Allocator>
inline CDynamicBitSet<InitialSize, Allocator>& CDynamicBitSet<InitialSize, Allocator>::operator = ( 
	const CDynamicBitSet<InitialSize, Allocator>& other )
{
	other.body.CopyTo( body );
	return *this;
}

template<int InitialSize, class Allocator>
inline void CDynamicBitSet<InitialSize, Allocator>::Empty()
{
	body.DeleteAll();
}

template<int InitialSize, class Allocator>
inline void CDynamicBitSet<InitialSize, Allocator>::FreeBuffer()
{
	body.FreeBuffer();
}

template<int InitialSize, class Allocator>
inline bool CDynamicBitSet<InitialSize, Allocator>::Has( const CDynamicBitSet<InitialSize, Allocator>& subset ) const
{
	int minLength = min( body.Size(), subset.body.Size() );
	for( int i = 0; i < minLength; i++ ) {
		if( ~body[i] & subset.body[i] ) {
			return false;
		}
	}
	
	return subset.isTailEmpty( minLength );
}

template<int InitialSize, class Allocator>
bool CDynamicBitSet<InitialSize, Allocator>::IsEmpty( int from, int count ) const
{
	PresumeFO( count >= 0 && from >= 0 );
	
	if( count == 0 || from >= GetBufferSize() ) {
		return true;
	}
	
	int to = min( GetBufferSize() - 1, from + count - 1 );
	PresumeFO( from <= to );
	int fromIndex = index( from );
	int toIndex = index( to );
	if( fromIndex != toIndex ) {
		if( ( body[fromIndex] & maskFrom( from ) ) != 0 || ( body[toIndex] & maskTo( to ) ) != 0 ) {
			return false;
		}
		for( int i = fromIndex + 1; i < toIndex; i++ ) {
			if( body[i] != 0 ) {
				return false;
			}
		}
		return true;
	}
	return ( body[fromIndex] & maskFrom( from ) & maskTo( to ) ) == 0;
}

template<int InitialSize, class Allocator>
inline bool CDynamicBitSet<InitialSize, Allocator>::Has( int bit ) const
{
	const int bitIndex = index( bit );
	return bitIndex < body.Size() && ( body[bitIndex] & mask( bit ) ) != 0;
}

template<int InitialSize, class Allocator>
inline CDynamicBitSet<InitialSize, Allocator> CDynamicBitSet<InitialSize, Allocator>::operator|( 
	const CDynamicBitSet<InitialSize, Allocator>& set ) const
{
	return CDynamicBitSet<InitialSize, Allocator>( *this ) |= set;
}

template<int InitialSize, class Allocator>
inline CDynamicBitSet<InitialSize, Allocator> CDynamicBitSet<InitialSize, Allocator>::operator|( int element ) const
{
	return CDynamicBitSet<InitialSize, Allocator>( *this ) |= element;
}

template<int InitialSize, class Allocator>
inline CDynamicBitSet<InitialSize, Allocator> CDynamicBitSet<InitialSize, Allocator>::operator^( 
	const CDynamicBitSet<InitialSize, Allocator>& set ) const
{
	return CDynamicBitSet<InitialSize, Allocator>( *this ) ^= set;
}

template<int InitialSize, class Allocator>
inline CDynamicBitSet<InitialSize, Allocator> CDynamicBitSet<InitialSize, Allocator>::operator^( int element ) const
{
	return CDynamicBitSet<InitialSize, Allocator>( *this ) ^= element;
}

template<int InitialSize, class Allocator>
inline CDynamicBitSet<InitialSize, Allocator> CDynamicBitSet<InitialSize, Allocator>::operator&( 
	const CDynamicBitSet<InitialSize, Allocator>& set ) const
{
	return CDynamicBitSet<InitialSize, Allocator>( *this ) &= set;
}

template<int InitialSize, class Allocator>
inline CDynamicBitSet<InitialSize, Allocator> CDynamicBitSet<InitialSize, Allocator>::operator&( int element ) const
{
	return CDynamicBitSet<InitialSize, Allocator>( *this ) &= element;
}

template<int InitialSize, class Allocator>
inline CDynamicBitSet<InitialSize, Allocator> CDynamicBitSet<InitialSize, Allocator>::operator-( 
	const CDynamicBitSet<InitialSize, Allocator>& set ) const
{
	return CDynamicBitSet<InitialSize, Allocator>( *this ) -= set;
}

template<int InitialSize, class Allocator>
inline CDynamicBitSet<InitialSize, Allocator> CDynamicBitSet<InitialSize, Allocator>::operator-( int bit ) const
{
	return CDynamicBitSet<InitialSize, Allocator>( *this ) -= bit;
}

template<int InitialSize, class Allocator>
inline const CDynamicBitSet<InitialSize, Allocator>& CDynamicBitSet<InitialSize, Allocator>::operator|=( 
	const CDynamicBitSet<InitialSize, Allocator>& set )
{
	if( set.body.Size() > body.Size() ) {
		grow( set.body.Size() );
	}
	for( int i = 0; i < set.body.Size(); i++ ) {
		body[i] |= set.body[i];
	}
	return *this;
}

template<int InitialSize, class Allocator>
inline const CDynamicBitSet<InitialSize, Allocator>& CDynamicBitSet<InitialSize, Allocator>::operator|=( int element )
{
	const int bitIndex = index( element );
	if( bitIndex >= body.Size() ) {
		grow( bitIndex + 1 );
	}
	body[bitIndex] |= mask( element );
	return *this;
}

template<int InitialSize, class Allocator>
inline const CDynamicBitSet<InitialSize, Allocator>& CDynamicBitSet<InitialSize, Allocator>::operator^=( 
	const CDynamicBitSet<InitialSize, Allocator>& set )
{
	if( set.body.Size() > body.Size() ) {
		grow( set.body.Size() );
	}
	for( int i = 0; i < set.body.Size(); i++ ) {
		body[i] ^= set.body[i];
	}
	return *this;
}

template<int InitialSize, class Allocator>
inline const CDynamicBitSet<InitialSize, Allocator>& CDynamicBitSet<InitialSize, Allocator>::operator^=( int element )
{
	const int bitIndex = index( element );
	if( bitIndex >= body.Size() ) {
		grow( bitIndex + 1 );
	}
	body[bitIndex] ^= mask( element );
	return *this;
}

template<int InitialSize, class Allocator>
inline const CDynamicBitSet<InitialSize, Allocator>& CDynamicBitSet<InitialSize, Allocator>::operator&=( 
	const CDynamicBitSet<InitialSize, Allocator>& set )
{
	const int minLength = min( body.Size(), set.body.Size() );
	for( int i = 0; i < minLength; i++ ) {
		body[i] &= set.body[i];
	}
	cleanTail( minLength );
	return *this;
}

template<int InitialSize, class Allocator>
inline const CDynamicBitSet<InitialSize, Allocator>& CDynamicBitSet<InitialSize, Allocator>::operator&=( int element )
{
	const int bitIndex = index( element );
	if( bitIndex < body.Size() ) {
		bodyType newBody = body[bitIndex] & mask( element );
		Clean();
		body[bitIndex] = newBody;
	} else {
		Clean();
	}
	return *this;
}

template<int InitialSize, class Allocator>
inline const CDynamicBitSet<InitialSize, Allocator>& CDynamicBitSet<InitialSize, Allocator>::operator-=( 
	const CDynamicBitSet<InitialSize, Allocator>& set )
{
	const int minLength = min( body.Size(), set.body.Size() );
	for( int i = 0; i < minLength; i++ ) {
		body[i] &= ~set.body[i];
	}
	return *this;
}

template<int InitialSize, class Allocator>
inline const CDynamicBitSet<InitialSize, Allocator>& CDynamicBitSet<InitialSize, Allocator>::operator-=( int element )
{
	const int bitIndex = index( element );
	if( bitIndex < body.Size() ) {
		body[bitIndex] &= ~mask( element );
	}
	return *this;
} 

template<int InitialSize, class Allocator>
inline bool CDynamicBitSet<InitialSize, Allocator>::operator==( 
	const CDynamicBitSet<InitialSize, Allocator>& set ) const
{
	const int minLength = min( body.Size(), set.body.Size() );
	int i;
	for( i = 0; i < minLength; i++ ) {
		if( body[i] != set.body[i] ) {
			return false;
		}
	}
	return isTailEmpty( i ) && set.isTailEmpty( i );
}

template<int InitialSize, class Allocator>
inline bool CDynamicBitSet<InitialSize, Allocator>::Intersects( 
	const CDynamicBitSet<InitialSize, Allocator>& set ) const
{
	const int minLength = min( body.Size(), set.body.Size() );
	for( int i = 0; i < minLength; i++ ) {
		if( body[i] & set.body[i] ) {
			return true;
		}
	}
	return false;
}

template<int InitialSize, class Allocator>
void CDynamicBitSet<InitialSize, Allocator>::ShiftForward()
{
	bodyType firstBit = 0;
	for( int i = 0; i < body.Size(); i++ ) {
		bodyType newFirstBit = ( body[i] & ( 1 << ( BitsPerElement - 1 ) ) ) >> ( BitsPerElement - 1 );
		body[i] = ( body[i] << 1 ) | firstBit;
		firstBit = newFirstBit;
	}
	if( firstBit != 0 ) {
		body.Add( firstBit );
	}
}

template<int InitialSize, class Allocator>
void CDynamicBitSet<InitialSize, Allocator>::ShiftBackward()
{
	bodyType lastBit = 0;
	for( int i = body.Size() - 1; i >= 0; i-- ) {
		bodyType newLastBit = ( body[i] & 1 ) << ( BitsPerElement - 1 );
		body[i] = ( body[i] >> 1 ) | lastBit;
		lastBit = newLastBit;
	}
	PresumeFO( lastBit == 0 );
}

template<int InitialSize, class Allocator>
inline int CDynamicBitSet<InitialSize, Allocator>::FindFirstElement() const
{
	if( GetBufferSize() == 0 ) {
		return NotFound;
	}
	if( Has( 0 ) ) {
		return 0;
	}
	return FindNextElement( 0 );
}

template<int InitialSize, class Allocator>
inline int CDynamicBitSet<InitialSize, Allocator>::FindLastElement() const
{
	return FindPrevElement( GetBufferSize() );
}

template<int InitialSize, class Allocator>
void CDynamicBitSet<InitialSize, Allocator>::grow( int newLength )
{
	int oldLength = body.Size();
	AssertFO( newLength > oldLength );
	body.SetSize( newLength );
	cleanTail( oldLength );
}

template<int InitialSize, class Allocator>
int CDynamicBitSet<InitialSize, Allocator>::ElementsCount() const
{
	const BYTE* ptr = reinterpret_cast<const BYTE*>(body.GetPtr());
	int size = sizeof( bodyType ) * body.Size();
	int count = 0;
	for( int i = 0; i < size; i++ ) {
		count += BitSetElementsTable[*ptr];
		ptr++;
	}

	return count;
}

template<int InitialSize, class Allocator>
int CDynamicBitSet<InitialSize, Allocator>::FindNextElement( int from ) const
{
	PresumeFO( from >= 0 );

	if( from >= GetBufferSize() - 1 ) {
		return NotFound;
	}

	const unsigned char* bodyPtr = reinterpret_cast<const unsigned char*>( body.GetPtr() );
	PresumeFO( bodyPtr != 0 );
	int bodyIndex = ( from + 1 ) / CHAR_BIT;
	PresumeFO( bodyIndex < GetBufferSize() / CHAR_BIT );
	unsigned char element = BitSetNextMaskTable[from % CHAR_BIT] & bodyPtr[bodyIndex];

	if( element != 0 ) {
		return BitSetFirstElementsTable[element] + bodyIndex * CHAR_BIT;
	}

	const int bytesCount = ( GetBufferSize() + CHAR_BIT - 1 ) / CHAR_BIT;
	for( bodyIndex++; bodyIndex < bytesCount; bodyIndex++ ) {
		PresumeFO( bodyIndex < GetBufferSize() / CHAR_BIT );
		element = bodyPtr[bodyIndex];
		if( element != 0 ) {
			return BitSetFirstElementsTable[element] + bodyIndex * CHAR_BIT;
		}
	}
	return NotFound;
}

template<int InitialSize, class Allocator>
int CDynamicBitSet<InitialSize, Allocator>::FindPrevElement( int from ) const
{
	PresumeFO( from >= 0 );

	from = min( from, GetBufferSize() );

	if( from <= 0 ) {
		return NotFound;
	}

	const unsigned char* bodyPtr = reinterpret_cast<const unsigned char*>( body.GetPtr() );
	PresumeFO( bodyPtr != 0 );
	int bodyIndex = ( from - 1 ) / CHAR_BIT;
	PresumeFO( bodyIndex < GetBufferSize() / CHAR_BIT );
	unsigned char element = BitSetPrevMaskTable[from % CHAR_BIT] & bodyPtr[bodyIndex];

	if( element != 0 ) {
		return BitSetLastElementsTable[element] + bodyIndex * CHAR_BIT;
	}

	for( bodyIndex--; bodyIndex >= 0; bodyIndex-- ) {
		PresumeFO( bodyIndex < GetBufferSize() / CHAR_BIT );
		element = bodyPtr[bodyIndex];
		if( element != 0 ) {
			return BitSetLastElementsTable[element] + bodyIndex * CHAR_BIT;
		}
	}
	return NotFound;
}

template<int InitialSize, class Allocator>
void CDynamicBitSet<InitialSize, Allocator>::Set( int from, int count )
{
	PresumeFO( count >= 0 );
	if( count == 0 ) {
		return;
	}
	SetBufferSize( from + count );
	int to = from + count - 1;
	int fromIndex = index( from );
	int toIndex = index( to );
	if( fromIndex != toIndex ) {
		for( int i = fromIndex + 1; i < toIndex; i++ ) {
			body[i] = ~bodyType( 0 );
		}
		body[fromIndex] |= maskFrom( from );
		body[toIndex] |= maskTo( to );
	} else {
		body[fromIndex] |= maskFrom( from ) & maskTo( to );
	}
}

template<int InitialSize, class Allocator>
void CDynamicBitSet<InitialSize, Allocator>::Reset( int from, int count )
{
	PresumeFO( count >= 0 );
	if( count == 0 ) {
		return;
	}
	SetBufferSize( from + count );
	int to = from + count - 1;
	int fromIndex = index( from );
	int toIndex = index( to );
	if( fromIndex != toIndex ) {
		for( int i = fromIndex + 1; i < toIndex; i++ ) {
			body[i] = 0;
		}
		body[fromIndex] &= ~maskFrom( from );
		body[toIndex] &= ~maskTo( to );
	} else {
		body[fromIndex] &= ~( maskFrom( from ) & maskTo( to ) );
	}
}
	
template<int InitialSize, class Allocator>
void CDynamicBitSet<InitialSize, Allocator>::Invert( int from, int count )
{
	PresumeFO( count >= 0 );
	if( count == 0 ) {
		return;
	}
	SetBufferSize( from + count );
	int to = from + count - 1;
	int fromIndex = index( from );
	int toIndex = index( to );
	if( fromIndex != toIndex ) {
		for( int i = fromIndex + 1; i < toIndex; i++ ) {
			body[i] = ~body[i];
		}
		body[fromIndex] ^= maskFrom( from );
		body[toIndex] ^= maskTo( to );
	} else {
		body[fromIndex] ^= maskFrom( from ) & maskTo( to );
	}
}

template<int InitialSize, class Allocator>
inline void CDynamicBitSet<InitialSize, Allocator>::SetBufferSize( int elementsCount )
{
	if( elementsCount == 0 ) {
		return;
	}
	const int newBodyLength = index( elementsCount - 1 ) + 1;
	if( newBodyLength > body.Size() ) {
		grow( newBodyLength );
	}
}

template<int InitialSize, class Allocator>
int CDynamicBitSet<InitialSize, Allocator>::Compare( const CDynamicBitSet<InitialSize, Allocator>& other ) const
{
	int minSize = min( body.Size(), other.body.Size() );
	int minDiff = ::memcmp( body.GetPtr(), other.body.GetPtr(), minSize * sizeof( bodyType ) );
	if(minDiff != 0) {
		return minDiff;
	}
	if( !isTailEmpty( minSize ) ) {
		return 1;
	} else if( !other.isTailEmpty( minSize ) ) {
		return -1;
	}
	return 0;
}

template<int InitialSize, class Allocator>
inline int CDynamicBitSet<InitialSize, Allocator>::HashKey() const
{
	int result = 0;
	for( int i = body.Size() - 1; i >= 0; i-- ) {
		result = ( result << 5 ) + result + static_cast<int>( body[i] );
	}
	return result;
}

template<int InitialSize, class Allocator>
CArchive& operator<<( CArchive& archive, const CDynamicBitSet<InitialSize, Allocator>& set )
{
	unsigned int count = set.body.Size();
	for( ; count > 0 && set.body[count - 1] == 0; count-- ) {
	}

	archive << count;
	archive.Write( set.body.GetPtr(), count * sizeof( DWORD ) );
	return archive;
}

template<int InitialSize, class Allocator>
CArchive& operator>>( CArchive& archive, CDynamicBitSet<InitialSize, Allocator>& set )
{
	unsigned int count;
	archive >> count;
	check( static_cast<int>( count ) >= 0, ERR_BAD_ARCHIVE, archive.Name() );
	set.body.SetSize( count );
	archive.Read( set.body.GetPtr(), count * sizeof( DWORD ) );
	return archive;
}

//-------------------------------------------------------------------------------------

template<int InitialSize, class Allocator>
inline void ArrayMemMoveElement( CDynamicBitSet<InitialSize, Allocator>* dest, 
	CDynamicBitSet<InitialSize, Allocator>* source )
{
	PresumeFO( dest != source );
	new( dest ) CDynamicBitSet<InitialSize, Allocator>;
	source->MoveTo( *dest );
	source->~CDynamicBitSet<InitialSize, Allocator>();
}

} // namespace FObj
