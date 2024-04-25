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

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <NeoMathEngine/BlobType.h>
#include <initializer_list>

namespace NeoML {

// The names of the blob dimensions
enum TBlobDim {
	BD_BatchLength = 0, // the sequence length
	BD_BatchWidth,	// the batch size
	BD_ListSize,	// the size of the list to be processed together
	BD_Height,
	BD_Width,
	BD_Depth,
	BD_Channels,

	BD_Count	// this constant is equal to the number of dimensions
};

const TBlobDim DimensionName[BD_Count] = { BD_BatchLength, BD_BatchWidth, BD_ListSize, BD_Height, BD_Width, BD_Depth, BD_Channels };

// operators for iterating through the blob dimensions:
inline TBlobDim& operator++( TBlobDim& d )
{
	d = TBlobDim( int( d ) + 1 );
	return d;
}

constexpr inline TBlobDim operator+( TBlobDim d, int i )
{
	return TBlobDim( int( d ) + i );
}

//---------------------------------------------------------------------------------------------------------------------

// CBlobDesc is the base blob descriptor for Math Engine functions
class NEOMATHENGINE_API CBlobDesc final {
public:
	static const int MaxDimensions = BD_Count;
	static const int FirstObjectDim = 3; // the number of the first object dimension (BD_Height)

	explicit CBlobDesc( TBlobType dataType = CT_Invalid );
	CBlobDesc( std::initializer_list<int> list );

	CBlobDesc( CBlobDesc&& ) = default;
	CBlobDesc( const CBlobDesc& ) = default;

	CBlobDesc& operator=( CBlobDesc&& ) = default;
	CBlobDesc& operator=( const CBlobDesc& other );

	bool operator==( const CBlobDesc& other ) const { return type == other.type && HasEqualDimensions( other ); }
	bool operator!=( const CBlobDesc& other ) const { return !( *this == other ); }

	// The maximum possible sequence length for a recurrent network
	int BatchLength() const { return dimensions[BD_BatchLength]; }
	// The number of sequences in the blob
	int BatchWidth() const { return dimensions[BD_BatchWidth]; }
	// The object list length
	int ListSize() const { return dimensions[BD_ListSize]; }
	// The "image" height
	int Height() const { return dimensions[BD_Height]; }
	// The "image" width
	int Width() const { return dimensions[BD_Width]; }
	// The "image" depth
	int Depth() const { return dimensions[BD_Depth]; }
	// The number of "color" channels
	int Channels() const { return dimensions[BD_Channels]; }
	// The blob size, in elements
	int BlobSize() const;
	// The empirically better size for this blob, in elements
	int MemorySize() const { return memorySize; }
	// The size of one object in the blob
	int ObjectSize() const { return Height() * Width() * Depth() * Channels(); }
	// The number of objects in the blob
	int ObjectCount() const { return BatchLength() * BatchWidth() * ListSize(); }
	// The geometrical dimensions of one object
	int GeometricalSize() const { return Height() * Width() * Depth(); }

	// The size of the dimension with a given index
	int DimSize( int d ) const { return dimensions[d]; }
	// Sets the size of the dimension with a given index
	void SetDimSize( int d, int size );

	// If memory size of original blob >= required, the dimensions could be reinterpreted
	bool FitForReinterpretDimensions( const CBlobDesc& other ) const;
	// Checks if the described blob has the same dimensions
	bool HasEqualDimensions( const CBlobDesc& other ) const;
	bool HasEqualDimensions( const CBlobDesc& other, std::initializer_list<int> dimensions ) const;
	// Gets the dimensions of the blob
	void GetDimSizes( int s[MaxDimensions] ) const;

	TBlobType GetDataType() const { return type; }
	void SetDataType( TBlobType dataType ) { type = dataType; }

private:
	int dimensions[MaxDimensions]{};
	TBlobType type = CT_Invalid;
	int memorySize = 0; // empirically better size for this blob, count in elements

	void setMemorySize( int blobSize ) { memorySize = ( memorySize > blobSize ) ? memorySize : blobSize; }
};

//---------------------------------------------------------------------------------------------------------------------

inline CBlobDesc::CBlobDesc( TBlobType dataType ) :
	type( dataType ),
	memorySize( 1 )
{
	for( int i = 0; i < MaxDimensions; i++ ) {
		dimensions[i] = 1;
	}
}

inline CBlobDesc::CBlobDesc( std::initializer_list<int> list ) :
	type( CT_Float ),
	memorySize( 0 )
{
	int i = BD_Count - 1;
	int j = static_cast<int>( list.size() ) - 1;

	while( i >= 0 && j >= 0 ) {
		dimensions[i] = list.begin()[j];
		i--;
		j--;
	}

	while( i >= 0 ) {
		dimensions[i] = 1;
		i--;
	}
	setMemorySize( BlobSize() );
}

inline CBlobDesc& CBlobDesc::operator=( const CBlobDesc& other )
{
	if( this != &other ) {
		for( int i = 0; i < MaxDimensions; i++ ) {
			dimensions[i] = other.dimensions[i];
		}
		type = other.type;
		setMemorySize( other.BlobSize() );
	}
	return *this;
}

inline int CBlobDesc::BlobSize() const
{
	int blobSize = 1;
	for( int i = 0; i < MaxDimensions; i++ ) {
		blobSize *= dimensions[i];
	}
	return blobSize;
}

inline void CBlobDesc::SetDimSize( int d, int size )
{
	dimensions[d] = size;
	setMemorySize( BlobSize() );
}

inline bool CBlobDesc::FitForReinterpretDimensions( const CBlobDesc& other ) const
{
	return BlobSize() <= other.MemorySize();
}

inline bool CBlobDesc::HasEqualDimensions( const CBlobDesc& other ) const
{
	for( TBlobDim d = TBlobDim( 0 ); d < BD_Count; ++d ) {
		if( dimensions[d] != other.dimensions[d] ) {
			return false;
		}
	}
	return true;
}

inline bool CBlobDesc::HasEqualDimensions( const CBlobDesc& other,
	std::initializer_list<int> otherDimensions ) const
{
	for( const int* d = otherDimensions.begin(); d < otherDimensions.end(); ++d ) {
		if( dimensions[*d] != other.dimensions[*d] ) {
			return false;
		}
	}
	return true;
}

inline void CBlobDesc::GetDimSizes( int s[MaxDimensions] ) const
{
	for( int i = 0; i < MaxDimensions; i++ ) {
		s[i] = dimensions[i];
	}
}

} // namespace NeoML
