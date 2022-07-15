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

#include "NeoOnnxCheck.h"

namespace NeoOnnx {

// Desribes how the tensor is represented in memory
// tensorLayout[i] is the blob dimension, where i'th onnx axis is located
class CTensorLayout: public CFastArray<TBlobDim, 8> {
public:
	CTensorLayout() = default;
	// Returns default layout for dimCount-dimensional tensor
	explicit CTensorLayout( int dimCount );
	explicit CTensorLayout( std::initializer_list<TBlobDim> list ) : CFastArray<TBlobDim, 8>( list ) {}
	CTensorLayout( const CTensorLayout& other ) : CFastArray<TBlobDim, 8>() { other.CopyTo( *this ); }

	CTensorLayout& operator=( std::initializer_list<TBlobDim> list );
	CTensorLayout& operator=( const CTensorLayout& other );

	bool operator==( const CTensorLayout& other ) const;
	bool operator!=( const CTensorLayout& other ) const { return !operator==( other ); }

	static CTensorLayout IOLayout( int dimCount );
};

inline CTensorLayout::CTensorLayout( int dimCount )
{
	SetBufferSize( dimCount );
	// Next dimensions are educated guesses
	// If they'll match future layers then it will save one transform operation
	// Transpose (if needed) can't be avoided anyway
	static_assert( BD_Count == 7, "BD_Count != 7" );
	switch( dimCount ) {
		case 0:
			// dimCount == 0 means scalar
			break;
		case 1:
			Add( { BD_Channels } );
			break;
		case 2:
			Add( { BD_BatchWidth, BD_Channels } );
			break;
		case 3:
			Add( { BD_BatchLength, BD_BatchWidth, BD_Channels } );
			break;
		case 4:
			Add( { BD_BatchWidth, BD_Height, BD_Width, BD_Channels } );
			break;
		case 5:
			Add( { BD_BatchLength, BD_BatchWidth, BD_Height, BD_Width, BD_Channels } );
			break;
		case 6:
			Add( { BD_BatchLength, BD_BatchWidth, BD_Height, BD_Width, BD_Depth, BD_Channels } );
			break;
		case 7:
			Add( { BD_BatchLength, BD_BatchWidth, BD_ListSize, BD_Height, BD_Width, BD_Depth, BD_Channels } );
			break;
		default:
			CheckNeoOnnxSupport( false, "unsupported dimension count" );
			break;
	}
}

inline CTensorLayout& CTensorLayout::operator=( std::initializer_list<TBlobDim> list )
{
	CFastArray<TBlobDim, 8>::operator=( list );
	return *this;
}

inline CTensorLayout& CTensorLayout::operator=(  const CTensorLayout& other )
{
	other.CopyTo( *this );
	return *this;
}

inline bool CTensorLayout::operator==( const CTensorLayout& other ) const
{
	if( Size() != other.Size() ) {
		return false;
	}

	for( int i = 0; i < Size(); ++i ) {
		if( other[i] != ( *this )[i] ) {
			return false;
		}
	}

	return true;
}

// Returns true if data in blob is not in the same order, as it would be in an onnx tensor
inline bool IsTransposedLayout( const CTensorLayout& layout )
{
	for( int dimIndex = 0; dimIndex < layout.Size() - 1; ++dimIndex ) {
		if( layout[dimIndex] > layout[dimIndex + 1] ) {
			return true;
		}
	}
	return false;
}

// Returns layout for input and output blobs with the given number of dimensions
inline CTensorLayout CTensorLayout::IOLayout( int dimCount )
{
	CTensorLayout layout;
	layout.SetSize( dimCount );
	for( int i = 0; i < dimCount; ++i ) {
		layout[i] = static_cast<TBlobDim>( i );
	}
	return layout;
}

} // namespace NeoOnnx

