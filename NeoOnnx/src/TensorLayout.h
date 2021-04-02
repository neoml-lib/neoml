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

namespace NeoOnnx {

// Desribes how tensor is represented in memory
// tensorLayout[i] is the blob dimension, where i'th onnx axis is located
class CTensorLayout: public CFastArray<TBlobDim, 8> {
public:
	CTensorLayout() {}
	explicit CTensorLayout( int dimCount );
	CTensorLayout( std::initializer_list<TBlobDim> list ) : CFastArray<TBlobDim, 8>( list ) {}
	CTensorLayout( const CTensorLayout& other ) { other.CopyTo( *this ); }

	CTensorLayout& operator=( std::initializer_list<TBlobDim> list );
	CTensorLayout& operator=( const CTensorLayout& other );

	bool operator==( const CTensorLayout& other ) const;
	bool operator!=( const CTensorLayout& other ) const { return !operator==( other ); }
};

inline CTensorLayout::CTensorLayout( int dimCount )
{
	SetSize( dimCount );
	for( int i = 0; i < dimCount; ++i ) {
		operator[](i) = static_cast<TBlobDim>( i );
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

} // namespace NeoOnnx
