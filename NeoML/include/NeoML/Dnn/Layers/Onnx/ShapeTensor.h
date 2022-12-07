/* Copyright © 2017-2022 ABBYY Production LLC

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

// Special tensor which is calculated during Reshape phase
// Unlike blobs it has variable number of dimensions
// It may contain only integer data and is always located in CPU RAM
class CShapeTensor {
public:
	// Default constructor builds tensor in an uninitialized state
	CShapeTensor() = default;
	CShapeTensor( const CShapeTensor& other );

	// Has tensor been initialized already
	bool IsInitialized() const { return !dims.IsEmpty() || !data.IsEmpty(); }
	// Sets the rank and size of tensor along each of dimensions
	// Initializes tensor if it was uninitialized
	void Resize( const CFastArray<int, 8>& newDims );
	// Number of dimensions
	// Tensor of rank 0 is a scalar
	int Rank() const { return dims.Size(); }
	// Returns sizes of each of tensor dimensions
	const CFastArray<int, 8>& Size() const { return dims; }
	// Returns the number of elements in tensor
	int ElementCount() const { return data.Size(); }

	// Copies current tensor to other
	void CopyTo( CShapeTensor& other ) const;

	// Accessors

	// Raw flat accessor
	// Does not support negative indices nor broadcasting
	int& operator[]( int x );
	const int& operator[]( int x ) const;

	// Multidimensional acessors
	// Negative indices are supported
	// Broadcasting is not supported
	int& At();
	int& At( int x0 );
	int& At( int x0, int x1 );
	int& At( const CFastArray<int, 8>& x );
	int At() const;
	int At( int x0 ) const;
	int At( int x0, int x1 ) const;
	int At( const CFastArray<int, 8>& x ) const;

	// Accessors with broadcasting supported (NumPy broadcasting)
	// Negative indices are not supported
	int& BroadcastingAt( const CFastArray<int, 8>& x );
	int BroadcastingAt( const CFastArray<int, 8>& x ) const;

	void Serialize( CArchive& archive );

	// Broadcasts first tensor size with second and writes the result into first
	static void BroadcastSize( CFastArray<int, 8>& first, const CFastArray<int, 8>& second );

private:
	CFastArray<int, 8> dims;
	CFastArray<int, 8> data;
};

inline CShapeTensor::CShapeTensor( const CShapeTensor& other )
{
	other.dims.CopyTo( dims );
	other.data.CopyTo( data );
}

inline void CShapeTensor::Resize( const CFastArray<int, 8>& newDims )
{
	newDims.CopyTo( dims );
	int totalSize = 1;
	for( int i = 0; i < dims.Size(); ++i ) {
		NeoPresume( dims[i] > 0 );
		totalSize *= dims[i];
	}
	data.SetSize( totalSize );
}

inline void CShapeTensor::CopyTo( CShapeTensor& other ) const
{
	NeoPresume( IsInitialized() );
	dims.CopyTo( other.dims );
	data.CopyTo( other.data );
}

inline int& CShapeTensor::operator[]( int x )
{
	NeoPresume( IsInitialized() );
	NeoPresume( data.IsValidIndex( x ) );
	return data[x];
}

inline const int& CShapeTensor::operator[]( int x ) const
{
	NeoPresume( IsInitialized() );
	NeoPresume( data.IsValidIndex( x ) );
	return data[x];
}

inline int& CShapeTensor::At()
{
	NeoPresume( IsInitialized() );
	NeoPresume( dims.IsEmpty() );
	return data[0];
}

inline int& CShapeTensor::At( int x0 )
{
	NeoPresume( IsInitialized() );
	NeoPresume( dims.Size() == 1 );
	NeoPresume( x0 >= -dims[0] );
	NeoPresume( x0 < dims[0] );
	return data[x0 < 0 ? x0 + dims[0] : x0];
}

inline int& CShapeTensor::At( int x0, int x1 )
{
	NeoPresume( IsInitialized() );
	NeoPresume( dims.Size() == 2 );
	NeoPresume( x0 >= -dims[0] );
	NeoPresume( x0 < dims[0] );
	NeoPresume( x1 >= -dims[1] );
	NeoPresume( x1 < dims[1] );
	x0 = x0 < 0 ? x0 + dims[0] : x0;
	return data[x0 * dims[1] + (x1 < 0 ? x1 + dims[1] : x1)];
}

inline int& CShapeTensor::At( const CFastArray<int, 8>& x )
{
	NeoPresume( IsInitialized() );
	NeoPresume( dims.Size() == x.Size() );
	int flatIndex = 0;
	for( int i = 0; i < x.Size(); ++i ) {
		NeoPresume( x[i] >= -dims[i] );
		NeoPresume( x[i] < dims[i] );
		if( i > 0 ) {
			flatIndex *= dims[i];
		}
		flatIndex += x[i] < 0 ? x[i] + dims[i] : x[i];
	}
	return data[flatIndex];
}

inline int CShapeTensor::At() const
{
	NeoPresume( IsInitialized() );
	NeoPresume( dims.IsEmpty() );
	return data[0];
}

inline int CShapeTensor::At( int x0 ) const
{
	NeoPresume( IsInitialized() );
	NeoPresume( dims.Size() == 1 );
	NeoPresume( x0 >= -dims[0] );
	NeoPresume( x0 < dims[0] );
	return data[x0 < 0 ? x0 + dims[0] : x0];
}

inline int CShapeTensor::At( int x0, int x1 ) const
{
	NeoPresume( IsInitialized() );
	NeoPresume( dims.Size() == 2 );
	NeoPresume( x0 >= -dims[0] );
	NeoPresume( x0 < dims[0] );
	NeoPresume( x1 >= -dims[1] );
	NeoPresume( x1 < dims[1] );
	x0 = x0 < 0 ? x0 + dims[0] : x0;
	return data[x0 * dims[1] + (x1 < 0 ? x1 + dims[1] : x1)];
}

inline int CShapeTensor::At( const CFastArray<int, 8>& x ) const
{
	NeoPresume( IsInitialized() );
	NeoPresume( dims.Size() == x.Size() );
	int flatIndex = 0;
	for( int i = 0; i < x.Size(); ++i ) {
		NeoPresume( x[i] >= -dims[i] );
		NeoPresume( x[i] < dims[i]);
		if( i > 0 ) {
			flatIndex *= dims[i];
		}
		flatIndex += x[i] < 0 ? x[i] + dims[i] : x[i];
	}
	return data[flatIndex];
}

inline int& CShapeTensor::BroadcastingAt( const CFastArray<int, 8>& x )
{
	NeoPresume( IsInitialized() );
	NeoPresume( dims.Size() <= x.Size() );
	int offsetPerElem = 1;
	int flatIndex = 0;
	for( int shift = -1; shift >= -dims.Size(); --shift ) {
		const int coord = x[x.Size() + shift];
		const int dimSize = dims[dims.Size() + shift];
		NeoPresume( coord >= -dimSize );
		flatIndex += ( ( coord + dimSize ) % dimSize ) * offsetPerElem;
		offsetPerElem *= dimSize;
	}
	NeoPresume( flatIndex >= 0 );
	NeoPresume( flatIndex < data.Size() );
	return data[flatIndex];
}

inline int CShapeTensor::BroadcastingAt( const CFastArray<int, 8>& x ) const
{
	NeoPresume( IsInitialized() );
	NeoPresume( dims.Size() <= x.Size() );
	int offsetPerElem = 1;
	int flatIndex = 0;
	for( int shift = -1; shift >= -dims.Size(); --shift ) {
		const int dimSize = dims[dims.Size() + shift];
		NeoPresume( x[x.Size() + shift] >= 0 );
		flatIndex += ( x[x.Size() + shift] % dimSize ) * offsetPerElem;
		offsetPerElem *= dimSize;
	}
	NeoPresume( flatIndex >= 0 );
	NeoPresume( flatIndex < data.Size() );
	return data[flatIndex];
}

inline void CShapeTensor::Serialize( CArchive& archive )
{
	dims.Serialize( archive );
	data.Serialize( archive );
}

inline void CShapeTensor::BroadcastSize( CFastArray<int, 8>& first, const CFastArray<int, 8>& second )
{
	if( second.Size() > first.Size() ) {
		first.InsertAt( 1, 0, second.Size() - first.Size() );
	}

	for( int shift = -1; shift >= -second.Size(); --shift ) {
		int& firstSize = first[first.Size() + shift];
		int secondSize = second[second.Size() + shift];
		NeoPresume( firstSize == 1 || secondSize == 1 || firstSize == secondSize );
		firstSize = std::max<int>( firstSize, secondSize );
	}
}

} // namespace NeoML
