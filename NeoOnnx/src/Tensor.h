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

// Tensor shape (in onnx notation)
typedef CFastArray<int, 8> CTensorShape;

// Match between onnx tensor axes and NeoML dimensions
typedef CFastArray<TBlobDim, 8> CTensorDim;

// Tensor in onnx graph
struct CTensor {
	// Shape in onnx
	// Has variable amount of dimensions
	CTensorShape Shape;

	// Tensor data
	// nullptr if data can't be pre-calcualated (it depends on data, provided by user)
	// It's stored in order of onnx dimensions (independent of its NeoML names)
	CPtr<CDnnBlob> Data;

	CTensor() : Data( nullptr ) {}
	CTensor( const CTensor& other );
	CTensor& operator=( const CTensor& other );

	// Sets NeoML dimensions of the tensor
	// Returns true if there is no conflicts
	bool SetTensorDim( const CTensorDim& newDim );
};

// --------------------------------------------------------------------------------------------------------------------

inline CTensor::CTensor( const CTensor& other ) :
	Data( other.Data )
{
	other.Shape.CopyTo( Shape );
}

inline CTensor& CTensor::operator=( const CTensor &other )
{
	if( this != &other ) {
		Data = other.Data;
		other.Shape.CopyTo( Shape );
	}

	return *this;
}

inline bool SetTensorDim( const CTensorShape& shape, const CTensorDim& newDim, CTensorDim& dim )
{
	if( dim.IsEmpty() ) {
		if( newDim.Size() == shape.Size() ) {
			// It's the first request for a match
			// And the number of dimensions is matching with the shape
			newDim.CopyTo( dim );
			return true;
		}
		// Dimensions number mismatch
		return false;
	}

	if( newDim.Size() != dim.Size() ) {
		// Dimensions number mismatch
		return false;
	}

	for( int dimIndex = 0; dimIndex < newDim.Size(); ++dimIndex ) {
		if( dim[dimIndex] != newDim[dimIndex] ) {
			// Supposed dimensions doesn't match with previously set one
			return false;
		}
	}

	// Number of dimensions and their values match
	return true;
}

} // namespace NeoOnnx
