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

// Determines type of the dimensions in tensor
enum TDimType
{
	// Onnx dimensions
	// Unlike NeoML's, Onnx tensor axes don't have specific meaning (channels, height, batch, etc.).
	// In memory n-dimensional Onnx tensor is represented by a blob with first n dimensions used
	// in order of TBlobDim enumberation
	// That's the default NeoOnnx pack
	DT_Onnx,

	// NeoML dimensions
	// That means if tensor contains batch of 2-dimensional images
	// it uses BD_BatchWidth, BD_Height, BD_Width and BD_Channels
	// NeoML layers, whose behavior depends on the exact dimension sizes
	// will use tensors with these dimensions (Conv, Pool, Softmax, etc.)
	DT_NeoML,

	DT_Count
};

// Tensor dimensions order
typedef CFastArray<TBlobDim, 8> CDimOrder;

// Desribes how tensor is represented in memory
struct CTensorLayout
{
	// Tensor dimensions type
	TDimType DimType;

	// The order of dimensions of Onnx tensor
	// Empty when DimType != DT_NeoML
	// When DimType == DT_NeoML OnnxOrder.Size() is equal to the number of dimensions of
	// corresponding onnx tensor and OnnxOrder[i] shows which of the TBlobDim should be
	// of i'th position in order to restore original Onnx tensor
	CDimOrder OnnxOrder;

	CTensorLayout() : DimType( DT_Onnx ) {}
	explicit CTensorLayout( const CDimOrder& onnxOrder ) :
		DimType( onnxOrder.IsEmpty() ? DT_Onnx : DT_NeoML ) { onnxOrder.CopyTo( OnnxOrder ); }

	explicit CTensorLayout( const CTensorLayout& other ) :
		DimType( other.DimType ) { other.OnnxOrder.CopyTo( OnnxOrder ); }

	CTensorLayout& operator=( const CTensorLayout& other );

	bool operator==( const CTensorLayout& other ) const;
};

inline CTensorLayout& CTensorLayout::operator=( const CTensorLayout& other )
{
	if( this != &other ) {
		DimType = other.DimType;
		other.OnnxOrder.CopyTo( OnnxOrder );
	}

	return *this;
}

inline bool CTensorLayout::operator==( const CTensorLayout& other ) const
{
	if( DimType != other.DimType || OnnxOrder.Size() != other.OnnxOrder.Size() ) {
		return false;
	}

	for( int dimIndex = 0; dimIndex < OnnxOrder.Size(); ++dimIndex ) {
		if( OnnxOrder[dimIndex] != other.OnnxOrder[dimIndex] ) {
			return false;
		}
	}

	return true;
}

} // namespace NeoOnnx
