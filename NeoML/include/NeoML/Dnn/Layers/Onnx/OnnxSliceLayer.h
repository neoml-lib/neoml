/* Copyright © 2017-2023 ABBYY Production LLC

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
#include <NeoML/Dnn/Layers/Onnx/OnnxLayerBase.h>

namespace NeoML {

// Layer which emulates Onnx Slice operator
// Has from 3 up to 5 inputs
//    1. Data - blob or shape-blob of any data type, which will be sliced
//    2. Starts - coordinates of starts of slices, integer shape-blob
//                Its' size is equal to the number of slices
//    3. Ends - coordinates of ends of slices, integer shape-blob
//              Its' size is equal to the number of slices
//    4. Axes - [optional] axes indices along which each of slices should be performed, integer shape-blob
//              Its' size is equal to the number of slices
//              [0, 1, 2, ..., N-1] where N is the size of Starts and Ends
//              Indices are relative to Onnx dim number and layout
//    5. Steps - [optional] step of each slice performed
//              Its' size is equal to the number of slices
// Has 1 output: sliced blob or shape of any data type (the same as Data input)

// Onnx sometimes makes slice of size 0 (Starts[i] == Ends[i])
// In that case CDnnBlob will be allocated of wrong size and special flag will be set (DoesOutputHaveElements())
class NEOML_API COnnxSliceLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxSliceLayer )
public:
	explicit COnnxSliceLayer( IMathEngine& mathEngine ) : COnnxLayerBase( mathEngine, "OnnxSliceLayer" ),
		outputHasElements( true ) {}

	// Input onnx tensor layout
	// Its size determines the rank of the tensor
	// TensorLayout()[i] contains the blob dimension which contains i'th dimension of Onnx tensor
	// It's used for determining which blob dims are sliced, because Axes input are 
	const CFastArray<TBlobDim, 8>& TensorLayout() const { return tensorLayout; }
	CFastArray<TBlobDim, 8>& TensorLayout() { return tensorLayout; }

	// Returns true if output is not a tensor with 0 elements (any of dims is 0)
	// Returns false otherwise
	// This can be called during the CBaseLayer::Reshape of layers after this one
	bool DoesOutputHaveElements() const { return outputHasElements; }

	void Serialize( CArchive& archive );

protected:
	void CalculateShapes() override;
	void RunOnce() override;

private:
	CFastArray<TBlobDim, 8> tensorLayout;
	bool outputHasElements;

	int getSliceCount() const;
	TBlobDim getAxis( int index ) const;
	int getStart( int index, int dimSize ) const;
	int getEnd( int index, int dimSize ) const;
	int getStep( int index ) const;
	CBlobDesc sliceDesc( const CBlobDesc& inputDesc ) const;
	void sliceBlob( const CDnnBlob& inputBlob, CDnnBlob& output ) const;
};

} // namespace NeoML
