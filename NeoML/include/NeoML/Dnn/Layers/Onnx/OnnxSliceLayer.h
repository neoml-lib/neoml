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
#include <NeoML/Dnn/Layers/Onnx/BaseReshaper.h>

namespace NeoML {

// Layer which emulates Onnx Slice operator
class NEOML_API COnnxSliceLayer : public CBaseReshaper {
	NEOML_DNN_LAYER( COnnxSliceLayer )
public:
	explicit COnnxSliceLayer( IMathEngine& mathEngine ) : CBaseReshaper( mathEngine, "OnnxSliceLayer" ),
		outputHasElements( true ) {}

	// Input onnx tensor layout
	// Its size determines the rank of the tensor
	// TensorLayout()[i] contains the blob dimension which contains i'th dimension of Onnx tensor
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
	CPtr<CDnnBlob> sliceBlob( const CDnnBlob& inputBlob ) const;
};

} // namespace NeoML
