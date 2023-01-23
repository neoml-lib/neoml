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
#include <NeoML/Dnn/Layers/Onnx/OnnxLayerBase.h>

namespace NeoML {

// Layer which emulates Onnx Reshape operator
// It reshapes blob from the first input according to the shape from the second input
// The first input may be a blob or shape-blob of any data type
// The second input must be a shape-blob
// The only output contains transformed first input (blob or shape-blob of the same data type)
class NEOML_API COnnxReshapeLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxReshapeLayer )
public:
	explicit COnnxReshapeLayer( IMathEngine& mathEngine ) : COnnxLayerBase( mathEngine, "OnnxReshapeLayer" ) {}

	// Onnx tensor layout
	// Its size determines the rank of the tensor
	// TensorLayout()[i] contains the blob dimension which contains i'th dimension of onnx tensor

	// Number of dimensions and their location in CDnnBlob of the first input
	const CFastArray<TBlobDim, 8>& InputLayout() const { return inputLayout; }
	CFastArray<TBlobDim, 8>& InputLayout() { return inputLayout; }

	// Number of dimensions and their location in CDnnBlob of the output
	const CFastArray<TBlobDim, 8>& OutputLayout() const { return outputLayout; }
	CFastArray<TBlobDim, 8>& OutputLayout() { return outputLayout; }

	void Serialize( CArchive& archive );

protected:
	void CalculateShapes() override;
	void RunOnce() override;

private:
	CFastArray<TBlobDim, 8> inputLayout;
	CFastArray<TBlobDim, 8> outputLayout;
};

} // namespace NeoML