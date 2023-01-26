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

// Layer which emulates Onnx Expand operator
// It broadcasts blob from the first input to shape from the second input
// First input must be a blob of any data type (inputBlobs[0])
// Second input must be an integer shape-blob from another COnnxLayerBase (inputShapeBlobs[1])
// The only output is an usual blob of the same data type as first input
class NEOML_API COnnxExpandLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxExpandLayer )
public:
	explicit COnnxExpandLayer( IMathEngine& mathEngine ) : COnnxLayerBase( mathEngine, "OnnxExpandLayer" ) {}

	// Onnx tensor layout
	// Its size determines the rank of onnx tensor
	// TensorLayout()[i] contains the blob dimension which contains i'th dimension of onnx tensor from the first input
	// It is used to determine which dimensions of the NeoML blob must be expanded
	const CFastArray<TBlobDim, 8>& TensorLayout() const { return tensorLayout; }
	CFastArray<TBlobDim, 8>& TensorLayout() { return tensorLayout; }

	void Serialize( CArchive& archive ) override;

protected:
	void CalculateShapes() override;
	void RunOnce() override;

private:
	CFastArray<TBlobDim, 8> tensorLayout;
};

} // namespace NeoML

