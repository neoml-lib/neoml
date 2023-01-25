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

// Layer which emulates Onnx Shape operator
// Accepts shape-blob or blob as its only input
// Always returns shape-blob as its only output
class NEOML_API COnnxShapeLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxShapeLayer )
public:
	explicit COnnxShapeLayer( IMathEngine& mathEngine ) : COnnxLayerBase( mathEngine, "OnnxShapeLayer" ) {}

	void Serialize( CArchive& archive ) override;

	// Onnx tensor layout
	// Its size determines the rank of the tensor
	// TensorLayout()[i] contains the blob dimension which contains i'th dimension of Onnx tensor
	const CFastArray<TBlobDim, 8>& TensorLayout() const { return tensorLayout; }
	CFastArray<TBlobDim, 8>& TensorLayout() { return tensorLayout; }

protected:
	void CalculateShapes() override;

private:
	CFastArray<TBlobDim, 8> tensorLayout;
};

} // namespace NeoML
