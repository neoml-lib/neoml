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

// Layer which emulates Onnx NonZero operator
// Accepts shape-blob of any data type as the only input
// Returns usual integer blob as the only output
// BD_BatchLength of output is equal to the number of nonzero elements
// BD_BathcWidth of output is equal to the number of dimensions of input Onnx tensor
class NEOML_API COnnxNonZeroLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxNonZeroLayer )
public:
	explicit COnnxNonZeroLayer( IMathEngine& mathEngine ) : COnnxLayerBase( mathEngine, "OnnxNonZeroLayer" ) {}

	CFastArray<TBlobDim, 8>& InputLayout() { return inputLayout; }
	const CFastArray<TBlobDim, 8>& InputLayout() const { return inputLayout; }

	void Serialize( CArchive& archive );

protected:
	void CalculateShapes() override;
	void RunOnce() override;

private:
	CFastArray<TBlobDim, 8> inputLayout;
};

} // namespace NeoML
