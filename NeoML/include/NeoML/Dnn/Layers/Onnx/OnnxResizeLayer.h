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
#include <NeoML/Dnn/Layers/InterpolationLayer.h>

namespace NeoML {

// Layer which emulates Onnx Resize operator
// It is a wrapper around CInterpolationLayer with Onnx-friendly interface
// It has 2 inputs
//    Data - an usual float blob
//           All the dimensions not mentioned in TensorLayout must be 1
//    NewShape - shape-blob of any data type of total size equal to TensorLayout().Size()
// The only output contains blob filled with interpolation
// If NewShape is an integer shape-blob then OutputShape[TensorLayout[i]] == NewShape[i]
// If NewShape is a float shape-blob then OutputShape[TensorLayout[i]] == NewShape[i] * InputShape[TensorLayout[i]]
// Output dimensions not mentioned in TensorLayout are equal to 1
class NEOML_API COnnxResizeLayer : public CInterpolationLayer {
	NEOML_DNN_LAYER( COnnxResizeLayer )
public:
	explicit COnnxResizeLayer( IMathEngine& mathEngine ) : CInterpolationLayer( mathEngine ) {}

	const CFastArray<TBlobDim, 8>& TensorLayout() const { return tensorLayout; }
	CFastArray<TBlobDim, 8>& TensorLayout() { return tensorLayout; }

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;

private:
	CFastArray<TBlobDim, 8> tensorLayout;
};

} // namespace NeoML
