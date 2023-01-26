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

// Layer which emulates Onnx OneHot operator
// It filles output with OneHot vectors (one element is equal to 1, the rest are zeroes)
// Has 2 inputs:
//    Indices - blob or shape-blob of any data type
//              The indices where 1 must be placed in the corresponding output vector
//              BD_Channels of this blob must be 1
//    Depth - shape-blob of single integer
//            Contains the size of output OneHot vectors
// The only output contains blob or shape-blob of any data typed (everything is derived from Indices input)
// The BD_Channels of output is equal to the value from Depth input
// All other dimensions are equal to the corresponding dimensions of Indices input
class NEOML_API COnnxOneHotLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxOneHotLayer )
public:
	explicit COnnxOneHotLayer( IMathEngine& mathEngine ) : COnnxLayerBase( mathEngine, "OnnxOneHotLayer" ) {}

	void Serialize( CArchive& archive );

private:
	void CalculateShapes() override;
	void RunOnce() override;
};

} // namespace NeoML
