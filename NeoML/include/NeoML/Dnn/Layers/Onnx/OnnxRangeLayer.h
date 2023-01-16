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

// Layer which emulates Onnx Range operator
class NEOML_API COnnxRangeLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxRangeLayer )
public:
	explicit COnnxRangeLayer( IMathEngine& mathEngine ) : COnnxLayerBase( mathEngine, "OnnxRangeLayer" ) {}

	void Serialize( CArchive& archive );

protected:
	void CalculateShapes() override;
	void RunOnce() override;
};

} // namespace NeoML
