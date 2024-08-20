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

// Layer which emulates Onnx ConstantOfShape operator
// It expects single integer shape-blob as input
// It's only output contains usual blob where:
//    1. First N dimensions are equal to the shape from first input
//    2. Value and data type are derived from value blob (see SetValue and GetValue)
class NEOML_API COnnxConstantOfShapeLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxConstantOfShapeLayer )
public:
	explicit COnnxConstantOfShapeLayer( IMathEngine& mathEngine );

	// The value used for filling the output (its type determines the type of output)
	void SetValue( const CDnnBlob& blob );
	const CDnnBlob& GetValue() const { return *value; }

	void Serialize( CArchive& archive ) override;

protected:
	void CalculateShapes() override;
	void RunOnce() override;

private:
	CPtr<CDnnBlob> value;
};

} // namespace NeoML
