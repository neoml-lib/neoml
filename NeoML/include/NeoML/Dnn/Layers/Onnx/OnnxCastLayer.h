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

// Layer which emulates Onnx Cast operator
class NEOML_API COnnxCastLayer : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxCastLayer )
public:
	explicit COnnxCastLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Sets output blob type
	// CT_Float by default
	void SetOutputType( TBlobType type );
	TBlobType GetOutputType() const { return outputType; }

protected:
	// CBaseLayer methods
	void CalculateShapes() override;
	void RunOnce() override;

private:
	// output blob type
	TBlobType outputType;
};

} // namespace NeoML
