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

// Reinterprets dimensions of input blob or shape-blob without any changes to data
class NEOML_API COnnxTransformHelper : public COnnxLayerBase {
	NEOML_DNN_LAYER( COnnxTransformHelper )
public:
	explicit COnnxTransformHelper( IMathEngine& mathEngine );

	// Puts the size of input's inputDim to output's outputDim
	// If inputDim == BD_Count then the following outputDim will be set to 1 (default)
	void SetRule( TBlobDim inputDim, TBlobDim outputDim ) { transformInfo[outputDim] = inputDim; }
	TBlobDim GetRule( TBlobDim outputDim ) const { return transformInfo[outputDim]; }

	void Serialize( CArchive& archive ) override;

protected:
	void CalculateShapes() override;
	void RunOnce() override;

private:
	CFastArray<TBlobDim, 8> transformInfo;
	CBlobDesc outputDesc;
};

} // namespace NeoML
