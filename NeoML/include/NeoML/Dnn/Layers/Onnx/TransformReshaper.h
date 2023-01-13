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
#include <NeoML/Dnn/Layers/Onnx/BaseReshaper.h>

namespace NeoML {

// Reinterprets dimensions of inputShapeBlob without any changes to data
class NEOML_API CTransformReshaper : public CBaseReshaper {
	NEOML_DNN_LAYER( CTransformReshaper )
public:
	explicit CTransformReshaper( IMathEngine& mathEngine );

	// Puts the size of inputDim from to outputDim to
	// If from == BD_Count then the following output dimension will be 1 (default)
	void SetRule( TBlobDim inputDim, TBlobDim outputDim ) { transformInfo[outputDim] = inputDim; }
	TBlobDim GetRule( TBlobDim outputDim ) const { return transformInfo[outputDim]; }

	void Serialize( CArchive& archive );

protected:
	void CalculateShapes() override;
	void RunOnce();

private:
	CFastArray<TBlobDim, 8> transformInfo;
	CBlobDesc outputDesc;
};

} // namespace NeoML
