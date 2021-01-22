/* Copyright � 2017-2020 ABBYY Production LLC

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
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// CGlobalMeanPoolingLayer implements a layer which performs mean pooling on `Height`, `Width`, and `Depth` dimensions of the input.
class NEOML_API CGlobalMeanPoolingLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CGlobalMeanPoolingLayer )
public:
	explicit CGlobalMeanPoolingLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	CPtr<CDnnBlob> coeff;
};

NEOML_API CLayerWrapper<CGlobalMeanPoolingLayer> GlobalMeanPooling();

} // namespace NeoML
