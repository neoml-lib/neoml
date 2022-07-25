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
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/BaseInPlaceLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// Takes single integer blob of any size as an input.
// The only output contains integer blob of the same size where
//    output[i] = input[i] == 0 ? 1 : 0
class NEOML_API CNotLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CNotLayer )
public:
	explicit CNotLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	void OnReshaped() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

NEOML_API CLayerWrapper<CNotLayer> Not();

} // namespace NeoML
