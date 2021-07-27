/* Copyright © 2017-2021 ABBYY Production LLC

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

// Activation layer which calculates error function (erf)
// https://en.cppreference.com/w/cpp/numeric/math/erf
class NEOML_API CErfLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CErfLayer )
public:
	explicit CErfLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
};

NEOML_API CLayerWrapper<CErfLayer> Erf();

} // namespace NeoML
