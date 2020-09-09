/* Copyright © 2017-2020 ABBYY Production LLC

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

// Activation layer with formual x * Ф(x),
// where Ф(x) - cumulative distribution function for the normal distribution N(0, 1)
// This layer uses next approximation: x * sigmoid(1.702 * x)
class NEOML_API CGELULayer : public CBaseLayer {
	NEOML_DNN_LAYER( CGELULayer )
public:
	explicit CGELULayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// Constant 1.702f
	CFloatHandleVar multiplierVar;
};

} // namespace NeoML
