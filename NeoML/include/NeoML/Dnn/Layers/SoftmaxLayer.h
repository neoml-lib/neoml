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
#include <NeoML/Dnn/Layers/BaseInPlaceLayer.h>

namespace NeoML {

// Implements the layer that calculates the softmax function. 
// Note that if you are using cross-entropy loss function you may set softmax to be calculated there and will not need this layer.
class NEOML_API CSoftmaxLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CSoftmaxLayer )
public:
	explicit CSoftmaxLayer( IMathEngine& mathEngine );

	// The area (dimensions) over which the values are normalized
	enum TNormalizationArea {
		NA_ObjectSize = 0,
		NA_BatchLength,
		NA_ListSize,
		NA_Channel,

		NA_Count
	};

	void SetNormalizationArea( TNormalizationArea newArea ) { area = newArea; }
	TNormalizationArea GetNormalizationArea() const { return area; }

	void Serialize( CArchive& archive ) override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;

private:
	TNormalizationArea area; // the normalization area
};

} // namespace NeoML
