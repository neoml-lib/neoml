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
#include <NeoML/Dnn/Layers/RecurrentLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/SplitLayer.h>

namespace NeoML {

// Gated Recurrent Unit (GRU) is a recurrent layer that incorporates several layers (that is, it's a composite layer)
class NEOML_API CGruLayer : public CRecurrentLayer {
	NEOML_DNN_LAYER( CGruLayer )
public:
	explicit CGruLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The hidden layer size
	int GetHiddenSize() const { return mainLayer->GetNumberOfElements(); }
	void SetHiddenSize(int size);

	CPtr<CDnnBlob> GetMainWeightsData() const { return mainLayer->GetWeightsData(); }
	CPtr<CDnnBlob> GetMainFreeTermData() const { return mainLayer->GetFreeTermData(); }

	void SetMainWeightsData(CDnnBlob* newWeights) { mainLayer->SetWeightsData(newWeights); }
	void SetMainFreeTermData(CDnnBlob* newFreeTerm) { mainLayer->SetFreeTermData(newFreeTerm); }

	CPtr<CDnnBlob> GetGateWeightsData() const { return gateLayer->GetWeightsData(); }
	CPtr<CDnnBlob> GetGateFreeTermData() const { return gateLayer->GetFreeTermData(); }

	void SetGateWeightsData(CDnnBlob* newWeights) { gateLayer->SetWeightsData(newWeights); }
	void SetGateFreeTermData(CDnnBlob* newFreeTerm) { gateLayer->SetFreeTermData(newFreeTerm); }

private:
	// The indices of the gates in the hidden layer output
	enum TGateOut {
		G_Update = 0,	// Update gate
		G_Reset,	// Reset gate

		G_Count
	};

	CPtr<CFullyConnectedLayer> mainLayer;
	CPtr<CFullyConnectedLayer> gateLayer;
	CPtr<CSplitChannelsLayer> splitLayer;
	CPtr<CBackLinkLayer> mainBackLink;

	void buildLayer();
};

} // namespace NeoML
