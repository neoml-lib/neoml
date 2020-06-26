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
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>

namespace NeoML {

// Long-Short-Term-Memory (LSTM) recurrent layer is a composite layer (consists of several layers).
class NEOML_API CLstmLayer : public CRecurrentLayer {
	NEOML_DNN_LAYER( CLstmLayer )
public:
	explicit CLstmLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The hidden layer size
	int GetHiddenSize() const { return inputHiddenLayer->GetNumberOfElements() / G_Count; }
	void SetHiddenSize(int size);

	// The input hidden layers weights (the blob size is (4*HiddenSize)x1x1xInputSize)
	CPtr<CDnnBlob> GetInputWeigthsData() const { return inputHiddenLayer->GetWeightsData(); }
	CPtr<CDnnBlob> GetInputFreeTermData() const { return inputHiddenLayer->GetFreeTermData(); }
	void SetInputWeightsData( const CPtr<CDnnBlob>& inputWeights ) { inputHiddenLayer->SetWeightsData( inputWeights ); }
	void SetInputFreeTermData( const CPtr<CDnnBlob>& inputFreeTerm ) { inputHiddenLayer->SetFreeTermData( inputFreeTerm ); }

	// The recurrent hidden layers weights (the blob size is (4*HiddenSize)x1x1xHiddenSize)
	CPtr<CDnnBlob> GetRecurWeigthsData() const { return recurHiddenLayer->GetWeightsData(); }
	CPtr<CDnnBlob> GetRecurFreeTermData() const { return recurHiddenLayer->GetFreeTermData(); }
	void SetRecurWeightsData( const CPtr<CDnnBlob>& recurWeights ) { recurHiddenLayer->SetWeightsData( recurWeights ); }
	void SetRecurFreeTermData( const CPtr<CDnnBlob>& recurFreeTerm ) { recurHiddenLayer->SetFreeTermData( recurFreeTerm ); }

	// The dropout rate for the hidden layer
	// Variational tied weights dropout is used (see https://arxiv.org/abs/1512.05287)
	float GetDropoutRate() const { return inputDropoutLayer == 0 ? 0 : inputDropoutLayer->GetDropoutRate(); }
	void SetDropoutRate(float newDropoutRate);

	// The activation function to be used in the recurrent layer
	// Sigmoid by default
	TActivationFunction GetRecurrentActivation() const { return recurrentActivation; }
	void SetRecurrentActivation( TActivationFunction newActivation );

	// Backward compatibility setting. 
	// By default, the layer returns the resetGate output. If you turn on compatibility mode, 
	// it will return the result of tanh function that is passed to the resetGate.
	bool IsInCompatibilityMode() const { return isInCompatibilityMode; }
	void SetCompatibilityMode( bool compatibilityMode );

private:
	// The gate numbers for the hidden layer output
	enum TGateOut {
		G_Main = 0,	// The main (data) output
		G_Forget,	// Forget gate
		G_Input,	// Input gate
		G_Reset,	// Reset gate

		G_Count
	};

	CPtr<CFullyConnectedLayer> inputHiddenLayer;
	CPtr<CFullyConnectedLayer> recurHiddenLayer;

	CPtr<CDropoutLayer> inputDropoutLayer;
	CPtr<CDropoutLayer> recurDropoutLayer;

	CPtr<CSplitChannelsLayer> splitLayer;
	CPtr<CBackLinkLayer> mainBackLink;
	CPtr<CBackLinkLayer> stateBackLink;
	CPtr<CTanhLayer> outputTanh;
	CPtr<CEltwiseMulLayer> resetGate;

	TActivationFunction recurrentActivation;
	bool isInCompatibilityMode;

	void buildLayer(float dropout);
	void setWeightsData(const CPtr<CDnnBlob>& newWeigths);
};

} // namespace NeoML
