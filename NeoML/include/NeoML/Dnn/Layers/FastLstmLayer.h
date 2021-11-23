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

// Long-Short-Term-Memory (LSTM) recurrent layer is a composite layer (consists of several layers).
class NEOML_API CFastLstmLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CFastLstmLayer )
public:
	explicit CFastLstmLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The hidden layer size
	int GetHiddenSize() const { return hiddenSize; }
	void SetHiddenSize( int size );

	// The input hidden layers weights (the blob size is (4*HiddenSize)x1x1xInputSize)
	CPtr<CDnnBlob> GetInputWeightsData() const { return inputWeights; }
	CPtr<CDnnBlob> GetInputFreeTermData() const { return inputFreeTerm; }
	void SetInputWeightsData( const CPtr<CDnnBlob>& newInputWeights ) { setWeightsData( newInputWeights, inputWeights ); }
	void SetInputFreeTermData( const CPtr<CDnnBlob>& newInputFreeTerm ) { setFreeTermData( newInputFreeTerm, inputFreeTerm ); }

	// The recurrent hidden layers weights (the blob size is (4*HiddenSize)x1x1xHiddenSize)
	CPtr<CDnnBlob> GetRecurWeightsData() const { return recurrentWeights; }
	CPtr<CDnnBlob> GetRecurFreeTermData() const { return recurrentFreeTerm; }
	void SetRecurWeightsData( const CPtr<CDnnBlob>& newRecurWeights ) { setWeightsData( newRecurWeights, recurrentWeights ); }
	void SetRecurFreeTermData( const CPtr<CDnnBlob>& newRecurFreeTerm ) { setFreeTermData( newRecurFreeTerm, recurrentFreeTerm ); }

	// The dropout rate for the hidden layer
	// Variational tied weights dropout is used (see https://arxiv.org/abs/1512.05287)
	float GetDropoutRate() const { return dropout == 0 ? 0 : dropoutRate; }
	void SetDropoutRate(float newDropoutRate);

	// The activation function to be used in the recurrent layer
	// Sigmoid by default
	TActivationFunction GetRecurrentActivation() const { return recurrentActivation; }
	void SetRecurrentActivation( TActivationFunction newActivation ) { recurrentActivation = newActivation;  }

	// Backward compatibility setting. 
	// By default, the layer returns the resetGate output. If you turn on compatibility mode, 
	// it will return the result of tanh function that is passed to the resetGate.
	bool IsInCompatibilityMode() const { return isInCompatibilityMode; }
	void SetCompatibilityMode( bool newCompatibilityMode ) { isInCompatibilityMode = newCompatibilityMode; }

	// Indicates that the sequence is processed in reverse order
	bool IsReverseSequence() const { return isReverseSequence; }
	void SetReverseSequence( bool _isReverseSequense );

	// A virtual method that creates output blobs using the input blobs
	virtual void Reshape() override;
	// A virtual method that implements one step of a forward pass
	virtual void RunOnce() override;
	// A virtual method that implements one step of a backward pass
	virtual void BackwardOnce() override;
private:
	// The gate numbers for the hidden layer output
	enum TGateOut {
		G_Main = 0,	// The main (data) output
		G_Forget,	// Forget gate
		G_Input,	// Input gate
		G_Reset,	// Reset gate

		G_Count
	};

	int hiddenSize;

	CPtr<CDnnBlob> inputWeights;
	CPtr<CDnnBlob> recurrentWeights;
	CPtr<CDnnBlob> inputFreeTerm;
	CPtr<CDnnBlob> recurrentFreeTerm;

	CPtr<CDnnBlob> mainBacklink;
	CPtr<CDnnBlob> stateBacklink;

	CPtr<CDnnBlob> dropout;
	float dropoutRate;

	TActivationFunction recurrentActivation;
	bool isInCompatibilityMode;
	bool isReverseSequence;

	static void setWeightsData( const CPtr<CDnnBlob>& src, CPtr<CDnnBlob>& dst );
	static void setFreeTermData( const CPtr<CDnnBlob>& src, CPtr<CDnnBlob>& dst );

};

NEOML_API CLayerWrapper<CFastLstmLayer> FastLstm(
	int hiddenSize, float dropoutRate, bool isInCompatibilityMode = false );

} // namespace NeoML
