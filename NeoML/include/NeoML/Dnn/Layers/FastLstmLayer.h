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
	// If makeCopy == true, input data will be copied internally and user can do with his data what he whant.
	CPtr<CDnnBlob> GetInputWeightsData() const { return getData( inputWeights() ); }
	CPtr<CDnnBlob> GetInputFreeTermData() const { return getData( inputFreeTerm() ); }
	void SetInputWeightsData( CDnnBlob* newInputWeights, bool makeCopy = true ) { setData( inputWeights(), newInputWeights, makeCopy ); }
	void SetInputFreeTermData( CDnnBlob* newInputFreeTerm, bool makeCopy = true ) { setData( inputFreeTerm(), newInputFreeTerm, makeCopy ); }

	// The recurrent hidden layers weights (the blob size is (4*HiddenSize)x1x1xHiddenSize)
	CPtr<CDnnBlob> GetRecurWeightsData() const { return getData( recurrentWeights() ); }
	CPtr<CDnnBlob> GetRecurFreeTermData() const { return getData( recurrentFreeTerm() ); }
	void SetRecurWeightsData( CDnnBlob* newRecurWeights, bool makeCopy = true ) { setData( recurrentWeights(), newRecurWeights, makeCopy ); }
	void SetRecurFreeTermData( CDnnBlob* newRecurFreeTerm, bool makeCopy = true ) { setData( recurrentFreeTerm(), newRecurFreeTerm, makeCopy ); }

	// The dropout rate for the hidden layer
	// Variational tied weights dropout is used (see https://arxiv.org/abs/1512.05287)
	float GetDropoutRate() const { return useDropout ? dropoutRate : 0; }
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

	// If second output of layer is set this CPtr will just point to outputBlobs[1]
	CPtr<CDnnBlob> stateBacklinkBlob;

	bool useDropout;
	float dropoutRate;
	CDropoutDesc* dropoutDesc;

	TActivationFunction recurrentActivation;
	bool isInCompatibilityMode;
	bool isReverseSequence;

	void initWeightAndFreeTerm( CDnnBlob* weight, CDnnBlob* freeTerm, int inputIndex, size_t objectSize );

	void setData( CPtr<CDnnBlob>& dst, CDnnBlob* src, bool makeCopy );
	CPtr<CDnnBlob> getData( const CPtr<CDnnBlob>& data ) const;

	void dropoutRunOnce( const CPtr<CDnnBlob>& src, CPtr<CDnnBlob>& dst );
	void dropoutBackwardOnce( const CPtr<CDnnBlob>& src, CPtr<CDnnBlob>& dst );
	void initBacklinkBlobs();
	void fullyconnectedRunOnce( const CDnnBlob* input, const CDnnBlob* weights, CDnnBlob* output, CDnnBlob* freeTerm );

	void processRestOfLstm( CDnnBlob* inputFullyConnectedResult, CDnnBlob* reccurentFullyConnectedResult,
		int inputPos, int outputPos );

	CPtr<CDnnBlob>& inputWeights() { return paramBlobs[0]; }
	CPtr<CDnnBlob>& inputFreeTerm() { return paramBlobs[1]; }
	CPtr<CDnnBlob>& recurrentWeights() { return paramBlobs[2]; }
	CPtr<CDnnBlob>& recurrentFreeTerm() { return paramBlobs[3]; }

	const CPtr<CDnnBlob>& inputWeights() const { return paramBlobs[0]; }
	const CPtr<CDnnBlob>& inputFreeTerm() const { return paramBlobs[1]; }
	const CPtr<CDnnBlob>& recurrentWeights() const { return paramBlobs[2]; }
	const CPtr<CDnnBlob>& recurrentFreeTerm() const { return paramBlobs[3]; }

	CPtr<CDnnBlob>& mainBacklink() { return outputBlobs[0]; }
	CPtr<CDnnBlob>& stateBacklink() { return stateBacklinkBlob; }

	const CPtr<CDnnBlob>& mainBacklink() const { return outputBlobs[0]; }
	const CPtr<CDnnBlob>& stateBacklink() const { return stateBacklinkBlob; }

};

NEOML_API CLayerWrapper<CFastLstmLayer> FastLstm(
	int hiddenSize, float dropoutRate, bool isInCompatibilityMode = false );

} // namespace NeoML
