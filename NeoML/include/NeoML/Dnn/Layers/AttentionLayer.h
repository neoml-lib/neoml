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
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/RecurrentLayer.h>

namespace NeoML {

class CAttentionRecurrentLayer;
class CAttentionLayer;
class CSoftmaxLayer;
class CTanhLayer;
class CConvLayer;
class CFullyConnectedLayer;
class CSplitChannelsLayer;

///////////////////////////////////////////////////////////////////////////////////
// AttentionDecoderLayer implements a layer that converts the input sequence 
// into the output sequence, not necessarily of the same length
// The layer inputs: 
// #0 - the input sequence
// #1 - the special character used to initialize the output sequence, 
// or (when teacher forcing) the sample output sequence for training
// The layer outputs:
// #0 - the output sequence

// The estimate function type (for estimating alignment of the input and output sequences)
enum TAttentionScore {
	AS_DotProduct, // x*W*y
	AS_Additive // tanh(x*Wx + y*Wy)*v
};

class NEOML_API CAttentionDecoderLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CAttentionDecoderLayer )
public:
	explicit CAttentionDecoderLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The layer inputs
	enum TInput {
		I_InputSequence = 0,
		I_OutputInitializer = 1
	};
	// Sets the estimate function
	void SetAttentionScore( TAttentionScore newScore );

	// Gets and sets the output object size (the number of channels) and the length of the output sequence
	int GetOutputObjectSize() const;
	void SetOutputObjectSize(int outObjectSize);
	int GetOutputSequenceLen() const;
	void SetOutputSequenceLen(int outSeqLen);

	// The size of the hidden layer
	int GetHiddenLayerSize() const;
	void SetHiddenLayerSize( int size );

private:
	TAttentionScore score; // estimate function
	CPtr<CFullyConnectedLayer> initLayer;
	// The fully connected layer that accepts the input sequence
	CPtr<CFullyConnectedLayer> hiddenLayer;
	// The internal recurrent layer that performs the calculations
	CPtr<CAttentionRecurrentLayer> recurrentLayer;

	void buildLayer();
};

///////////////////////////////////////////////////////////////////////////////////
// AttentionRecurrentLayer is the internal recurrent layer that converts the input sequence into the output
// The layer inputs: 
// #0 - the transposed input sequence (BD_BatchLength <-> BD_ListSize)
// #1 - the transposed input sequence (BD_BatchLength <-> BD_ListSize), after the fully-connected layer
// #2 - the initial state of the decoder
// #3 - the special character used to initialize the output sequence, 
// or (when teacher forcing) the sample output sequence for training
// The layer outputs:
// #0 - the output sequence
class NEOML_API CAttentionRecurrentLayer : public CRecurrentLayer {
	NEOML_DNN_LAYER( CAttentionRecurrentLayer )
public:
	explicit CAttentionRecurrentLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The layer inputs
	enum TInput {
		I_InputList = 0,
		I_ProcessedInputList = 1,
		I_InitialState = 2,
		I_OutputInitializer = 3
	};
	// Sets the estimate function type
	void SetAttentionScore( TAttentionScore newScore );

	// Gets and sets the output object size (the number of channels) and the length of the output sequence
	int GetOutputObjectSize() const;
	void SetOutputObjectSize(int outObjectSize);

	// The hidden layer size (only for AS_Additive):
	void SetHiddenLayerSize( int size );

private:
	TAttentionScore score; // the estimate function
	static const CString hiddenLayerName; // the fully-connected layer name
	// The fully-connected layer to which the input sequence is passed (only for AS_Additive)
	CPtr<CFullyConnectedLayer> hiddenLayer;

	CPtr<CFullyConnectedLayer> mainLayer;
	CPtr<CFullyConnectedLayer> gateLayer;
	CPtr<CSplitChannelsLayer> splitLayer;
	CPtr<CFullyConnectedLayer> outputLayer;

	// The layer that calculates alignment
	CPtr<CAttentionLayer> attentionLayer;
	// The backward link from the decoder to attentionLayer
	CPtr<CBackLinkLayer> stateBackLink;
	// The backward link from the output to the decoder
	CPtr<CBackLinkLayer> mainBackLink;

	void buildLayer();
};

// Attention is a layer that calculates the input and output sequences alignment
// The layer inputs:
// #0 - (V) the transposed input sequence (BD_BatchLength <-> BD_ListSize)
// #1 - (K) the transposed input sequence (BD_BatchLength <-> BD_ListSize), after the fully-connected layer
// #2 - (Q) the current state of the decoder
// The layer outputs:
// #0 - the weighted total of the input sequence elements,
//		weighted by the estimated alignment with a given output sequence element
// DO NOT USE DIRECTLY (use CAttentionDecoderLayer or CMultiheadAttentionLayer)
class NEOML_API CAttentionLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CAttentionLayer )
public:
	explicit CAttentionLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The layer inputs
	enum TInput {
		I_InputList = 0,
		I_ProcessedInputList = 1,
		I_DecoderState = 2
	};
	// Sets the alignment estimating function type
	void SetAttentionScore( TAttentionScore newScore );

	// The fully-connected layer weights
	// May only be used for AS_Additive
	CPtr<CDnnBlob> GetFcWeightsData() const;
	void SetFcWeightsData( const CPtr<CDnnBlob>& newWeights );

	// The fully-connected layer free term
	// May only be used for AS_Additive
	CPtr<CDnnBlob> GetFcFreeTermData() const;
	void SetFcFreeTermData( const CPtr<CDnnBlob>& newFreeTerms );

private:
	TAttentionScore score; // the alignment estimating function
	CPtr<CFullyConnectedLayer> tanhFc; // the fully-connected layer for AS_Additive

	void buildLayer();
};

// The weighted total of the input sequence
// The layer inputs:
// #0 - the transposed input sequence  (BD_BatchLength <-> BD_ListSize, non-recurrent input)
// #1 - the recurrent input containing the weights for the input sequence elements
// The layer output #0 contains the weighted total of the input sequence elements
class NEOML_API CAttentionWeightedSumLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CAttentionWeightedSumLayer )
public:
	explicit CAttentionWeightedSumLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

// The dot product of the input and output sequences
// The layer inputs:
// #0 - the transposed input sequence  (BD_BatchLength <-> BD_ListSize, non-recurrent input)
// #1 - the output sequence element (recurrent input)
// The layer output #0 contains the transposed input sequence with each element multiplied by the output sequence element
class NEOML_API CAttentionDotProductLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CAttentionDotProductLayer )
public:
	explicit CAttentionDotProductLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

// The sum of the input and output sequences
// The layer inputs:
// #0 - the transposed input sequence  (BD_BatchLength <-> BD_ListSize, non-recurrent input)
// #1 - the output sequence element (recurrent input)
// The layer output #0 contains the transposed input sequence with the output sequence element added to each of its elements
class NEOML_API CAttentionSumLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CAttentionSumLayer )
public:
	explicit CAttentionSumLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

} // namespace NeoML
