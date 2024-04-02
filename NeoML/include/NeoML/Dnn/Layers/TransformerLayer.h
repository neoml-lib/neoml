/* Copyright © 2017-2024 ABBYY

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
#include <NeoML/Dnn/Layers/MultiheadAttentionLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>

namespace NeoML {
class CActivationDesc;

// Transformer encoder layer
//
// Encoder from the "Attention is all you need"
// Optional layers are mentioned in (brackets)
//
// ------------------------------------------------------------
// 
// Standard architecture
// 
//     feedForwardNorm
//           |
//     feedForwardSum
//      |          |
//      |      (dropout)
//      |          |
//      |         fc2
//      |          |
//      |      (dropout)
//      |          |
//      |   fc1 + activation
//      |          |
//      +----------+
//           |
//    selfAttentionNorm
//           |
//    selfAttentionSum
//      |          |
//      |      (dropout)
//      |          |
//      |       selfAttention
//      |          |     |
//      +----------+     |
//           |           |
//       inputData  (inputMask)
// 
// ------------------------------------------------------------
// 
// Architecture with pre-normalization (if preNorm set)
// 
//     feedForwardNorm
//           |
//     feedForwardSum
//      |          |
//      |      (dropout)
//      |          |
//      |         fc2
//      |          |
//      |      (dropout)
//      |          |
//      |   fc1 + activation
//      |          |
//      +----------+
//           |
//    selfAttentionSum
//      |          |
//      |      (dropout)
//      |          |
//      |       selfAttention
//      |          |     |
//      +----------+     |
//           |           |
//    selfAttentionNorm  |
//           |           |
//       inputData  (inputMask)
// 
// ------------------------------------------------------------
//
// Inputs:
//      1. input data - float blob of size:
//          - BatchLength, Height, Width and Depth must be equal to 1
//          - BatchWidth - number of sequences in batch
//          - ListSize - length of sequences
//          - Channels - size of the elements in sequences
//      2. (optional) input mask - float blob containing 1.f or 0.f where 1.f means IGNORED object
//			if GetMaskType() == MT_OneObject:
//				- Width(seq_Q) and Channels(seq_V) must be equal to the ListSize of the first input
//				- Other dimensions must be equal to 1
//			if GetMaskType() == MT_Eltwise:
//				- Width(seq_Q) and Channels(seq_V) must be equal to the ListSize of the first input
//				- BatchWidth must be equal to the BatchWidth of the first input
//				- ListSize - equal to the number of head counts
//				- Other dimensions must be equal to 1
//				- it can be easily created with TransformerSourceMaskLayer
// Outputs:
//      1. output data - float blob of size:
//          - BatchWidth and ListSize are equal to the corresponding dims of the first input
//          - BatchLength, Height, Width and Depth are equal to 1
//          - Channels is equal to the Channels of the first input
//
class NEOML_API CTransformerEncoderLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CTransformerEncoderLayer )
public:
	explicit CTransformerEncoderLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Place of normalization layer: right after input or before feedForward as usual
	bool GetPreNorm() const { return preNorm; }
	void SetPreNorm( bool preNorm );

	// Number of heads in the self-attention
	int GetHeadCount() const { return selfAttention->GetHeadCount(); }
	void SetHeadCount( int headCount );

	// Hidden size of the attention layer
	// Must be a multiple of head count
	int GetHiddenSize() const { return selfAttention->GetHiddenSize(); }
	void SetHiddenSize( int hiddenSize );

	// Set probability of zero values in 3 dropout layers in the transformer layer
	float GetDropoutRate() const;
	void SetDropoutRate( float rate );

	// Set probability of zero values in a dropout layer in the self-attention layer
	float GetSelfAttentionDropoutRate() const { return selfAttention->GetDropoutRate(); }
	void SetSelfAttentionDropoutRate( float rate ){ selfAttention->SetDropoutRate( rate ); }

	// Sets the size of the first fully-connected layer inside of feed-forward
	int GetFeedForwardSize() const { return CheckCast<CFullyConnectedLayer>( fc1 )->GetNumberOfElements(); }
	void SetFeedForwardSize( int size );

	// Sets activation between fully-connected layers inside of feed-forward
	// ReLU by default
	void SetActivation( const CActivationDesc& param );

	// Mask used in self-attention
	// MT_OneObject by default
	void SetMaskType( CMultiheadAttentionLayer::TMaskType type );
	CMultiheadAttentionLayer::TMaskType GetMaskType() const { return selfAttention->GetMaskType(); }

protected:
	void Reshape() override;

private:
	enum TInpits {
		I_Sequence, // Input sequence
		I_Mask // Mask for attention (optional)
	};

	CPtr<CMultiheadAttentionLayer> selfAttention;
	CPtr<CDropoutLayer> dropoutSelfAttention;
	CPtr<CEltwiseSumLayer> selfAttentionSum;
	CPtr<CBaseLayer> fc1;
	CPtr<CDropoutLayer> dropoutFc1;
	CPtr<CBaseLayer> fc2;
	CPtr<CDropoutLayer> dropoutFc2;
	CPtr<CEltwiseSumLayer> feedForwardSum;

	bool preNorm = false; // if true place normalization before attention, else as usual

	void buildLayer();
	void addDropoutLayers();
	void removeDropoutLayers();
};

NEOML_API CLayerWrapper<CTransformerEncoderLayer> TransformerEncoder( int headCount, int hiddenSize,
	float dropout, int feedForwardSize, TActivationFunction activation, bool preNorm = false );

} // namespace NeoML
