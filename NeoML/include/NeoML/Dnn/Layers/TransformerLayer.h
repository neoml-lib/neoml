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
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/MultiheadAttentionLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>

namespace NeoML {

// Transformer encoder layer
//
// Encoder from the "Attention is all you need"
// Optional layers are mentioned in (brackets)
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
// Inputs:
//      1. input data - float blob of size:
//          - BatchLength, Height, Width and Depth must be equal to 1
//          - BatchWidth - number of sequences in batch
//          - ListSize - length of sequences
//          - Channels - size of the elements in sequences
//      2. (optional) input mask - float blob containing 1.f or 0.f where 1.f means IGNORED object
//          - Width(seq_Q) and Channels(seq_V) must be equal to the ListSize of the first input
//          - Other dimensions must be equal to 1
// Outputs:
//      1. output data - float blob of size:
//          - BatchWidth and ListSize are equal to the corresponding dims of the first input
//          - BatchLength, Height, Width and Depth are equal to 1
//          - Channels is equal to the Channels of the first input
class NEOML_API CTransformerEncoderLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CTransformerEncoderLayer )
public:
	explicit CTransformerEncoderLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Number of heads in the self-attention
	int GetHeadCount() const { return selfAttention->GetHeadCount(); }
	void SetHeadCount( int headCount );

	// Hidden size of the attention layer
	// Must be a multiple of head count
	int GetHiddenSize() const { return selfAttention->GetHiddenSize(); }
	void SetHiddenSize( int hiddenSize );

	// Dropout rate
	float GetDropoutRate() const;
	void SetDropoutRate( float rate );

	// Sets the size of the first fully-connected layer inside of feed-forward
	int GetFeedForwardSize() const { return fc1->GetNumberOfElements(); }
	void SetFeedForwardSize( int size );

	// Sets activation between fully-connected layers inside of feed-forward
	// ReLU by default
	void SetActivation( TActivationFunction newFunction );

protected:
	void Reshape() override;

private:
	CPtr<CMultiheadAttentionLayer> selfAttention;
	CPtr<CDropoutLayer> dropoutSelfAttention;
	CPtr<CEltwiseSumLayer> selfAttentionSum;
	CPtr<CFullyConnectedLayer> fc1;
	CPtr<CDropoutLayer> dropoutFc1;
	CPtr<CFullyConnectedLayer> fc2;
	CPtr<CDropoutLayer> dropoutFc2;
	CPtr<CEltwiseSumLayer> feedForwardSum;

	void buildLayer();
	void addDropoutLayers();
	void removeDropoutLayers();
};

NEOML_API CLayerWrapper<CTransformerEncoderLayer> TransformerEncoder( int headCount, int hiddenSize,
	float dropout, int feedForwardSize, TActivationFunction activation );

// --------------------------------------------------------------------------------------------------------------------

// Transformer decoder layer
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
//   mheadAttentionNorm
//           |
//   mheadAttentionSum
//      |          |
//      |      (dropout)
//      |          |
//      |           mheadAttention
//      |          |     |        |
//      |          | encoderData (encoderMask)
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
// Inputs:
//      1. input data - float blob of size:
//          - BatchLength, Height, Width and Depth must be equal to 1
//          - BatchWidth - number of sequences in batch
//          - ListSize - length of output sequences
//          - Channels - size of the elements in input sequences
//      2. encoder data - float blob of size:
//          - BatchLength, Height, Width and Depth must be equal to 1
//          - BatchWidth must be equal to the BatchWidth of the first input
//          - ListSize - length of encoder sequences
//          - Channels - size of the elements in encoder sequences
//      3. (optional) input mask - float blob containing 1.f or 0.f where 1.f means that object will be IGNORED
//          - Width(seq_Q) and Channels(seq_V) must be equal to the ListSize of the first input
//          - Other dimensions must be equal to 1
//      4. (optional) encoder mask - float blob containing 1.f or 0.f where 1.f means that object will be IGNORED
//          - Width must be equal to the ListSize of the first input
//          - Channels must be equal to the ListSize of the second input
//          - Other dimensions must be equal to 1
// Outputs:
//      1. output data - float blob of size:
//          - BatchWidth and ListSize are equal to the corresponding dims of the first input
//          - BatchLength, Height, Width and Depth are equal to 1
//          - Channels is equal to the Channels of the first input
class NEOML_API CTransformerDecoderLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CTransformerDecoderLayer )
public:
	explicit CTransformerDecoderLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Number of heads in the attention layers
	int GetHeadCount() const;
	void SetHeadCount( int headCount );

	// Hidden size of the attentions layer
	// Must be a multiple of head count
	int GetHiddenSize() const;
	void SetHiddenSize( int hiddenSize );

	// Dropout rate
	float GetDropoutRate() const;
	void SetDropoutRate( float rate );

	// Sets the size of the first fully-connected layer inside of feed-forward
	int GetFeedForwardSize() const { return fc1->GetNumberOfElements(); }
	void SetFeedForwardSize( int size );

	// Sets activation between fully-connected layers inside of feed-forward
	// ReLU by default
	void SetActivation( TActivationFunction newFunction );

protected:
	void Reshape() override;

private:
	CPtr<CMultiheadAttentionLayer> selfAttention;
	CPtr<CDropoutLayer> dropoutSelfAttention;
	CPtr<CEltwiseSumLayer> selfAttentionSum;
	CPtr<CMultiheadAttentionLayer> mheadAttention;
	CPtr<CDropoutLayer> dropoutMheadAttention;
	CPtr<CEltwiseSumLayer> mheadAttentionSum;
	CPtr<CFullyConnectedLayer> fc1;
	CPtr<CDropoutLayer> dropoutFc1;
	CPtr<CFullyConnectedLayer> fc2;
	CPtr<CDropoutLayer> dropoutFc2;
	CPtr<CEltwiseSumLayer> feedForwardSum;

	void buildLayer();
	void addDropoutLayers();
	void removeDropoutLayers();
};

NEOML_API CLayerWrapper<CTransformerDecoderLayer> TransformerDecoder( int headCount, int hiddenSize,
	float dropout, int feedForwardSize, TActivationFunction activation );

} // namespace NeoML
