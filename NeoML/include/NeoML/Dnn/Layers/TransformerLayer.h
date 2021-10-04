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

// Forward declaration(s)

// Transformer-layer
//
//     feedForwardNorm
//           |
//     feedForwardSum
//      |          |
//      |    fc2 (with optional dropout)
//      |          |
//      |    (with optional dropout)
//      |    fc1 with activation
//      |          |
//      +----------+
//           |
//      attentionNorm
//           |
//      attentionSum
//      |          |
//      |    selfAttention
//      |          |
//      +----------+
//           |
//       inputData
//
class NEOML_API CTransformerLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CTransformerLayer )
public:
	explicit CTransformerLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Number of heads in the self-attention
	int GetHeadCount() const { return selfAttention->GetHeadCount(); }
	void SetHeadCount( int headCount );

	// Hidden size of the attention layer
	int GetHiddenSize() const { return selfAttention->GetHiddenSize(); }
	void SetHiddenSize( int hiddenSize );

	// Size of the output of the layer (and self-attention)
	int GetOutputSize() const;
	void SetOutputSize( int outputSize );

	// Dropout rate of the self-attention
	float GetAttentionDropout() const { return selfAttention->GetDropoutRate(); }
	void SetAttentionDropout( float rate );

	// Sets the size of the first fully-connected layer inside of feed-forward
	int GetFeedForwardSize() const { return fc1->GetNumberOfElements(); }
	void SetFeedForwardSize( int size );

	// Sets activation between fully-connected layers inside of feed-forward
	// ReLU by default
	void SetActivation( TActivationFunction newFunction );

	// Dropout rate inside of feed-forward
	float GetFeedForwardDropout() const;
	void SetFeedForwardDropout( float rate );

private:
	CPtr<CMultiheadAttentionLayer> selfAttention;
	CPtr<CFullyConnectedLayer> fc1;
	CPtr<CDropoutLayer> dropout1;
	CPtr<CFullyConnectedLayer> fc2;
	CPtr<CDropoutLayer> dropout2;
	CPtr<CEltwiseSumLayer> feedForwardResidual;

	void buildLayer();
	void addDropoutLayers();
	void removeDropoutLayers();
};

NEOML_API CLayerWrapper<CTransformerLayer> Transformer( int headCount, int hiddenSize, int outputSize,
	float attentionDropout, int feedForwardSize, float feedForwardDropout, TActivationFunction activation );

} // namespace NeoML
