/* Copyright © 2023-2024 ABBYY

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

#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

struct CFavorAttentionDesc;

// Computes FAVOR normalized self-attention.
// https://arxiv.org/pdf/2009.14794.pdf.
// 
// Inputs: query, key, value
// Emulates equation: Output ~~ softmax( query * ( key )^T / normalizer ) * value
//
//         output
//           ^
//           |
//   +---------------+
//   |   F A V O R   | <-- projection matrix
//   |   Attention   |     (random features)
//   +---------------+
//    ^      ^      ^
//    |      |      |
//  query   key   value
//
class NEOML_API CFavorAttentionPerformerLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CFavorAttentionPerformerLayer )
public:
	// Possible activation kernel transformations
	enum class TAKernel { SoftMax = 0, ReLU = 1 };
	// Layer inputs numeration
	enum TInput { TI_Q = 0, TI_K = 1, TI_V = 2 };
	// Constructs a random matrix Q using
	enum class TRandomMaxrixStructMode {
		QMatrix, // QR-factorization of a random 2D-tensor
		GivensRotations // Givens random rotations
	};
	static constexpr TRandomMaxrixStructMode StructMode = TRandomMaxrixStructMode::GivensRotations;
	// For normalization of a random matrix Q use sum of rows' norms of a random matrix, or just =sqrt(dim)
	static constexpr bool Scaling = false;

	// Constructor
	CFavorAttentionPerformerLayer( IMathEngine& mathEngine, const char* name = nullptr );

	// The projection matrix columns size if it is used, or 0 if not
	// Set to 0, if the projection matrix should not be used
	int GetRandomFeaturesCount() const { return randomFeaturesCount; }
	void SetRandomFeaturesCount( int randomFeaturesCount );
	// The activation kernel transformations is used
	int GetActivationKernel() const { return static_cast<int>( activation ); }
	void SetActivationKernel( int activation );
	// The auto-regressive attention is used or not 
	bool GetCausal() const { return causal; }
	void SetCausal( bool causal );

	void Serialize( CArchive& archive ) override;

protected:
	~CFavorAttentionPerformerLayer();

	// Create output blobs using the input blobs
	void Reshape() override;
	// One step of a forward pass
	void RunOnce() override;
	// One step of a backward pass
	void BackwardOnce() override;

private:
	// Number of random features to be used
	// For SoftMax should be > 0, the random projection matrix should be applied
	int randomFeaturesCount = 0;
	TAKernel activation = TAKernel::SoftMax; // Activation Kernel type
	bool causal = false; // Auto-regressive attention or not
	CFavorAttentionDesc* desc = nullptr; // Favor Attention desctiption

	void destroyFavorAttentionDesc();
};

NEOML_API CLayerWrapper<CFavorAttentionPerformerLayer> FavorAttentionPerformer(
	int randomFeaturesCount, int activation, bool causal );

} // namespace NeoML
