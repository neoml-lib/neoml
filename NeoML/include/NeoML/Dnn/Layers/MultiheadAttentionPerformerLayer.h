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
#include <NeoML/Dnn/Layers/CompositeLayer.h>

namespace NeoML {

// Multihead Self Attention Performer
// https://arxiv.org/pdf/2009.14794.pdf
// Implementation of multiheaded FAVOR-attention & FAVOR-self-attention layers.
//
//  +----------------------+--------+-------------------------------------------------------
//  | Parameter            | Type   | Description
//  +----------------------+--------+-------------------------------------------------------
//  | HiddenSize           | int    | size of trainable matrices, output dim of hidden layer
//  | HeadCount            | int    | number of heads to repeat the same attention structure
//  | OutputSize           | int    | size of the output
//  | ActivationKernel     | int    | activation (ReLU or SoftMax) kernel transformation
//  | RandomFeaturesCount  | int    | projection matrix columns number, or 0 if isn't used
//  | Casual               | bool   | auto-regressive attention is used or not
//  +----------------------+--------+-------------------------------------------------------
class NEOML_API CMultiheadAttentionPerformerLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CMultiheadAttentionPerformerLayer )
public:
	explicit CMultiheadAttentionPerformerLayer( IMathEngine& mathEngine );

	// Activation kernel type: SoftMax(=0), ReLU(=1)
	// By default is SoftMax
	int GetActivationKernel() const { return activationKernel; }
	void SetActivationKernel( int activationKernel, int randomFeaturesCount, bool casual );
	int GetRandomFeaturesCount() const { return randomFeaturesCount; }
	bool GetCasual() const { return casual; }

	// The number of heads in attention
	// The GetHiddenSize() must be a multiple of this value
	// By default attention consist of 1 head
	int GetHeadCount() const { return headCount; }
	void SetHeadCount( int headCount );
	
	// The size of trainable matrices
	// Must be a multiple of GetHeadCount()
	int GetHiddenSize() const { return hiddenSize; }
	void SetHiddenSize( int hiddenSize );

	// The size of output
	int GetOutputSize() const { return outputSize; }
	void SetOutputSize( int outputSize );

	void Serialize( CArchive& archive ) override;

	// Recreates the layer if forceRebuild is true or it doesn't contain sublayers
	void Rebuild( bool forceRebuild );

protected:
	void Reshape() override;

private:
	// FAVOR+ attention settings
	int activationKernel; // Activation kernel transformation
	int randomFeaturesCount; // Projection matrix size, if > 0
	bool casual; // Auto-regression or not

	// The amount of heads
	int headCount;
	// The size of the trainable matrix
	int hiddenSize;
	// Output size
	int outputSize;

	// Layer inputs numeration
	enum TInputs { I_Q = 0, I_K = 1, I_V = 2 };
	
	bool isCreated() const { return HasLayer( "Q" ); }
	void create();
	
	CBaseLayer* multiplyInputByMatrixWeights( int size, const char* name, TInputs input );
	CBaseLayer* multiplyByMatrixWeights( CBaseLayer* input, int width );
	CBaseLayer* prepareQ( CBaseLayer* input );
	CBaseLayer* prepareKV( CBaseLayer* input, bool isK );
	CBaseLayer* prepareOutput( CBaseLayer* input );
};

NEOML_API CLayerWrapper<CMultiheadAttentionPerformerLayer> MultiheadAttentionPerformer(
	int headCount, int hiddenSize, int outputSize, int activationKernel, int randomFeaturesCount, bool casual );

} // namespace NeoML
