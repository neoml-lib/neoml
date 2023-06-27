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

#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>

namespace NeoML {

// Multihead Self Attention
// UNIVERSAL TRANSFORMERS https://arxiv.org/pdf/1706.03762.pdf 
// Attention Is All You Need https://arxiv.org/pdf/1807.03819.pdf

//  Inputs:
//  #0 - matrix Q (1 x BatchWidth x ListSize_Q x 1 x 1 x 1 x Channels_Q)
//  #1 - matrix K (1 x BatchWidth x ListSize_V x 1 x 1 x 1 x Channels_Q)
//  #2 - matrix V (1 x BatchWidth x ListSize_V x 1 x 1 x 1 x Channels_V)
//  #3 - mask, can be 0.
//
//  Q = W_Q * Q
//  K = W_K * K
//  V = W_V * V
//  where W_* are trainable matrices of size (Channels_* x GetHiddenSize())
// 
//  Attention(Q, K, V) = softmax( Q * K_t / sqrt(d_K) ) * V
//  where d_k - dimension of k
//  
//  MultiHeadAttention = dropout_if_needed(concat( head_1, ..., head_N )) * W_O
//  where head_i = Attention( W_Q_i * X, W_K_i * X, W_V_i * X ) 
//  W_* - trainable parameters and W_O is an additional trainable matrix of size (GetHiddenSize() x GetOutputSize())
//
// Result has size (1, BatchWidth, ListSize_Q, 1, 1, 1, GetOutputSize())
class NEOML_API CMultiheadAttentionLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CMultiheadAttentionLayer )
public:
	explicit CMultiheadAttentionLayer( IMathEngine& mathEngine );

	// Mask type
	enum TMaskType {
		// One mask for all objects
		// Its shape is (1 x 1 x 1 x 1 x ListSize_Q x 1 x ListSize_V)
		MT_OneObject = 0,

		// Different masks for different objects
		// Its shape is (1 x BatchWidth x headCount x 1 x ListSize_Q x 1 x ListSize_V)
		MT_Eltwise = 1
	};

	// The number of heads in attention
	// The GetHiddenSize() must be a multiple of this value
	// By default attention consist of 1 head
	int GetHeadCount() const { return headCount; }
	void SetHeadCount( int headCount );

	// The size of trainable matrices
	// Must be a multiple of GetHeadCount()
	int GetHiddenSize() const { return hiddenSize; }
	void SetHiddenSize( int _hiddenSize );

	// Rate of dropout applied to the softmax
	// Negative value means no dropout
	// By default the dropout rate is -1
	float GetDropoutRate() const { return dropoutRate; }
	void SetDropoutRate( float dropoutRate ); 

	// Mask usage
	bool GetUseMask() const { return useMask; }
	void SetUseMask( bool newValue );

	// Mask type
	TMaskType GetMaskType() const { return maskType; }
	void SetMaskType( TMaskType _maskType );

	// The size of output
	int GetOutputSize() const { return outputSize; }
	void SetOutputSize( int _outputSize );

	// Scale in attention previously depended on number of heads
	bool IsInCompatibilityMode() const { return isInCompatibilityMode; }
	void SetCompatibilityMode( bool value );

	void Serialize( CArchive& archive ) override;

	// Recreates the layer if forceRebuild is true or it doesn't contain sublayers
	void Rebuild( bool forceRebuild );

protected:
	void Reshape() override;

private:
	// The amount of heads
	int headCount;
	// The size of the trainable matrix
	int hiddenSize;
	// Dropout rate
	float dropoutRate;
	// Mask usage
	bool useMask;
	// Type of mask used
	TMaskType maskType;
	// Output size
	int outputSize;
	// scale in attention
	bool isInCompatibilityMode;
	// layer applying scale
	CString multiplyByConstLayerName;

	void create();

	// Layer inputs
	enum TInputs {
		I_Q = 0,
		I_K = 1,
		I_V = 2,
		I_Mask = 3
	};

	// Layer outputs
	enum TOutputs {
		O_Output = 0,
		O_Softmax = 1
	};

	CBaseLayer* multiplyInputByMatrixWeights( int size, const char* name, TInputs input );
	CBaseLayer* multiplyByMatrixWeights( CBaseLayer* input, 
		int width, const char* prefix );
	CBaseLayer* softmaxByChannels( CBaseLayer& input );
	CBaseLayer* applyMask( CBaseLayer* layer );
	CBaseLayer* prepareQ( CBaseLayer* input );
	CBaseLayer* prepareK( CBaseLayer* input );
	CBaseLayer* prepareV( CBaseLayer* input );
	CBaseLayer* prepareOutput( CBaseLayer* input );

	// divide dot product by sqrt( d_k ) or sqrt( d_K ) in compatibility mode, where k - one key, K - concatenation of keys from all heads
	float getScalingFactor() const
		{ return static_cast<float>( 1.0 / sqrt( 1.0 * hiddenSize / ( isInCompatibilityMode ? 1 : headCount ) ) ); }
};

NEOML_API CLayerWrapper<CMultiheadAttentionLayer> MultiheadAttention(
	int headCount, int hiddenSize, int outputSize, float dropoutRate );

} // namespace NeoML
