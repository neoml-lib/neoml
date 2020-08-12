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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/MultiheadAttentionLayer.h>
#include <NeoML/Dnn/Layers/MatrixMultiplicationLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/AddToObjectLayer.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/TransformLayer.h>
#include <NeoML/Dnn/Layers/TransposeLayer.h>
#include <NeoML/Dnn/Layers/SoftmaxLayer.h>

namespace NeoML {

CMultiheadAttentionLayer::CMultiheadAttentionLayer( IMathEngine& mathEngine ) :
	CCompositeLayer( mathEngine ),
	headCount( 1 ),
	hiddenSize( 8 ),
	dropoutRate( -1 ),
	useMask( false ),
	outputSize( 8 )
{
}

void CMultiheadAttentionLayer::SetHeadCount( int _headCount )
{
	NeoAssert( _headCount >= 1 );

	headCount = _headCount;
	DeleteAllLayers();
}

void CMultiheadAttentionLayer::SetHiddenSize( int _hiddenSize )
{
	NeoAssert( _hiddenSize >= 1 );

	hiddenSize = _hiddenSize;
	DeleteAllLayers();
}

void CMultiheadAttentionLayer::SetDropoutRate( float _dropoutRate )
{
	dropoutRate = _dropoutRate;
	DeleteAllLayers();
}

void CMultiheadAttentionLayer::SetUseMask( bool newValue )
{
	useMask = newValue;
	DeleteAllLayers();
}

void CMultiheadAttentionLayer::SetOutputSize( int _outputSize )
{
	NeoAssert( _outputSize > 0 );

	outputSize = _outputSize;
}

static const int MultiheadAttentionLayerVersion = 0;

void CMultiheadAttentionLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MultiheadAttentionLayerVersion );
	CCompositeLayer::Serialize( archive );
	archive.Serialize( headCount );
	archive.Serialize( hiddenSize );
	archive.Serialize( dropoutRate );
	archive.Serialize( useMask );
	archive.Serialize( outputSize );
}

void CMultiheadAttentionLayer::Reshape()
{
	if( !HasLayer( "Q" ) ) {
		create();
	}

	CCompositeLayer::Reshape();
}

// Creates layer with new parameters
// Here and further blob sizes are shown as [BathcWidth, ListSize, Width, Channels]
void CMultiheadAttentionLayer::create()
{
	NeoAssert( headCount > 0 );
	NeoAssert( hiddenSize % headCount == 0 );

	// scaling factor
	const float multiplier = static_cast<float>( 1.0 / sqrt( 1.0 * hiddenSize ) );

	// Applying W_Q, W_K and W_V to the corresponding inputs
	// [B, seq_Q, 1, hiddenSize]
	CBaseLayer* Q = multiplyInputByMatrixWeights( hiddenSize, "Q", I_Q );

	// [B, seq_to, 1, hiddenSize]
	CBaseLayer* K = multiplyInputByMatrixWeights( hiddenSize, "K", I_K );
	CBaseLayer* V = multiplyInputByMatrixWeights( hiddenSize, "V", I_V );

	// [B, n_head, seq_Q, d_k]
	Q = prepareQ( Q );

	// [B, n_head, d_k, seq_to]
	CBaseLayer* Kt = prepareK( K );

	// [B, n_head, seq_to, d_k]
	V = prepareV( V );
	
	// Multiplying Q by K_t
	// [B, n_head, seq_Q, seq_to]
	CPtr<CBaseLayer> QKt = new CMatrixMultiplicationLayer( MathEngine() );
	QKt->Connect( 0, *Q );
	QKt->Connect( 1, *Kt );
	QKt->SetName( GetName() + CString( ".QKt" ) );
	AddLayer( *QKt );

	// Applying scaling factor
	// [B, n_head, seq_Q, seq_to]
	CPtr<CLinearLayer> multiplierLayer = new CLinearLayer( MathEngine() );
	multiplierLayer->SetName( GetName() + CString( ".MultiplyByConst" ) );
	multiplierLayer->Connect( *QKt );
	multiplierLayer->SetMultiplier( multiplier );
	multiplierLayer->SetFreeTerm( 0 );
	AddLayer( *multiplierLayer );

	// Softmax
	// [B, n_head, seq_Q, seq_to]

	CBaseLayer* beforeSoftmax = multiplierLayer;
	if( useMask ) {
		beforeSoftmax = applyMask( beforeSoftmax );
	}

	CPtr<CBaseLayer> softmax = softmaxByChannels( *beforeSoftmax );

	// Dropout (if needed)
	// [B, n_head, seq_Q, seq_to]
	CPtr<CBaseLayer> afterSoftmax = softmax;
	if( dropoutRate > 0 ) {
		CPtr<CDropoutLayer> dropout = new CDropoutLayer( MathEngine() );
		dropout->SetName( GetName() + CString( ".Dropout" ) );
		dropout->Connect( *softmax );
		dropout->SetDropoutRate( dropoutRate );
		AddLayer( *dropout );

		afterSoftmax = dropout;
	}

	// Multiplying by V [B, n_head, seq_Q, seq_to] * [B, n_head, seq_to, d_k]
	// [B, n_head, seq_Q, d_k]
	CPtr<CMatrixMultiplicationLayer> head = new CMatrixMultiplicationLayer( MathEngine() );
	head->Connect( 0, *afterSoftmax );
	head->Connect( 1, *V );
	head->SetName( "MatrixDot" );
	AddLayer( *head );
	
	// [B, seq_Q, 1, hidden_size]
	CPtr<CBaseLayer> output = prepareOutput( head );

	output = multiplyByMatrixWeights( output, outputSize, "Out.Dense" );

	SetOutputMapping( O_Output, *output );
	SetOutputMapping( O_Softmax, *afterSoftmax );
}

// Multiplies input by trainable weights
CBaseLayer* CMultiheadAttentionLayer::multiplyInputByMatrixWeights( 
	int size, const char* name, TInputs input )
{
	NeoAssert( size > 0 );

	CPtr<CFullyConnectedLayer> fcLayer = new CFullyConnectedLayer( MathEngine() );
	fcLayer->SetNumberOfElements( size );
	fcLayer->SetZeroFreeTerm( false );
	fcLayer->SetName( name );
	AddLayer( *fcLayer );

	// Вход маппится на этот слой.
	SetInputMapping( input, *fcLayer, 0 );

	return fcLayer;
}

// Multiplies by trainable weights
CBaseLayer* CMultiheadAttentionLayer::multiplyByMatrixWeights( CBaseLayer* input, 
	int width, const char* name )
{
	NeoAssert( width >= 0 );
	NeoAssert( input != 0 );

	CPtr<CFullyConnectedLayer> fcLayer = new CFullyConnectedLayer( MathEngine() );
	fcLayer->SetNumberOfElements( width );
	fcLayer->Connect( *input );
	fcLayer->SetZeroFreeTerm( false );
	fcLayer->SetName( name );
	AddLayer( *fcLayer );

	return fcLayer;
}

// [B, n_head, seq_Q, seq_to]
CBaseLayer* CMultiheadAttentionLayer::softmaxByChannels( CBaseLayer& input )
{
	// input [B, n_head, seq_Q, seq_to]

	//  [B, n_head * seq_Q, 1,  seq_to]
	CPtr<CTransformLayer> reshape0 = new CTransformLayer( MathEngine() );
	reshape0->SetName( GetName() + CString( ".reshape0.Softmax" ) );
	reshape0->Connect( input );
	reshape0->SetDimensionRule( BD_BatchLength, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_BatchWidth, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_ListSize, CTransformLayer::O_Remainder, 0 );
	reshape0->SetDimensionRule( BD_Height, CTransformLayer::O_SetSize, 1 );
	reshape0->SetDimensionRule( BD_Width, CTransformLayer::O_SetSize, 1 );
	reshape0->SetDimensionRule( BD_Depth, CTransformLayer::O_SetSize, 1 );
	reshape0->SetDimensionRule( BD_Channels, CTransformLayer::O_Multiply, 1 );
	AddLayer( *reshape0 );

	CPtr<CSoftmaxLayer> softmax = new CSoftmaxLayer( MathEngine() );
	softmax->SetNormalizationArea( CSoftmaxLayer::NA_ObjectSize );
	softmax->Connect( *reshape0 );
	softmax->SetName( "Softmax.SoftmaxByChannels" );
	AddLayer( *softmax );

	//  [B, n_head, seq_Q,  seq_to]
	CPtr<CTransformLayer> reshape1 = new CTransformLayer( MathEngine() );
	reshape1->SetName( "Softmax.reshape1" );
	reshape1->Connect( *softmax );
	reshape1->SetDimensionRule( BD_BatchLength, CTransformLayer::O_Multiply, 1 );
	reshape1->SetDimensionRule( BD_BatchWidth, CTransformLayer::O_Multiply, 1 );
	reshape1->SetDimensionRule( BD_ListSize, CTransformLayer::O_SetSize, headCount );
	reshape1->SetDimensionRule( BD_Height, CTransformLayer::O_Multiply, 1 );
	reshape1->SetDimensionRule( BD_Width, CTransformLayer::O_Remainder, 0 );
	reshape1->SetDimensionRule( BD_Depth, CTransformLayer::O_SetSize, 1 );
	reshape1->SetDimensionRule( BD_Channels, CTransformLayer::O_Multiply, 1 );
	AddLayer( *reshape1 );

	return reshape1;
}

// Applies mask
CBaseLayer* CMultiheadAttentionLayer::applyMask( CBaseLayer* layer )
{
	NeoAssert( layer != 0 );

	CPtr<CLinearLayer> multiplierLayer = new CLinearLayer( MathEngine() );
	multiplierLayer->SetName( GetName() + CString( ".Mask.MultiplyByConst" ) );
	// The value is taken from the original realization
	multiplierLayer->SetMultiplier( -1e+9 );
	multiplierLayer->SetFreeTerm( 0 );
	AddLayer( *multiplierLayer );
	SetInputMapping( I_Mask, *multiplierLayer, 0 );

	CPtr<CAddToObjectLayer> sumLayer = new CAddToObjectLayer( MathEngine() );
	sumLayer->SetName( GetName() + CString( ".Mask.ObjEltwiseSum" ) );
	sumLayer->Connect( 0, *layer );
	sumLayer->Connect( 1, *multiplierLayer );
	AddLayer( *sumLayer );

	return sumLayer;
}

// [B, n_head, seq_Q, d_k]
CBaseLayer* CMultiheadAttentionLayer::prepareQ( CBaseLayer* input )
{
	NeoAssert( input != 0 );
	
	// input
	// [B, seq_Q, 1, hiddenSize]

	// [B, seq_Q, n_head, d_k]
	CPtr<CTransformLayer> reshape0 = new CTransformLayer( MathEngine() );
	reshape0->SetName( "Q.reshape0" );
	reshape0->Connect( *input );
	reshape0->SetDimensionRule( BD_BatchLength, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_BatchWidth, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_ListSize, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_Height, CTransformLayer::O_SetSize, 1 );
	reshape0->SetDimensionRule( BD_Width, CTransformLayer::O_SetSize, headCount );
	reshape0->SetDimensionRule( BD_Depth, CTransformLayer::O_SetSize, 1 );
	reshape0->SetDimensionRule( BD_Channels, CTransformLayer::O_SetSize, hiddenSize / headCount );
	AddLayer( *reshape0 );
	
	// [B, n_head, seq_Q, d_k]
	CPtr<CTransposeLayer> transpose0 = new CTransposeLayer( MathEngine() );
	transpose0->SetName( "Q.transpose0" );
	transpose0->SetTransposedDimensions( BD_ListSize, BD_Width );
	transpose0->Connect( *reshape0 );
	AddLayer( *transpose0 );
	
	return transpose0;
}

// [B, n_head, d_k, seq_to]
CBaseLayer* CMultiheadAttentionLayer::prepareK( CBaseLayer* input )
{
	NeoAssert( input != 0 );
	// input
	// [B, seq_to, 1, hiddenSize]

	// [B, hiddenSize, 1, seq_to]
	CPtr<CTransposeLayer> transpose0 = new CTransposeLayer( MathEngine() );
	transpose0->SetName( "K.transpose0" );
	transpose0->SetTransposedDimensions( BD_ListSize, BD_Channels );
	transpose0->Connect( *input );
	AddLayer( *transpose0 );
	
	// [B, n_head, d_k, seq_to]
	CPtr<CTransformLayer> reshape0 = new CTransformLayer( MathEngine() );
	reshape0->SetName( "K.reshape0" );
	reshape0->Connect( *transpose0 );
	reshape0->SetDimensionRule( BD_BatchLength, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_BatchWidth, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_ListSize, CTransformLayer::O_SetSize, headCount );
	reshape0->SetDimensionRule( BD_Height, CTransformLayer::O_SetSize, 1 );
	reshape0->SetDimensionRule( BD_Width, CTransformLayer::O_SetSize, hiddenSize / headCount );
	reshape0->SetDimensionRule( BD_Depth, CTransformLayer::O_SetSize, 1 );
	reshape0->SetDimensionRule( BD_Channels, CTransformLayer::O_Multiply, 1 );
	AddLayer( *reshape0 );

	return reshape0;
}

// [B, n_head, seq_to, d_k]
CBaseLayer* CMultiheadAttentionLayer::prepareV( CBaseLayer* input )
{
	NeoAssert( input != 0 );

	// input
	// [B, seq_to, 1, hiddenSize]

	// [B, seq_to, n_head, d_k]
	CPtr<CTransformLayer> reshape0 = new CTransformLayer( MathEngine() );
	reshape0->SetName( "V.reshape0" );
	reshape0->Connect( *input );
	reshape0->SetDimensionRule( BD_BatchLength, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_BatchWidth, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_ListSize, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_Height, CTransformLayer::O_SetSize, 1 );
	reshape0->SetDimensionRule( BD_Width, CTransformLayer::O_SetSize, headCount );
	reshape0->SetDimensionRule( BD_Depth, CTransformLayer::O_SetSize, 1 );
	reshape0->SetDimensionRule( BD_Channels, CTransformLayer::O_SetSize, hiddenSize / headCount );
	AddLayer( *reshape0 );

	// [B, n_head, seq_to, d_k]
	CPtr<CTransposeLayer> transpose0 = new CTransposeLayer( MathEngine() );
	transpose0->SetName( "V.transpose0" );
	transpose0->SetTransposedDimensions( BD_ListSize, BD_Width );
	transpose0->Connect( *reshape0 );
	AddLayer( *transpose0 );

	return transpose0;
}

// [B, seq_Q, 1, hidden_size]
CBaseLayer* CMultiheadAttentionLayer::prepareOutput( CBaseLayer* input )
{
	NeoAssert( input != 0 );

	// input
	// [B, n_head, seq_Q, d_k]
	
	// [B, seq_Q, n_head, d_k]
	CPtr<CTransposeLayer> transpose0 = new CTransposeLayer( MathEngine() );
	transpose0->SetName( "Out.transpose0.Out" );
	transpose0->SetTransposedDimensions( BD_ListSize, BD_Width );
	transpose0->Connect( *input );
	AddLayer( *transpose0 );

	// [B, seq_Q, 1, hidden_size]
	CPtr<CTransformLayer> reshape0 = new CTransformLayer( MathEngine() );
	reshape0->SetName( "Out.reshape0.Out" );
	reshape0->Connect( *transpose0 );
	reshape0->SetDimensionRule( BD_BatchLength, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_BatchWidth, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_ListSize, CTransformLayer::O_Multiply, 1 );
	reshape0->SetDimensionRule( BD_Height, CTransformLayer::O_SetSize, 1 );
	reshape0->SetDimensionRule( BD_Width, CTransformLayer::O_SetSize, 1 );
	reshape0->SetDimensionRule( BD_Depth, CTransformLayer::O_SetSize, 1 );
	reshape0->SetDimensionRule( BD_Channels, CTransformLayer::O_SetSize, hiddenSize );
	AddLayer( *reshape0 );

	return reshape0;
}

} // namespace NeoML
