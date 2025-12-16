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

#include <common.h>
#pragma hdrstop

#include <cmath>
#include <NeoML/Dnn/Layers/MultiheadAttentionPerformerLayer.h>
#include <NeoML/Dnn/Layers/FavorAttentionPerformerLayer.h>
//#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/TransformLayer.h>
#include <NeoML/Dnn/Layers/TransposeLayer.h>

namespace NeoML {

//---------------------------------------------------------------------------------------------------------------------

CMultiheadAttentionPerformerLayer::CMultiheadAttentionPerformerLayer( IMathEngine& mathEngine ) :
	CCompositeLayer( mathEngine ),
	activationKernel( 0 ),
	randomFeaturesCount( 0 ),
	casual( true ),
	headCount( 1 ),
	hiddenSize( 8 ),
	outputSize( 8 )
{}

void CMultiheadAttentionPerformerLayer::SetActivationKernel( int _activationKernel, int _randomFeaturesCount, bool _casual )
{
	NeoAssert( _activationKernel == 0 || _activationKernel == 1 );
	NeoAssert( _randomFeaturesCount >= 0 );

	if( activationKernel != _activationKernel
		|| randomFeaturesCount != _randomFeaturesCount
		|| casual != _casual )
	{
		activationKernel = _activationKernel;
		randomFeaturesCount = _randomFeaturesCount;
		casual = _casual;

		DeleteAllLayers();
	}
}

void CMultiheadAttentionPerformerLayer::SetHeadCount( int _headCount )
{
	NeoAssert( _headCount >= 1 );
	if( headCount != _headCount ) {
		headCount = _headCount;
		DeleteAllLayers();
	}
}

void CMultiheadAttentionPerformerLayer::SetHiddenSize( int _hiddenSize )
{
	NeoAssert( _hiddenSize >= 1 );
	if( hiddenSize != _hiddenSize ) {
		hiddenSize = _hiddenSize;
		DeleteAllLayers();
	}
}

void CMultiheadAttentionPerformerLayer::SetOutputSize( int _outputSize )
{
	NeoAssert( _outputSize > 0 );
	if( outputSize != _outputSize ) {
		outputSize = _outputSize;
		DeleteAllLayers();
	}
}

static const int MultiheadAttentionPerformerLayerVersion = 0;

void CMultiheadAttentionPerformerLayer::Serialize( CArchive& archive )
{
	( void ) archive.SerializeVersion( MultiheadAttentionPerformerLayerVersion );
	CCompositeLayer::Serialize( archive );

	archive.Serialize( activationKernel );
	archive.Serialize( randomFeaturesCount );
	archive.Serialize( casual );
	archive.Serialize( headCount );
	archive.Serialize( hiddenSize );
	archive.Serialize( outputSize );
}

void CMultiheadAttentionPerformerLayer::Reshape()
{
	CheckInputs();
	CheckLayerArchitecture( GetInputCount() == 3, "MultiheadAttentionPerformer layer inputs count should be 3" );
	CheckLayerArchitecture( GetOutputCount() == 1, "MultiheadAttentionPerformer layer outputs count should be 1" );

	if( !isCreated() ) {
		create();
	}
	CFullyConnectedLayer* Q = CheckCast<CFullyConnectedLayer>( GetLayer( "Q" ) );
	const bool uninitializedWeights = ( Q->Weights() == nullptr );

	CCompositeLayer::Reshape();

	if( uninitializedWeights ) {
		// Glorot initialization 
		
		// For layers for linearly projecting the queries, keys, and values initialization
		const int inputSize = inputDescs[I_Q].Channels();
		const float attentionGlorotLimit = static_cast<float>( std::sqrt( 6.0 / ( inputSize + hiddenSize ) ) );
		CDnnUniformInitializer attentionInitializer( GetDnn()->Random(), -attentionGlorotLimit, attentionGlorotLimit );

		attentionInitializer.InitializeLayerParams( *( Q->Weights() ), /*unused*/0 );
		attentionInitializer.InitializeLayerParams( *( Q->FreeTerms() ), /*unused*/0 );

		CFullyConnectedLayer* K = CheckCast<CFullyConnectedLayer>( GetLayer( "K" ) );
		attentionInitializer.InitializeLayerParams( *( K->Weights() ), /*unused*/0 );
		attentionInitializer.InitializeLayerParams( *( K->FreeTerms() ), /*unused*/0 );

		CFullyConnectedLayer* V = CheckCast<CFullyConnectedLayer>( GetLayer( "V" ) );
		attentionInitializer.InitializeLayerParams( *( V->Weights() ), /*unused*/0 );
		attentionInitializer.InitializeLayerParams( *( V->FreeTerms() ), /*unused*/0 );

		// Output layer
		const float outputGlorotLimit = static_cast<float>( std::sqrt( 6.0 / ( hiddenSize + hiddenSize ) ) );
		CDnnUniformInitializer outputInitializer( GetDnn()->Random(), -outputGlorotLimit, outputGlorotLimit );
		CFullyConnectedLayer* Out = CheckCast<CFullyConnectedLayer>( GetLayer( "Out.Dense" ) );
		outputInitializer.InitializeLayerParams( *( Out->Weights() ), /*unused*/0 );
		outputInitializer.InitializeLayerParams( *( Out->FreeTerms() ), /*unused*/0 );
	}
}

// Recreates the layer if forceRebuild is true or it doesn't contain sublayers
void CMultiheadAttentionPerformerLayer::Rebuild( bool forceRebuild )
{
	if( forceRebuild && isCreated() ) {
		DeleteAllLayers();
	}
	if ( !isCreated() ) {
		create();
	}
}

// Creates layer with new parameters
// Here and further blob sizes are shown as [BatchWidth, ListSize, Width, Channels]
void CMultiheadAttentionPerformerLayer::create()
{
	NeoAssert( headCount > 0 );
	NeoAssert( hiddenSize % headCount == 0 );

	// Applying W_Q, W_K and W_V to the corresponding inputs
	// [B, seq_Q, 1, hiddenSize]
	CBaseLayer* Q = multiplyInputByMatrixWeights( hiddenSize, "Q", I_Q );

	// [B, seq_to, 1, hiddenSize]
	CBaseLayer* K = multiplyInputByMatrixWeights( hiddenSize, "K", I_K );
	CBaseLayer* V = multiplyInputByMatrixWeights( hiddenSize, "V", I_V );

	// [B, n_head, seq_Q, d_k]
	Q = prepareQ( Q );
	// [B, n_head, seq_to, d_k]
	K = prepareKV( K, true );
	// [B, n_head, seq_to, d_k]
	V = prepareKV( V, false );

	CPtr<CFavorAttentionPerformerLayer> favor = new CFavorAttentionPerformerLayer( MathEngine(), "favor" );
	favor->SetActivationKernel( activationKernel );
	favor->SetRandomFeaturesCount( randomFeaturesCount );
	favor->SetCausal( casual );
	favor->Connect( CFavorAttentionPerformerLayer::TI_Q, *Q );
	favor->Connect( CFavorAttentionPerformerLayer::TI_K, *K );
	favor->Connect( CFavorAttentionPerformerLayer::TI_V, *V );
	AddLayer( *favor );

	// [B, seq_Q, 1, hidden_size]
	CPtr<CBaseLayer> output = prepareOutput( favor );
	output = multiplyByMatrixWeights( output, outputSize );

	SetOutputMapping( /*O_Output*/0, *output );
}

// Multiplies input by trainable weights
CBaseLayer* CMultiheadAttentionPerformerLayer::multiplyInputByMatrixWeights( 
	int size, const char* name, TInputs input )
{
	NeoAssert( size > 0 );

	CPtr<CFullyConnectedLayer> fcLayer = new CFullyConnectedLayer( MathEngine(), name );
	fcLayer->SetNumberOfElements( size );
	fcLayer->SetZeroFreeTerm( false );
	AddLayer( *fcLayer );

	// Connect input with this sublayer
	SetInputMapping( input, *fcLayer, 0 );

	return fcLayer;
}

// Multiplies by trainable weights
CBaseLayer* CMultiheadAttentionPerformerLayer::multiplyByMatrixWeights( CBaseLayer* input, int width )
{
	NeoAssert( width >= 0 );
	NeoAssert( input != 0 );

	CPtr<CFullyConnectedLayer> fcLayer = new CFullyConnectedLayer( MathEngine(), "Out.Dense" );
	fcLayer->SetNumberOfElements( width );
	fcLayer->SetZeroFreeTerm( false );
	fcLayer->Connect( *input );
	AddLayer( *fcLayer );

	return fcLayer;
}

// [B, n_head, seq_Q, d_k]
CBaseLayer* CMultiheadAttentionPerformerLayer::prepareQ( CBaseLayer* input )
{
	NeoAssert( input != 0 );

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

// [B, n_head, seq_to, d_k]
CBaseLayer* CMultiheadAttentionPerformerLayer::prepareKV( CBaseLayer* input, bool isK )
{
	NeoAssert( input != 0 );

	// [B, seq_to, n_head, d_k]
	CPtr<CTransformLayer> reshape0 = new CTransformLayer( MathEngine() );
	reshape0->SetName( isK ? "K.reshape0" : "V.reshape0" );
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
	transpose0->SetName( isK ? "K.transpose0" : "V.transpose0" );
	transpose0->SetTransposedDimensions( BD_ListSize, BD_Width );
	transpose0->Connect( *reshape0 );
	AddLayer( *transpose0 );

	return transpose0;
}

// [B, seq_Q, 1, hidden_size]
CBaseLayer* CMultiheadAttentionPerformerLayer::prepareOutput( CBaseLayer* input )
{
	NeoAssert( input != 0 );
	
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

//---------------------------------------------------------------------------------------------------------------------

CLayerWrapper<CMultiheadAttentionPerformerLayer> MultiheadAttentionPerformer(
	int headCount, int hiddenSize, int outputSize, int activationKernel, int randomFeaturesCount, bool casual )
{
	return CLayerWrapper<CMultiheadAttentionPerformerLayer>( "MultiheadAttentionPerformer",
		[=]( CMultiheadAttentionPerformerLayer* result ) {
			result->SetHeadCount( headCount );
			result->SetHiddenSize( hiddenSize );
			result->SetOutputSize( outputSize );
			result->SetActivationKernel( activationKernel, randomFeaturesCount, casual );
		} );
}

} // namespace NeoML
