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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/TransformerLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/ObjectNormalizationLayer.h>

namespace NeoML {

static const char* selfAttentionName = "SelfAttention";
static const char* fc1Name = "FullyConnected1";
static const char* activationName = "Activation";
static const char* dropout1Name = "Dropout1";
static const char* fc2Name = "FullyConnected2";
static const char* dropout2Name = "Dropout2";
static const char* feedForwardResidualName = "FeedForwardResidual";

CTransformerLayer::CTransformerLayer( IMathEngine& mathEngine )
	: CCompositeLayer( mathEngine, "CTransformerLayer" )
{
	buildLayer();
}

static const int transformerLayerVersion = 0;

static CPtr<CDropoutLayer> getOptionalDropout( CDnnLayerGraph& dnn, const char* name )
{
	if( dnn.HasLayer( name ) ) {
		return CheckCast<CDropoutLayer>( dnn.GetLayer( name ) );
	}
	return nullptr;
}

void CTransformerLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( transformerLayerVersion );
	CCompositeLayer::Serialize( archive );
	if( archive.IsLoading() ) {
		selfAttention = CheckCast<CMultiheadAttentionLayer>( GetLayer( selfAttentionName ) );
		fc1 = CheckCast<CFullyConnectedLayer>( GetLayer( fc1Name ) );
		dropout1 = getOptionalDropout( *this, dropout1Name );
		fc2 = CheckCast<CFullyConnectedLayer>( GetLayer( fc2Name ) );
		dropout2 = getOptionalDropout( *this, dropout2Name );
	}
}

void CTransformerLayer::SetHeadCount( int headCount )
{
	NeoAssert( headCount > 0 );
	selfAttention->SetHeadCount( headCount );
	ForceReshape();
	NeoPresume( GetHeadCount() == headCount );
}

void CTransformerLayer::SetHiddenSize( int hiddenSize )
{
	NeoAssert( hiddenSize > 0 );
	selfAttention->SetHiddenSize( hiddenSize );
	ForceReshape();
	NeoPresume( GetHiddenSize() == hiddenSize );
}

int CTransformerLayer::GetOutputSize() const
{
	NeoAssert( selfAttention->GetOutputSize() == fc2->GetNumberOfElements() );
	return selfAttention->GetOutputSize();
}

void CTransformerLayer::SetOutputSize( int outputSize )
{
	NeoAssert( outputSize > 0 );
	NeoAssert( selfAttention->GetOutputSize() == fc2->GetNumberOfElements() );
	selfAttention->SetOutputSize( outputSize );
	fc2->SetNumberOfElements( outputSize );
	ForceReshape();
	NeoPresume( GetOutputSize() == outputSize );
}

void CTransformerLayer::SetAttentionDropout( float rate )
{
	NeoAssert( rate < 1.f );

	selfAttention->SetDropoutRate( rate );
	ForceReshape();

	NeoPresume( ( rate <= 0.f && GetAttentionDropout() <= 0.f )
		|| ( rate > 0.f && GetAttentionDropout() == rate ) );
}

void CTransformerLayer::SetFeedForwardSize( int size )
{
	NeoAssert( size > 0 );

	fc1->SetNumberOfElements( size );
	ForceReshape();

	NeoPresume( GetFeedForwardSize() == size );
}

void CTransformerLayer::SetActivation( TActivationFunction newFunction )
{
	NeoAssert( HasLayer( activationName ) );

	DeleteLayer( activationName );
	CPtr<CBaseLayer> activation = CreateActivationLayer( MathEngine(), newFunction );
	activation->SetName( activationName );
	activation->Connect( *fc1 );
	if( dropout1 == nullptr ) {
		fc2->Connect( *activation );
	} else {
		dropout1->Connect( *activation );
	}
	AddLayer( *activation );

	NeoPresume( HasLayer( activationName ) );
}

float CTransformerLayer::GetFeedForwardDropout() const
{
	return dropout1 == nullptr ? 0.f : dropout1->GetDropoutRate();
}

void CTransformerLayer::SetFeedForwardDropout( float rate )
{
	NeoAssert( rate < 1.f );
	if( rate > 0.f ) {
		addDropoutLayers();
		dropout1->SetDropoutRate( rate );
		dropout2->SetDropoutRate( rate );
	} else {
		removeDropoutLayers();
	}

	NeoPresume( ( rate <= 0.f && GetFeedForwardDropout() <= 0.f )
		|| ( rate > 0.f && GetFeedForwardDropout() == rate ) );
}

void CTransformerLayer::buildLayer()
{
	// Self-attention is the multihead attention with the same data
	// used for all 3 attention inputs
	selfAttention = FINE_DEBUG_NEW CMultiheadAttentionLayer( MathEngine() );
	selfAttention->SetName( selfAttentionName );
	selfAttention->SetHeadCount( 1 );
	selfAttention->SetHiddenSize( 1 );
	selfAttention->SetOutputSize( 1 );
	SetInputMapping( 0, *selfAttention, 0 );
	SetInputMapping( 0, *selfAttention, 1 );
	SetInputMapping( 0, *selfAttention, 2 );
	AddLayer( *selfAttention );

	// Sum attention result with the original input
	CPtr<CEltwiseSumLayer> attentionResidual = FINE_DEBUG_NEW CEltwiseSumLayer( MathEngine() );
	attentionResidual->SetName( "AttentionResidual" );
	SetInputMapping( 0, *attentionResidual, 0 );
	attentionResidual->Connect( 1, *selfAttention );
	AddLayer( *attentionResidual );

	// Normalize the sum
	CPtr<CObjectNormalizationLayer> attentionNorm = FINE_DEBUG_NEW CObjectNormalizationLayer( MathEngine() );
	attentionNorm->SetName( "AttentionNorm" );
	attentionNorm->Connect( *attentionResidual );
	AddLayer( *attentionNorm );

	// First fully-connected of feed-forward
	fc1 = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	fc1->SetName( fc1Name );
	fc1->SetNumberOfElements( 1 );
	fc1->Connect( *attentionNorm );
	AddLayer( *fc1 );

	// Add activation (ReLU by default)
	CPtr<CBaseLayer> activation = FINE_DEBUG_NEW CReLULayer( MathEngine() );
	activation->SetName( activationName );
	activation->Connect( *fc1 );
	AddLayer( *activation );

	// Second fully-connected of feed-forward
	fc2 = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	fc2->SetName( fc2Name );
	fc2->SetNumberOfElements( 1 );
	fc2->Connect( *activation );
	AddLayer( *fc2 );

	// Sum normalized attention with the feed-forward result
	feedForwardResidual = FINE_DEBUG_NEW CEltwiseSumLayer( MathEngine() );
	feedForwardResidual->SetName( feedForwardResidualName );
	feedForwardResidual->Connect( 0, *fc2 );
	feedForwardResidual->Connect( 1, *attentionNorm );
	AddLayer( *feedForwardResidual );

	// Normalize output
	CPtr<CObjectNormalizationLayer> feedForwardNorm = FINE_DEBUG_NEW CObjectNormalizationLayer( MathEngine() );
	feedForwardNorm->SetName( "FeedForwardNorm" );
	feedForwardNorm->Connect( *feedForwardResidual );
	AddLayer( *feedForwardNorm );

	SetOutputMapping( *feedForwardNorm );
}

void CTransformerLayer::addDropoutLayers()
{
	if( dropout1 != nullptr ) {
		NeoAssert( dropout1 != nullptr && dropout2 != nullptr );
		return;
	}

	NeoAssert( dropout1 == nullptr && dropout2 == nullptr );

	dropout1 = FINE_DEBUG_NEW CDropoutLayer( MathEngine() );
	dropout1->SetName( dropout1Name );
	dropout1->Connect( activationName );
	fc2->Connect( *dropout1 );
	AddLayer( *dropout1 );

	dropout2 = FINE_DEBUG_NEW CDropoutLayer( MathEngine() );
	dropout2->SetName( dropout2Name );
	dropout2->Connect( *fc2 );
	feedForwardResidual->Connect( *dropout2 );
	AddLayer( *dropout2 );

	NeoPresume( dropout1 != nullptr && dropout1->GetName() == CString( dropout1Name ) );
	NeoPresume( dropout2 != nullptr && dropout2->GetName() == CString( dropout2Name ) );
}

void CTransformerLayer::removeDropoutLayers()
{
	if( dropout1 == nullptr ) {
		NeoAssert( dropout1 == nullptr && dropout2 == nullptr );
		return;
	}

	NeoAssert( dropout1 != nullptr && dropout2 != nullptr );

	DeleteLayer( *dropout1 );
	dropout1 = nullptr;
	fc2->Connect( activationName );

	DeleteLayer( *dropout2 );
	dropout2 = nullptr;
	feedForwardResidual->Connect( *fc2 );

	NeoPresume( dropout1 == nullptr && !HasLayer( dropout1Name ) );
	NeoPresume( dropout2 == nullptr && !HasLayer( dropout2Name ) );
}

CLayerWrapper<CTransformerLayer> Transformer( int headCount, int hiddenSize, int outputSize,
	float attentionDropout, int feedForwardSize, float feedForwardDropout, TActivationFunction activation )
{
	return CLayerWrapper<CTransformerLayer>( "CTransformerLayer", [=]( CTransformerLayer* result ) {
		result->SetHeadCount( headCount );
		result->SetHiddenSize( hiddenSize );
		result->SetOutputSize( outputSize );
		result->SetAttentionDropout( attentionDropout );
		result->SetFeedForwardSize( feedForwardSize );
		result->SetFeedForwardDropout( feedForwardDropout );
		result->SetActivation( activation );
	} );
}

} // namespace NeoML
