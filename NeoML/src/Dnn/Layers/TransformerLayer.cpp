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
static const char* selfAttentionSumName = "SelfAttentionSum";
static const char* dropoutSelfAttentionName = "DropoutSelfAttention";
static const char* fc1Name = "FullyConnected1";
static const char* activationName = "Activation";
static const char* dropoutFc1Name = "DropoutFc1";
static const char* fc2Name = "FullyConnected2";
static const char* dropoutFc2Name = "DropoutFc2";
static const char* feedForwardSumName = "FeedForwardSum";

static CPtr<CDropoutLayer> getOptionalDropout( CDnnLayerGraph& dnn, const char* name )
{
	if( dnn.HasLayer( name ) ) {
		return CheckCast<CDropoutLayer>( dnn.GetLayer( name ) );
	}
	return nullptr;
}

static inline void checkBlob( const CBlobDesc& desc, const char* layerName, const char* blobName,
	int batchWidth, int listSize, int width, int channels )
{
	CheckArchitecture( desc.GetDataType() == CT_Float, layerName, CString( blobName ) + " must be float" );
	CheckArchitecture( desc.BatchLength() == 1, layerName, CString( blobName ) + "'s BatchLength must be 1" );
	CheckArchitecture( desc.Height() == 1, layerName, CString( blobName ) + "'s Height must be 1" );
	CheckArchitecture( desc.Depth() == 1, layerName, CString( blobName ) + "'s Depth must be 1" );
	if( batchWidth > 0 ) {
		CheckArchitecture( desc.BatchWidth() == batchWidth, layerName, CString( blobName ) + "'s BatchWidth mismatch" );
	}
	if( listSize > 0 ) {
		CheckArchitecture( desc.ListSize() == listSize, layerName, CString( blobName ) + "'s ListSize mismatch" );
	}
	if( width > 0 ) {
		CheckArchitecture( desc.Width() == width, layerName, CString( blobName ) + "'s Width mismatch" );
	}
	if( channels > 0 ) {
		CheckArchitecture( desc.Channels() == channels, layerName, CString( blobName ) + "'s Channels mismatch" );
	}
}

// --------------------------------------------------------------------------------------------------------------------

CTransformerEncoderLayer::CTransformerEncoderLayer( IMathEngine& mathEngine )
	: CCompositeLayer( mathEngine, "CTransformerEncoderLayer" )
{
	buildLayer();
}

static const int transformerEncoderLayerVersion = 0;

void CTransformerEncoderLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( transformerEncoderLayerVersion );
	CCompositeLayer::Serialize( archive );
	if( archive.IsLoading() ) {
		selfAttention = CheckCast<CMultiheadAttentionLayer>( GetLayer( selfAttentionName ) );
		dropoutSelfAttention = getOptionalDropout( *this, dropoutSelfAttentionName );
		selfAttentionSum = CheckCast<CEltwiseSumLayer>( GetLayer( selfAttentionSumName ) );
		fc1 = CheckCast<CFullyConnectedLayer>( GetLayer( fc1Name ) );
		dropoutFc1 = getOptionalDropout( *this, dropoutFc1Name );
		fc2 = CheckCast<CFullyConnectedLayer>( GetLayer( fc2Name ) );
		dropoutFc2 = getOptionalDropout( *this, dropoutFc2Name );
		feedForwardSum = CheckCast<CEltwiseSumLayer>( GetLayer( feedForwardSumName ) );
	}
}

void CTransformerEncoderLayer::SetHeadCount( int headCount )
{
	NeoAssert( headCount > 0 );
	selfAttention->SetHeadCount( headCount );
	ForceReshape();
	NeoPresume( GetHeadCount() == headCount );
}

void CTransformerEncoderLayer::SetHiddenSize( int hiddenSize )
{
	NeoAssert( hiddenSize > 0 );
	selfAttention->SetHiddenSize( hiddenSize );
	ForceReshape();
	NeoPresume( GetHiddenSize() == hiddenSize );
}

float CTransformerEncoderLayer::GetDropoutRate() const
{
	if( dropoutSelfAttention == nullptr ) {
		NeoPresume( dropoutSelfAttention == nullptr && dropoutFc1 == nullptr && dropoutFc2 == nullptr );
		return 0.f;
	}

	NeoPresume( dropoutSelfAttention != nullptr && dropoutFc1 != nullptr && dropoutFc2 != nullptr );
	NeoPresume( dropoutSelfAttention->GetDropoutRate() == dropoutFc1->GetDropoutRate() );
	NeoPresume( dropoutSelfAttention->GetDropoutRate() == dropoutFc2->GetDropoutRate() );
	return dropoutFc2->GetDropoutRate();
}

void CTransformerEncoderLayer::SetDropoutRate( float rate )
{
	NeoAssert( rate < 1.f );

	if( rate > 0.f ) {
		addDropoutLayers();
		dropoutSelfAttention->SetDropoutRate( rate );
		dropoutFc1->SetDropoutRate( rate );
		dropoutFc2->SetDropoutRate( rate );
	} else {
		removeDropoutLayers();
	}
}

void CTransformerEncoderLayer::SetFeedForwardSize( int size )
{
	NeoAssert( size > 0 );

	fc1->SetNumberOfElements( size );
	ForceReshape();

	NeoPresume( GetFeedForwardSize() == size );
}

void CTransformerEncoderLayer::SetActivation( TActivationFunction newFunction )
{
	NeoAssert( HasLayer( activationName ) );

	DeleteLayer( activationName );
	CPtr<CBaseLayer> activation = CreateActivationLayer( MathEngine(), newFunction );
	activation->SetName( activationName );
	activation->Connect( *fc1 );
	if( dropoutFc1 == nullptr ) {
		fc2->Connect( *activation );
	} else {
		dropoutFc1->Connect( *activation );
	}
	AddLayer( *activation );

	NeoPresume( HasLayer( activationName ) );
}

void CTransformerEncoderLayer::Reshape()
{
	CheckArchitecture( GetHiddenSize() % GetHeadCount() == 0, GetName(), "HiddenSize must be a multiple of HeadCount" );
	CheckArchitecture( GetInputCount() == 1 || GetInputCount() == 2, GetName(), "Layer must have 1 or 2 inputs" );
	checkBlob( inputDescs[0], GetName(), "input data", -1, -1, 1, -1 );
	if( GetInputCount() == 2 ) {
		checkBlob( inputDescs[1], GetName(), "input mask", 1, 1, inputDescs[0].ListSize(), inputDescs[0].ListSize() );
	}

	if( selfAttention->GetOutputSize() != inputDescs[0].Channels() ) {
		selfAttention->SetOutputSize( inputDescs[0].Channels() );
	}
	if( fc2->GetNumberOfElements() != inputDescs[0].Channels() ) {
		fc2->SetNumberOfElements( inputDescs[0].Channels() );
	}

	if( GetInputCount() == 2 && !selfAttention->GetUseMask() ) {
		selfAttention->SetUseMask( true );
		SetInputMapping( 1, *selfAttention, 3 );
	} else if( GetInputCount() == 1 && selfAttention->GetUseMask() ) {
		selfAttention->SetUseMask( false );
	}

	CCompositeLayer::Reshape();
}

void CTransformerEncoderLayer::buildLayer()
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
	selfAttentionSum = FINE_DEBUG_NEW CEltwiseSumLayer( MathEngine() );
	selfAttentionSum->SetName( selfAttentionSumName );
	SetInputMapping( 0, *selfAttentionSum, 0 );
	selfAttentionSum->Connect( 1, *selfAttention );
	AddLayer( *selfAttentionSum );

	// Normalize the sum
	CPtr<CObjectNormalizationLayer> selfAttentionNorm = FINE_DEBUG_NEW CObjectNormalizationLayer( MathEngine() );
	selfAttentionNorm->SetName( "SelfAttentionNorm" );
	selfAttentionNorm->Connect( *selfAttentionSum );
	AddLayer( *selfAttentionNorm );

	// First fully-connected of feed-forward
	fc1 = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	fc1->SetName( fc1Name );
	fc1->SetNumberOfElements( 1 );
	fc1->Connect( *selfAttentionNorm );
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
	feedForwardSum = FINE_DEBUG_NEW CEltwiseSumLayer( MathEngine() );
	feedForwardSum->SetName( feedForwardSumName );
	feedForwardSum->Connect( 0, *fc2 );
	feedForwardSum->Connect( 1, *selfAttentionNorm );
	AddLayer( *feedForwardSum );

	// Normalize output
	CPtr<CObjectNormalizationLayer> feedForwardNorm = FINE_DEBUG_NEW CObjectNormalizationLayer( MathEngine() );
	feedForwardNorm->SetName( "FeedForwardNorm" );
	feedForwardNorm->Connect( *feedForwardSum );
	AddLayer( *feedForwardNorm );

	SetOutputMapping( *feedForwardNorm );
}

void CTransformerEncoderLayer::addDropoutLayers()
{
	if( dropoutFc1 != nullptr ) {
		NeoPresume( dropoutSelfAttention != nullptr && dropoutFc1 != nullptr && dropoutFc2 != nullptr );
		return;
	}

	NeoPresume( dropoutSelfAttention == nullptr && dropoutFc1 == nullptr && dropoutFc2 == nullptr );

	dropoutSelfAttention = FINE_DEBUG_NEW CDropoutLayer( MathEngine() );
	dropoutSelfAttention->SetName( dropoutSelfAttentionName );
	dropoutSelfAttention->Connect( *selfAttention );
	selfAttentionSum->Connect( 1, *dropoutSelfAttention );
	AddLayer( *dropoutSelfAttention );

	dropoutFc1 = FINE_DEBUG_NEW CDropoutLayer( MathEngine() );
	dropoutFc1->SetName( dropoutFc1Name );
	dropoutFc1->Connect( activationName );
	fc2->Connect( *dropoutFc1 );
	AddLayer( *dropoutFc1 );

	dropoutFc2 = FINE_DEBUG_NEW CDropoutLayer( MathEngine() );
	dropoutFc2->SetName( dropoutFc2Name );
	dropoutFc2->Connect( *fc2 );
	feedForwardSum->Connect( *dropoutFc2 );
	AddLayer( *dropoutFc2 );

	NeoPresume( dropoutSelfAttention != nullptr
		&& dropoutSelfAttention->GetName() == CString( dropoutSelfAttentionName ) );
	NeoPresume( dropoutFc1 != nullptr && dropoutFc1->GetName() == CString( dropoutFc1Name ) );
	NeoPresume( dropoutFc2 != nullptr && dropoutFc2->GetName() == CString( dropoutFc2Name ) );
}

void CTransformerEncoderLayer::removeDropoutLayers()
{
	if( dropoutFc1 == nullptr ) {
		NeoPresume( dropoutFc1 == nullptr && dropoutFc2 == nullptr && dropoutSelfAttention == nullptr );
		return;
	}

	NeoPresume( dropoutFc1 != nullptr && dropoutFc2 != nullptr && dropoutSelfAttention != nullptr );

	DeleteLayer( *dropoutSelfAttention );
	dropoutSelfAttention = nullptr;
	selfAttentionSum->Connect( 1, *selfAttention );

	DeleteLayer( *dropoutFc1 );
	dropoutFc1 = nullptr;
	fc2->Connect( activationName );

	DeleteLayer( *dropoutFc2 );
	dropoutFc2 = nullptr;
	feedForwardSum->Connect( *fc2 );

	NeoPresume( dropoutSelfAttention == nullptr && !HasLayer( dropoutSelfAttentionName ) );
	NeoPresume( dropoutFc1 == nullptr && !HasLayer( dropoutFc1Name ) );
	NeoPresume( dropoutFc2 == nullptr && !HasLayer( dropoutFc2Name ) );
}

CLayerWrapper<CTransformerEncoderLayer> TransformerEncoder( int headCount, int hiddenSize,
	float dropout, int feedForwardSize, TActivationFunction activation )
{
	return CLayerWrapper<CTransformerEncoderLayer>( "CTransformerEncoderLayer", [=]( CTransformerEncoderLayer* result ) {
		result->SetHeadCount( headCount );
		result->SetHiddenSize( hiddenSize );
		result->SetDropoutRate( dropout );
		result->SetFeedForwardSize( feedForwardSize );
		result->SetActivation( activation );
	} );
}

} // namespace NeoML
