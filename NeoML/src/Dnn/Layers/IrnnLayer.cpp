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

#include <NeoML/Dnn/Layers/IrnnLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// Fixed names for internal layers
static const char* inputFcName = "inputFc";
static const char* recurFcName = "recurFc";
static const char* backLinkName = "backLink";

CIrnnLayer::CIrnnLayer( IMathEngine& mathEngine ) :
	CRecurrentLayer( mathEngine, "IrnnLayer" ),
	identityScale( 1.f ),
	inputWeightStd( 1e-3f )
{
	buildLayer();
}

static const int IrnnLayerVersion = 0;

void CIrnnLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( IrnnLayerVersion );
	CRecurrentLayer::Serialize( archive );

	archive.Serialize( identityScale );
	archive.Serialize( inputWeightStd );

	if( archive.IsLoading() ) {
		inputFc = CheckCast<CFullyConnectedLayer>( GetLayer( inputFcName ) );
		recurFc = CheckCast<CFullyConnectedLayer>( GetLayer( recurFcName ) );
		backLink = CheckCast<CBackLinkLayer>( GetLayer( backLinkName ) );
	}
}

void CIrnnLayer::SetHiddenSize( int size )
{
	inputFc->SetNumberOfElements( size );
	recurFc->SetNumberOfElements( size );
	backLink->SetDimSize( BD_Channels, size );
}

void CIrnnLayer::Reshape()
{
	CPtr<CDnnBlob> prevInputWeight = inputFc->GetWeightsData();
	CPtr<CDnnBlob> prevRecurWeight = recurFc->GetWeightsData();

	CRecurrentLayer::Reshape();

	CPtr<CDnnBlob> inputWeight = inputFc->GetWeightsData();
	CPtr<CDnnBlob> recurWeight = recurFc->GetWeightsData();

	if( prevInputWeight == nullptr || prevRecurWeight == nullptr // is first initialization
		|| !inputWeight->HasEqualDimensions( prevInputWeight ) // something changed the shape of the weight
		|| !recurWeight->HasEqualDimensions( prevRecurWeight ) )
	{
		// Initialization is needed
		// CRecurrentLayer::Reshape intialized weights with intializer from network
		// Here we overwrite previously initialized weights with IRNN-specific values
		identityInitialization( *recurWeight );
		recurFc->SetWeightsData( recurWeight );

		normalInitialization( *inputWeight );
		inputFc->SetWeightsData( inputWeight );
	}
}

// Creates and connects layers in internal dnn
void CIrnnLayer::buildLayer()
{
	backLink = new CBackLinkLayer( MathEngine() );
	backLink->SetName( backLinkName );
	AddBackLink( *backLink );

	inputFc = new CFullyConnectedLayer( MathEngine() );
	inputFc->SetName( inputFcName );
	SetInputMapping( 0, *inputFc, 0 );
	AddLayer( *inputFc );

	recurFc = new CFullyConnectedLayer( MathEngine() );
	recurFc->SetName( recurFcName );
	recurFc->Connect( *backLink );
	AddLayer( *recurFc );

	CPtr<CEltwiseSumLayer> sum = new CEltwiseSumLayer( MathEngine() );
	sum->Connect( 0, *inputFc );
	sum->Connect( 1, *recurFc );
	AddLayer( *sum );

	CPtr<CReLULayer> relu = new CReLULayer( MathEngine() );
	relu->Connect( *sum );
	AddLayer( *relu );

	SetOutputMapping( *relu );
	backLink->Connect( *relu );
}

// Fills blob with scaled identity matrix
void CIrnnLayer::identityInitialization( CDnnBlob& blob )
{
	// Only NxN matrix can be initialized in this manner
	const int objectSize = blob.GetObjectSize();
	NeoAssert( blob.GetObjectCount() == objectSize );

	float* buff = blob.GetBuffer<float>( 0, objectSize * objectSize, false );
	for( int i = 0; i < blob.GetDataSize(); ++i ) {
		buff[i] = i % objectSize == i / objectSize ? identityScale : 0.f;
	}
	blob.ReleaseBuffer( buff, true );
}

// Fills blob with values from N(0, inputWeightStd)
void CIrnnLayer::normalInitialization( CDnnBlob& blob )
{
	const int dataSize = blob.GetDataSize();

	float* buff = blob.GetBuffer<float>( 0, dataSize, false );
	CRandom& random = GetDnn()->Random();
	for( int i = 0; i < dataSize; ++i ) {
		buff[i] = static_cast<float>( random.Normal( 0., inputWeightStd ) );
	}
	blob.ReleaseBuffer( buff, true );
}

} // namespace NeoML
