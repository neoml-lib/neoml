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

#include <NeoML/Dnn/Layers/IndRnnLayer.h>

namespace NeoML {

// Names of layers inside of CIndRnnLayer
static const char* inputDropoutName = "InputDropout";
static const char* fcName = "Fc";
static const char* recurrentName = "IndRnnRecurrent";

CIndRnnLayer::CIndRnnLayer( IMathEngine& mathEngine ) :
	CCompositeLayer( mathEngine )
{
	buildLayer();
}

static const int IndRnnLayerVersion = 0;

void CIndRnnLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( IndRnnLayerVersion );
	CCompositeLayer::Serialize( archive );

	if( archive.IsLoading() ) {
		fc = CheckCast<CFullyConnectedLayer>( GetLayer( fcName ) );
		recurrent = CheckCast<CIndRnnRecurrentLayer>( GetLayer( recurrentName ) );

		if( HasLayer( inputDropoutName ) ) {
			inputDropout = CheckCast<CDropoutLayer>( GetLayer( inputDropoutName ) );
		} else {
			inputDropout = nullptr;
		}
	}
}

void CIndRnnLayer::SetHiddenSize( int hiddenSize )
{
	NeoAssert( hiddenSize > 0 );
	if( GetHiddenSize() == hiddenSize ) {
		return;
	}

	fc->SetNumberOfElements( hiddenSize );
	ForceReshape();
}

void CIndRnnLayer::SetDropoutRate( float rate )
{
	if( rate > 0 ) {
		if( inputDropout == nullptr ) {
			// Add required dropout layer
			inputDropout = new CDropoutLayer( MathEngine() );
			inputDropout->SetName( inputDropoutName );
			AddLayer( *inputDropout );
			SetInputMapping( *inputDropout );
			fc->Connect( *inputDropout );
		}
		inputDropout->SetDropoutRate( rate );
		recurrent->SetDropoutRate( rate );
	} else {
		if( inputDropout != nullptr ) {
			DeleteLayer( *inputDropout );
			SetInputMapping( *fc );
			inputDropout = nullptr;
		}
		recurrent->SetDropoutRate( rate );
	}
}

bool CIndRnnLayer::IsReverseSequence() const
{
	return recurrent->IsReverseSequence();
}

void CIndRnnLayer::SetReverseSequence( bool reverse )
{
	recurrent->SetReverseSequence( reverse );
}

CPtr<CDnnBlob> CIndRnnLayer::GetRecurrentWeights() const
{
	return recurrent->GetWeights();
}

void CIndRnnLayer::SetRecurrentWeights( const CDnnBlob* recurrentWeights )
{
	recurrent->SetWeights( recurrentWeights );
}

TActivationFunction CIndRnnLayer::GetActivation() const
{
	return recurrent->GetActivation();
}

void CIndRnnLayer::SetActivation( TActivationFunction activation )
{
	recurrent->SetActivation( activation );
}

void CIndRnnLayer::buildLayer()
{
	fc = new CFullyConnectedLayer( MathEngine() );
	fc->SetName( fcName );
	AddLayer( *fc );
	SetInputMapping( *fc );

	recurrent = new CIndRnnRecurrentLayer( MathEngine() );
	recurrent->SetName( recurrentName );
	AddLayer( *recurrent );
	recurrent->Connect( *fc );

	SetOutputMapping( *recurrent );
}

// --------------------------------------------------------------------------------------------------------------------
// CIndRnnRecurrentLayer

CIndRnnRecurrentLayer::CIndRnnRecurrentLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CIndRnnRecurrentLayer", true ),
	activation( AF_Sigmoid ),
	reverse( false ),
	dropoutRate( -1.f ),
	dropoutMask( nullptr )
{
	paramBlobs.SetSize( 1 );
}

static const int IndRnnRecurrentLayerVersion = 1;

void CIndRnnRecurrentLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( IndRnnRecurrentLayerVersion );
	CBaseLayer::Serialize( archive );
	archive.Serialize( reverse );
	archive.Serialize( dropoutRate );
	NeoPresume( dropoutMask == nullptr );

	// v1 - activation added
	if( version == 0 ) {
		activation = AF_Sigmoid;
	} else {
		int activationInt = static_cast<int>( activation );
		archive.Serialize( activationInt );
		activation = static_cast<TActivationFunction>( activationInt );
	}
}

void CIndRnnRecurrentLayer::SetDropoutRate( float rate )
{
	NeoAssert( rate < 1.f );
	dropoutRate = rate;
}

void CIndRnnRecurrentLayer::SetActivation( TActivationFunction newActivation )
{
	NeoAssert( newActivation == AF_Sigmoid || newActivation == AF_ReLU );
	activation = newActivation;
}

CPtr<CDnnBlob> CIndRnnRecurrentLayer::GetWeights() const
{
	if( paramBlobs[0] == nullptr ) {
		return nullptr;
	}

	return paramBlobs[0]->GetCopy();
}

void CIndRnnRecurrentLayer::SetWeights( const CDnnBlob* newWeights )
{
	if( newWeights == nullptr ) {
		paramBlobs[0] = nullptr;
		ForceReshape();
	} else if( paramBlobs[0] != nullptr && GetDnn() != nullptr ) {
		NeoAssert( paramBlobs[0]->GetDataSize() == newWeights->GetDataSize() );
		paramBlobs[0]->CopyFrom( newWeights );
	} else {
		paramBlobs[0] = newWeights->GetCopy();
	}
}

void CIndRnnRecurrentLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( inputDescs.Size() == 1, GetName(), "IndRNN recurrent with more than 1 input" );

	outputDescs[0] = inputDescs[0];

	CBlobDesc paramDesc = inputDescs[0];
	paramDesc.SetDimSize( BD_BatchLength, 1 );
	paramDesc.SetDimSize( BD_BatchWidth, 1 );
	paramDesc.SetDimSize( BD_ListSize, 1 );
	if( paramBlobs[0] == nullptr ) {
		paramBlobs[0] = CDnnBlob::CreateBlob( MathEngine(), CT_Float, paramDesc );
		InitializeParamBlob( 0, *paramBlobs[0] );
	} else {
		NeoAssert( paramBlobs[0]->GetDataSize() == paramDesc.BlobSize() );
	}
}

void CIndRnnRecurrentLayer::RunOnce()
{
	const int sequenceLength = inputBlobs[0]->GetBatchLength();
	const int batchSize = inputBlobs[0]->GetBatchWidth() * inputBlobs[0]->GetListSize();
	const int objectSize = inputBlobs[0]->GetObjectSize();

	if( IsBackwardPerformed() && dropoutRate > 0 ) {
		NeoPresume( dropoutMask == nullptr );
		dropoutMask.reset( new CFloatHandleVar( MathEngine(), batchSize * objectSize ) );
		MathEngine().VectorFillBernoulli( dropoutMask->GetHandle(), 1.f - dropoutRate, batchSize * objectSize,
			1.f / ( 1.f - dropoutRate ), GetDnn()->Random().Next() );
	}

	MathEngine().IndRnnRecurrent( reverse, sequenceLength, batchSize, objectSize, activation,
		inputBlobs[0]->GetData(), maskHandle(), paramBlobs[0]->GetData(), outputBlobs[0]->GetData() );
}

void CIndRnnRecurrentLayer::BackwardOnce()
{
	const int sequenceLength = inputBlobs[0]->GetBatchLength();
	const int batchSize = inputBlobs[0]->GetBatchWidth() * inputBlobs[0]->GetListSize();
	const int objectSize = inputBlobs[0]->GetObjectSize();

	NeoPresume( ( dropoutRate <= 0.f && dropoutMask == nullptr ) ||
		( dropoutRate > 0.f && dropoutMask != nullptr ) );

	MathEngine().IndRnnRecurrentBackward( !reverse, sequenceLength, batchSize, objectSize, activation,
		maskHandle(), paramBlobs[0]->GetData(), outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData() );

	if( !IsLearningPerformed() && dropoutMask != nullptr ) {
		dropoutMask.reset( nullptr );
	}
}

void CIndRnnRecurrentLayer::LearnOnce()
{
	const int sequenceLength = inputBlobs[0]->GetBatchLength();
	const int batchSize = inputBlobs[0]->GetBatchWidth() * inputBlobs[0]->GetListSize();
	const int objectSize = inputBlobs[0]->GetObjectSize();

	NeoPresume( ( dropoutRate <= 0.f && dropoutMask == nullptr ) ||
		( dropoutRate > 0.f && dropoutMask != nullptr ) );

	MathEngine().IndRnnRecurrentLearn( !reverse, sequenceLength, batchSize, objectSize, activation,
		maskHandle(), paramBlobs[0]->GetData(), outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		paramDiffBlobs[0]->GetData() );
	
	if( dropoutMask != nullptr ) {
		dropoutMask.reset( nullptr );
	}
}

// Returns dropout mask handle (or null handle if dropout is not initialized)
CConstFloatHandle CIndRnnRecurrentLayer::maskHandle() const
{
	return dropoutMask == nullptr ? CConstFloatHandle() : dropoutMask->GetHandle();
}

// --------------------------------------------------------------------------------------------------------------------

CLayerWrapper<CIndRnnLayer> IndRnn( int hiddenSize, float dropoutRate, bool reverse, TActivationFunction activation )
{
	return CLayerWrapper<CIndRnnLayer>( "IndRnn", [=]( CIndRnnLayer* result ) {
		result->SetHiddenSize( hiddenSize );
		result->SetDropoutRate( dropoutRate );
		result->SetReverseSequence( reverse );
		result->SetActivation( activation );
	} );
}

} // namespace NeoML
