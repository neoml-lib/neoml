/* Copyright © 2023 ABBYY

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

#include <NeoML/Dnn/Layers/LoraFullyConnectedLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

static const char* const baseFcName = "FullyConnectedBase";
static const char* const dropoutName = "Dropout";
static const char* const fcAName = "FullyConnectedA";
static const char* const fcBName = "FullyConnectedB";
static const char* const scalingName = "Scaling";
static const char* const sumName = "Sum";

//----------------------------------------------------------------------------------------------

CLoraFullyConnectedLayer::CLoraFullyConnectedLayer( IMathEngine& mathEngine ) :
	CCompositeLayer( mathEngine, nullptr )
{
	CLoraParams defaultParams;
	initialize( defaultParams );
}

CLoraFullyConnectedLayer::CLoraFullyConnectedLayer( CDnnBlob& baseWeights, CDnnBlob* baseFreeTerms,
		const CLoraParams& params ) :
	CCompositeLayer( baseWeights.GetMathEngine() )
{
	initialize( params );

	baseFc->SetNumberOfElements( baseWeights.GetObjectCount() );
	baseFc->Weights() = &baseWeights;
	baseFc->FreeTerms() = baseFreeTerms;
	baseFc->SetZeroFreeTerm( baseFreeTerms == nullptr );
}

static const int LoraFullyConnectedLayerVersion = 0;

template<typename T>
static void serializeTypedLayer( CArchive& archive, IMathEngine& mathEngine, CPtr<T>& layer )
{
	CPtr<CBaseLayer> baseLayer = layer.Ptr();
	SerializeLayer( archive, mathEngine, baseLayer );
	layer = CheckCast<T>( baseLayer );
}

void CLoraFullyConnectedLayer::Serialize( CArchive& archive )
{
	merge(); // guarantee that internal dnn contains only baseFc (other layers will be serialized manually)

	archive.SerializeVersion( LoraFullyConnectedLayerVersion );
	CCompositeLayer::Serialize( archive );

	serializeTypedLayer<CDropoutLayer>( archive, MathEngine(), dropout );
	serializeTypedLayer<CFullyConnectedLayer>( archive, MathEngine(), fcA );
	serializeTypedLayer<CFullyConnectedLayer>( archive, MathEngine(), fcB );
	serializeTypedLayer<CLinearLayer>( archive, MathEngine(), scaling );
	serializeTypedLayer<CEltwiseSumLayer>( archive, MathEngine(), sum );

	if( archive.IsLoading() ) {
		baseFc = CheckCast<CFullyConnectedLayer>( GetLayer( baseFcName ) );
	}
}

void CLoraFullyConnectedLayer::UpdateParams( const CLoraParams& newParams, CDnnBlob* newA, CDnnBlob* newB )
{
	split();

	dropout->SetDropoutRate( newParams.Dropout );
	fcA->SetNumberOfElements( newParams.Rank );
	scaling->SetMultiplier( newParams.Alpha );

	fcA->Weights() = newA;
	fcB->Weights() = newB;
}

void CLoraFullyConnectedLayer::Reshape()
{
	CheckLayerArchitecture( GetInputCount() == 1, "Layer must have only 1 input" );
	CheckLayerArchitecture( GetOutputCount() == 1, "Layer must have only 1 output" );

	bool needsInitialization = false;
	if( IsBackwardPerformed() || IsLearningPerformed() ) {
		split();
		needsInitialization = fcA->Weights() == nullptr;
	} else {
		merge();
	}

	CCompositeLayer::Reshape();

	if( needsInitialization ) {
		NeoAssert( fcB->Weights() != nullptr );
		fcB->Weights()->Clear();
	}
}

void CLoraFullyConnectedLayer::initialize( const CLoraParams& params )
{
	baseFc = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	baseFc->SetName( baseFcName );
	baseFc->DisableLearning();
	AddLayer( *baseFc );
	SetInputMapping( *baseFc );
	SetOutputMapping( *baseFc );

	dropout = FINE_DEBUG_NEW CDropoutLayer( MathEngine() );
	dropout->SetName( dropoutName );
	dropout->SetDropoutRate( params.Dropout );

	fcA = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	fcA->SetName( fcAName );
	fcA->SetZeroFreeTerm( true );
	fcA->SetNumberOfElements( params.Rank );

	fcB = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	fcB->SetName( fcBName );
	fcB->SetZeroFreeTerm( true );

	scaling = FINE_DEBUG_NEW CLinearLayer( MathEngine() );
	scaling->SetName( scalingName );
	scaling->SetMultiplier( params.Alpha );
	scaling->SetFreeTerm( 0.f );

	sum = FINE_DEBUG_NEW CEltwiseSumLayer( MathEngine() );
	sum->SetName( sumName );
}

void CLoraFullyConnectedLayer::merge()
{
	if( isMerged ) {
		return;
	}

	isMerged = true;

	if( dropout->GetDnn() != nullptr ) {
		DeleteLayer( *dropout );
	}
	DeleteLayer( *fcA );
	DeleteLayer( *fcB );
	if( scaling->GetDnn() != nullptr ) {
		DeleteLayer( *scaling );
	}
	DeleteLayer( *sum );

	SetInputMapping( *baseFc );
	SetOutputMapping( *baseFc );

	recalcBaseWeights();
}

void CLoraFullyConnectedLayer::split()
{
	if( !isMerged ) {
		return;
	}

	isMerged = false;

	AddLayer( *fcA );
	if( dropout->GetDropoutRate() != 0.f ) {
		AddLayer( *dropout );
		SetInputMapping( *dropout );
		fcA->Connect( *dropout );
	} else {
		SetInputMapping( *fcA );
	}

	AddLayer( *fcB );
	fcB->Connect( *fcA );
	if( fcB->Weights() == nullptr ) {
		fcB->SetNumberOfElements( baseFc->GetNumberOfElements() );
	}

	AddLayer( *sum );
	sum->Connect( 0, *baseFc );

	if( scaling->GetMultiplier() != 1.f ) {
		AddLayer( *scaling );
		scaling->Connect( *fcB );
		sum->Connect( 1, *scaling );
	} else {
		sum->Connect( 1, *fcB );
	}

	SetOutputMapping( *sum );

	recalcBaseWeights();
}

void CLoraFullyConnectedLayer::recalcBaseWeights()
{
	// isMerged is a newly changed state
	if( fcA->Weights() == nullptr ) {
		NeoAssert( fcB->Weights() == nullptr );
		// weights weren't initalized
		// the weights will be initialized in a way that untrained lora won't affect base weights
		// as a results we can relax for now
		return;
	}

	const int inputSize = fcA->Weights()->GetObjectSize();
	const int rank = fcA->Weights()->GetObjectCount();
	const int outputSize = fcB->GetNumberOfElements();
	const int bSize = rank * outputSize;

	CConstFloatHandle a = fcA->Weights()->GetData();
	CConstFloatHandle b = fcB->Weights()->GetData();

	CFloatHandleStackVar buff( MathEngine(), fcB->Weights()->GetDataSize() + static_cast<size_t>( 1 ) );
	CFloatHandle bTransposed = buff.GetHandle();
	CFloatHandle mult = buff.GetHandle() + bSize;

	MathEngine().TransposeMatrix( 1, b, outputSize, 1, rank, 1, bTransposed, buff.Size() - 1 );

	// during split we must substract A*B from merged weights
	const float multValue = isMerged ? scaling->GetMultiplier() : -scaling->GetMultiplier();
	if( multValue != 1 ) {
		mult.SetValue( multValue );
		MathEngine().VectorMultiply( bTransposed, bTransposed, bSize, mult );
	}

	MathEngine().MultiplyTransposedMatrixByMatrixAndAdd( bTransposed, rank, outputSize, outputSize,
		a, inputSize, inputSize, baseFc->Weights()->GetData(), inputSize, inputSize * outputSize );
}

} // namespace NeoML
