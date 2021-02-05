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

#include <NeoML/Dnn/Layers/LrnLayer.h>

namespace NeoML {

CLrnLayer::CLrnLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CLrnLayer", false ),
	desc( nullptr ),
	windowSize( 0 ),
	bias( 1.f ),
	alpha( 1e-4f ),
	beta( 0.75f )
{}

static const int LrnLayerVersion = 0;

void CLrnLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( LrnLayerVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( windowSize );
	archive.Serialize( bias );
	archive.Serialize( alpha );
	archive.Serialize( beta );
}

void CLrnLayer::SetWindowSize( int value )
{
	NeoAssert( value > 0 );
	if( value != windowSize ) {
		windowSize = value;
		if( GetDnn() != nullptr ) {
			destroyDesc();
		}
	}
}

void CLrnLayer::SetBias( float value )
{
	if( value != bias ) {
		bias = value;
		if( GetDnn() != nullptr ) {
			destroyDesc();
		}
	}
}

void CLrnLayer::SetAlpha( float value )
{
	if( value != alpha ) {
		alpha = value;
		if( GetDnn() != nullptr ) {
			destroyDesc();
		}
	}
}

void CLrnLayer::SetBeta( float value )
{
	if( value != beta ) {
		beta = value;
		if( GetDnn() != nullptr ) {
			destroyDesc();
		}
	}
}

void CLrnLayer::Reshape()
{
	CheckInputs();
	CheckOutputs();
	CheckArchitecture( GetInputCount() == 1, GetName(), "LRN with multiple inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetName(), "LRN with multiple outputs" );

	outputDescs[0] = inputDescs[0];

	if( IsBackwardPerformed() ) {
		invertedSum = CDnnBlob::CreateBlob( MathEngine(), inputDescs[0] );
		invertedSumBeta = CDnnBlob::CreateBlob( MathEngine(), inputDescs[0] );
		RegisterRuntimeBlob( invertedSum );
		RegisterRuntimeBlob( invertedSumBeta );
	} else {
		invertedSum = nullptr;
		invertedSumBeta = nullptr;
	}

	destroyDesc();
}

void CLrnLayer::RunOnce()
{
	initDesc();

	if( IsBackwardPerformed() ) {
		MathEngine().Lrn( *desc, inputBlobs[0]->GetData(), invertedSum->GetData(),
			invertedSumBeta->GetData(), outputBlobs[0]->GetData() );
	} else {
		MathEngine().Lrn( *desc, inputBlobs[0]->GetData(), CFloatHandle(), CFloatHandle(), outputBlobs[0]->GetData() );
	}
}

void CLrnLayer::BackwardOnce()
{
	MathEngine().LrnBackward( *desc, inputBlobs[0]->GetData(), outputBlobs[0]->GetData(),
		outputDiffBlobs[0]->GetData(), invertedSum->GetData(), invertedSumBeta->GetData(),
		inputDiffBlobs[0]->GetData() );
}

void CLrnLayer::initDesc()
{
	if( desc == nullptr ) {
		desc = MathEngine().InitLrn( inputBlobs[0]->GetDesc(), windowSize, bias, alpha, beta );
	}
}

void CLrnLayer::destroyDesc()
{
	if( desc != nullptr ) {
		delete desc;
		desc = nullptr;
	}
}

CLayerWrapper<CLrnLayer> Lrn( int windowSize, float bias, float alpha, float beta )
{
	return CLayerWrapper<CLrnLayer>( "Lrn", [=]( CLrnLayer* result ) {
		result->SetWindowSize( windowSize );
		result->SetBias( bias );
		result->SetAlpha( alpha );
		result->SetBeta( beta );
	} );
}

} // namespace NeoML
