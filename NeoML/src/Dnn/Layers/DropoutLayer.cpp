/* Copyright © 2017-2024 ABBYY

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

#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CDropoutLayer::CDropoutLayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CCnnDropoutLayer" ),
	desc( nullptr ),
	dropoutRate( 0 ),
	isSpatial( false ),
	isBatchwise( false )
{
}

static const int DropoutLayerVersion = 2000;

void CDropoutLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( DropoutLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );

	archive.Serialize( dropoutRate );
	archive.Serialize( isSpatial );
	archive.Serialize( isBatchwise );

	if( archive.IsLoading() ) {
		OnReshaped();
	}
}

void CDropoutLayer::SetDropoutRate( float value )
{
	NeoAssert( value >= 0.f && value < 1.f );
	if( dropoutRate != value ) {
		dropoutRate = value;
		if( GetDnn() != nullptr) {
			OnReshaped();
		}
	}
}

void CDropoutLayer::SetSpatial( bool value )
{
	if( value != isSpatial ) {
		isSpatial = value;
		if( GetDnn() != nullptr ) {
			OnReshaped();
		}
	}
}

void CDropoutLayer::SetBatchwise( bool value )
{
	if( value != isBatchwise ) {
		isBatchwise = value;
		if( GetDnn() != nullptr ) {
			OnReshaped();
		}
	}
}

void CDropoutLayer::OnReshaped()
{
	destroyDropoutDesc();
	initDropoutDesc();
}

void CDropoutLayer::RunOnce()
{
	CheckInput1();

	if( !IsBackwardPerformed() ) {
		MathEngine().VectorCopy( outputBlobs[0]->GetData(), inputBlobs[0]->GetData(),
			inputBlobs[0]->GetDataSize() );
		return;
	}

	MathEngine().UpdateDropout(desc, &(inputBlobs[0]->GetDesc()), &(outputBlobs[0]->GetDesc()),
		GetDnn()->Random().Next(), true);

	MathEngine().Dropout( *desc, inputBlobs[0]->GetData(), outputBlobs[0]->GetData() );
}

void CDropoutLayer::BackwardOnce()
{
	// Backward pass is only possible when learning
	NeoAssert( desc != nullptr );

	MathEngine().Dropout( *desc, outputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetData() );

	if( !GetDnn()->IsRecurrentMode() || GetDnn()->IsFirstSequencePos() ) {
		// Clear the memory after the whole sequence is processed
		disableDropoutDesc();
	}
}

CDropoutLayer::~CDropoutLayer()
{
	destroyDropoutDesc();
}

void CDropoutLayer::disableDropoutDesc()
{
	MathEngine().UpdateDropout(desc, nullptr, nullptr, 0, false);
}

void CDropoutLayer::destroyDropoutDesc()
{
	if( desc != nullptr ) {
		delete desc;
		desc = nullptr;
	}
}

void CDropoutLayer::initDropoutDesc()
{
	if (desc == nullptr) {
		desc = MathEngine().InitDropout(dropoutRate, isSpatial, isBatchwise);
	}
}

CLayerWrapper<CDropoutLayer> Dropout( float dropoutRate,
	bool isSpatial, bool isBatchwise )
{
	return CLayerWrapper<CDropoutLayer>( "Dropout", [=]( CDropoutLayer* result ) {
		result->SetSpatial( isSpatial );
		result->SetBatchwise( isBatchwise );
		result->SetDropoutRate( dropoutRate );
	} );
}

} // namespace NeoML
