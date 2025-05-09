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

#include <NeoML/Dnn/Layers/DropoutLayer.h>

namespace NeoML {

CDropoutLayer::CDropoutLayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CCnnDropoutLayer" ),
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
		destroyDropoutDescs();
	}
}

void CDropoutLayer::SetDropoutRate( float value )
{
	NeoAssert( value >= 0.f && value < 1.f );
	if( dropoutRate != value ) {
		dropoutRate = value;
		if( GetDnn() != 0 ) {
			destroyDropoutDescs();
		}
	}
}

void CDropoutLayer::SetSpatial( bool value )
{
	if( value != isSpatial ) {
		isSpatial = value;
		if( GetDnn() != 0 ) {
			destroyDropoutDescs();
		}
	}
}

void CDropoutLayer::SetBatchwise( bool value )
{
	if( value != isBatchwise ) {
		isBatchwise = value;
		if( GetDnn() != 0 ) {
			destroyDropoutDescs();
		}
	}
}

void CDropoutLayer::OnReshaped()
{
	destroyDropoutDescs();
}

void CDropoutLayer::RunOnce()
{
	if( !IsBackwardPerformed() ) {
		for( int i = 0; i < inputBlobs.Size(); ++i ) {
			MathEngine().VectorCopy( outputBlobs[i]->GetData(), inputBlobs[i]->GetData(),
				inputBlobs[i]->GetDataSize() );
		}
		return;
	}

	initDropoutDescs();

	for( int i = 0; i < inputBlobs.Size(); ++i ) {
		MathEngine().Dropout( *descs[i], inputBlobs[0]->GetData(), outputBlobs[0]->GetData());
	}
}

void CDropoutLayer::BackwardOnce()
{
	for( int i = 0; i < outputDiffBlobs.Size(); ++i ) {
		// Backward pass is only possible when learning
		NeoAssert( descs[i] != 0 );

		MathEngine().Dropout( *descs[i], outputDiffBlobs[i]->GetData(), inputDiffBlobs[i]->GetData());
	}

	if( !GetDnn()->IsRecurrentMode() || GetDnn()->IsFirstSequencePos() ) {
		// Clear the memory after the whole sequence is processed
		destroyDropoutDescs();
	}
}

void CDropoutLayer::initDropoutDescs()
{
	descs.SetSize( inputBlobs.Size() );
	for( int i = 0; i < descs.Size(); ++i ) {
		if( descs[i] == nullptr ) {
			descs.ReplaceAt( MathEngine().InitDropout( dropoutRate, isSpatial, isBatchwise, inputBlobs[0]->GetDesc(),
				outputBlobs[0]->GetDesc(), GetDnn()->Random().Next() ), i );
		}
	}
}

void CDropoutLayer::destroyDropoutDescs()
{
	for( int i = 0; i < descs.Size(); ++i ) {
		descs.ReplaceAt( nullptr, i );
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
