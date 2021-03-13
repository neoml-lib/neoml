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

#include <NeoML/Dnn/Layers/Upsampling2DLayer.h>

namespace NeoML {

CUpsampling2DLayer::CUpsampling2DLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnUpsampling2DLayer", false ),
	heightCopyCount( 0 ),
	widthCopyCount( 0 )
{
}

void CUpsampling2DLayer::SetHeightCopyCount( int newHeightCopyCount )
{
	NeoAssert( newHeightCopyCount > 0 );
	if( heightCopyCount != newHeightCopyCount ) {
		heightCopyCount = newHeightCopyCount;
		ForceReshape();
	}
}

void CUpsampling2DLayer::SetWidthCopyCount( int newWidthCopyCount )
{
	NeoAssert( newWidthCopyCount > 0 );
	if( widthCopyCount != newWidthCopyCount ) {
		widthCopyCount = newWidthCopyCount;
		ForceReshape();
	}
}

void CUpsampling2DLayer::Reshape()
{
	CheckInputs();
	CheckOutputs();

	NeoAssert( heightCopyCount > 0 );
	NeoAssert( widthCopyCount > 0 );
	NeoAssert( GetInputCount() == GetOutputCount() );

	for( int i = 0; i < GetInputCount(); ++i ) {
		NeoAssert( inputDescs[i].BatchLength() == 1 );

		outputDescs[0] = inputDescs[0];
		outputDescs[0].SetDimSize(BD_Height, heightCopyCount * inputDescs[i].Height());
		outputDescs[0].SetDimSize(BD_Width, widthCopyCount * inputDescs[i].Width());
	}
}

void CUpsampling2DLayer::RunOnce()
{
	NeoAssert( inputBlobs.Size() == outputBlobs.Size() );

	for( int i = 0; i < inputBlobs.Size(); ++i ) {
		MathEngine().Upsampling2DForward( inputBlobs[i]->GetDesc(), inputBlobs[i]->GetData(),
			heightCopyCount, widthCopyCount, outputBlobs[i]->GetDesc(), outputBlobs[i]->GetData() );
	}
}

void CUpsampling2DLayer::BackwardOnce()
{
	NeoAssert( inputDiffBlobs.Size() == outputDiffBlobs.Size() );
	for( int i = 0; i < inputDiffBlobs.Size(); ++i ) {
		MathEngine().Upsampling2DBackward( outputDiffBlobs[i]->GetDesc(), outputDiffBlobs[i]->GetData(),
			heightCopyCount, widthCopyCount, inputDiffBlobs[i]->GetDesc(), inputDiffBlobs[i]->GetData() );
	}
}

static const int Upsampling2DLayerVersion = 2000;

void CUpsampling2DLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( Upsampling2DLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( heightCopyCount );
	archive.Serialize( widthCopyCount );
}

CLayerWrapper<CUpsampling2DLayer> Upsampling2d( int heightCopyCount, int widthCopyCount )
{
	return CLayerWrapper<CUpsampling2DLayer>( "Upsampling2d", [=]( CUpsampling2DLayer* result ) {
		result->SetHeightCopyCount( heightCopyCount );
		result->SetWidthCopyCount( widthCopyCount );
	} );
}

} // namespace NeoML
