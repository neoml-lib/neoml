/* Copyright © 2017-2022 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/TransformerSourceMaskLayer.h>

namespace NeoML {

CTransformerSourceMaskLayer::CTransformerSourceMaskLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "TransformerSourceMaskLayer", false ),
	headCount( 1 )
{
}

void CTransformerSourceMaskLayer::Reshape()
{
	CheckInputs();

	NeoAssert( inputDescs.Size() == 2 && outputDescs.Size() == 1 );

	CheckArchitecture( inputDescs[I_Widths].BatchWidth() == inputDescs[I_Q].BatchWidth(), GetName(),
		"mask input batchWidth mismatch" );

	outputDescs[0].SetDimSize( BD_BatchWidth, inputDescs[1].BatchWidth() );
	outputDescs[0].SetDimSize( BD_ListSize, headCount );
	outputDescs[0].SetDimSize( BD_Width, inputDescs[1].ListSize() );
	outputDescs[0].SetDimSize( BD_Channels, inputDescs[1].ListSize() );
	outputDescs[0].SetDataType( CT_Float );
}

void CTransformerSourceMaskLayer::RunOnce()
{
	NeoAssert( inputBlobs.Size() == 2 && outputBlobs.Size() == 1 );

	CPtr<CDnnBlob> inputBlob = inputBlobs[I_Widths];
	CPtr<CDnnBlob> outputBlob = outputBlobs.First();
	CConstIntHandle inputHandle = inputBlob->GetData<const int>();
	CFloatHandle outputHandle = outputBlob->GetData();

	outputBlob->Fill( 0.f );

	int objectPosition = 0;
	const int maxWidth = outputBlob->GetWidth();
	for( int objectNum = 0; objectNum < outputBlob->GetBatchWidth(); ++objectNum ) {
		const int objectWidth = min( inputHandle.GetValueAt( objectNum ), maxWidth );
		const int padding = maxWidth - objectWidth;
		for( int head = 0; head < headCount; ++head ) {
			for( int width = 0; width < maxWidth; ++width ) {
				ptrdiff_t shift = 1ll * objectPosition + objectWidth;
				if( padding > 0 ) {
					MathEngine().VectorFill( outputHandle + shift, 1.f, padding );
				}
				objectPosition += maxWidth;
			}
		}
	}
}

void CTransformerSourceMaskLayer::BackwardOnce()
{
	// No action
}

static const int TransformerSourceMaskLayerVersion = 0;

void CTransformerSourceMaskLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( TransformerSourceMaskLayerVersion );
	CBaseLayer::Serialize( archive );
	archive.Serialize( headCount );
}

CLayerWrapper<CTransformerSourceMaskLayer> TransformerSourceMask( int headCount )
{
	return CLayerWrapper<CTransformerSourceMaskLayer>( "TransformerSourceMaskLayer", [=]( CTransformerSourceMaskLayer* result ) {
		result->SetHeadCount( headCount );
	} );
}

}  // namespace NeoML
