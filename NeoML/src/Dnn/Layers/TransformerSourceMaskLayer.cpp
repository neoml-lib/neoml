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

	outputDescs[0].SetDimSize( BD_BatchWidth, inputDescs[I_Q].BatchWidth() );
	outputDescs[0].SetDimSize( BD_ListSize, headCount );
	outputDescs[0].SetDimSize( BD_Width, inputDescs[I_Q].ListSize() );
	outputDescs[0].SetDimSize( BD_Channels, inputDescs[I_Q].ListSize() );
	outputDescs[0].SetDataType( CT_Float );
}

void CTransformerSourceMaskLayer::RunOnce()
{
	NeoAssert( inputBlobs.Size() == 2 && outputBlobs.Size() == 1 );

	CDnnBlobBuffer<int> inputBuffer( *inputBlobs[I_Widths], 0,
		inputBlobs[I_Widths]->GetDataSize(), TDnnBlobBufferAccess::Read );
	CPtr<CDnnBlob> outputBlob = outputBlobs.First();

	outputBlob->Fill( 0.f );

	ptrdiff_t objectPosition = 0;
	const int maxWidth = outputBlob->GetWidth();
	for( int objectNum = 0; objectNum < outputBlob->GetBatchWidth(); ++objectNum ) {
		if( inputBuffer[objectNum] == 0 ) {
			continue;
		}

		const int objectWidth = min( inputBuffer[objectNum], maxWidth );
		const int padding = maxWidth - objectWidth;
		if( padding > 0 ) {
			CFloatHandle objectHandle = outputBlob->GetData() + objectPosition;
			MathEngine().VectorFill( objectHandle + objectWidth, 1.f, padding );
			if( maxWidth * headCount > 1 ) {
				MathEngine().SetVectorToMatrixRows( objectHandle + maxWidth,
					maxWidth * headCount - 1, maxWidth, objectHandle );
			}
		}
		objectPosition += 1LL * headCount * maxWidth * maxWidth;
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
