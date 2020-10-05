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

#include <NeoML/Dnn/Layers/ProjectionPoolingLayer.h>

namespace NeoML {

CProjectionPoolingLayer::CProjectionPoolingLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CProjectionPoolingLayer", false ),
	direction( D_ByRows ),
	shouldRestoreOriginalImageSize( false ),
	desc( nullptr )
{
}

static const int currentVersion = 0;

void CProjectionPoolingLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( currentVersion );
	CBaseLayer::Serialize( archive );
	archive.SerializeEnum( direction );
	archive.Serialize( shouldRestoreOriginalImageSize );
}

void CProjectionPoolingLayer::Reshape()
{
	CheckInputs();
	CheckOutputs();
	CheckArchitecture( GetInputCount() == 1, GetName(), "Pooling with multiple inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetName(), "Pooling with multiple outputs" );
	CheckArchitecture( inputDescs[0].Depth() == 1 && inputDescs[0].BatchLength() == 1, GetName(),
		"Bad input blob dimensions" );

	outputDescs[0] = inputDescs[0];
	const TBlobDim dimensionToSqueeze = direction == D_ByRows ? BD_Width : BD_Height;
	if( shouldRestoreOriginalImageSize ) {
		CBlobDesc projectionResultBlobDesc = inputDescs[0];
		projectionResultBlobDesc.SetDimSize( dimensionToSqueeze, 1 );
		projectionResultBlob = CDnnBlob::CreateBlob( MathEngine(), projectionResultBlobDesc );

		RegisterRuntimeBlob( projectionResultBlob );
	} else {
		outputDescs[0].SetDimSize( dimensionToSqueeze, 1 );
	}

	destroyDesc();
}

void CProjectionPoolingLayer::RunOnce()
{
	const CBlobDesc& inputDesc = inputBlobs[0]->GetDesc();
	const CBlobDesc& outputDesc = outputBlobs[0]->GetDesc();
	initDesc();

	if( shouldRestoreOriginalImageSize ) {
		NeoAssert( projectionResultBlob != nullptr );
		// Calculate pooling result into the temporary blob
		MathEngine().BlobMeanPooling( *desc, inputBlobs[0]->GetData(), projectionResultBlob->GetData() );
		// Broadcatst pooling result along whole result blob
		outputBlobs[0]->Clear();
		if( direction == D_ByRows ) {
			MathEngine().AddVectorToMatrixRows( inputDesc.BatchWidth() * inputDesc.Height(), outputBlobs[0]->GetData(),
				outputBlobs[0]->GetData(), inputDesc.Width(), inputDesc.Channels(), projectionResultBlob->GetData() );
		} else {
			MathEngine().AddVectorToMatrixRows( inputDesc.BatchWidth(), outputBlobs[0]->GetData(), outputBlobs[0]->GetData(),
				inputDesc.Height(), inputDesc.Width() * inputDesc.Channels(),  projectionResultBlob->GetData() );
		}
	} else {
		// Calculate pooling result straight into the output blob
		MathEngine().BlobMeanPooling( *desc, inputBlobs[0]->GetData(), outputBlobs[0]->GetData() );
	}
}

void CProjectionPoolingLayer::BackwardOnce()
{
	const CBlobDesc& outputDesc = outputDiffBlobs[0]->GetDesc();
	const CBlobDesc& inputDesc = inputDiffBlobs[0]->GetDesc();

	if( shouldRestoreOriginalImageSize ) {
		NeoAssert( projectionResultBlob != nullptr );
		// Sum output diff's into the temporary blob
		if( direction == D_ByRows ) {
			MathEngine().SumMatrixRows( outputDesc.BatchWidth() * outputDesc.Height(),
				projectionResultBlob->GetData(), outputDiffBlobs[0]->GetData(), outputDesc.Width(), outputDesc.Channels() );
		} else {
			MathEngine().SumMatrixRows( outputDesc.BatchWidth(), projectionResultBlob->GetData(),
				outputDiffBlobs[0]->GetData(), outputDesc.Height(), outputDesc.Width() * outputDesc.Channels() );
		}
		// Calculate backprop of pooling
		MathEngine().BlobMeanPoolingBackward( *desc, projectionResultBlob->GetData(), inputDiffBlobs[0]->GetData() );
	} else {
		// Calculate backprop of pooling
		MathEngine().BlobMeanPoolingBackward( *desc, outputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetData() );
	}
}

void CProjectionPoolingLayer::initDesc()
{
	if( desc == nullptr ) {
		CDnnBlob* resultBlob = shouldRestoreOriginalImageSize ? projectionResultBlob : outputBlobs[0];

		if( direction == D_ByRows ) {
			desc = MathEngine().InitMeanPooling( inputBlobs[0]->GetDesc(), 1, inputBlobs[0]->GetWidth(),
				1, inputBlobs[0]->GetWidth(), resultBlob->GetDesc() );
		} else {
			desc = MathEngine().InitMeanPooling( inputBlobs[0]->GetDesc(), inputBlobs[0]->GetHeight(), 1,
				inputBlobs[0]->GetHeight(), 1, resultBlob->GetDesc() );
		}
	}
}

void CProjectionPoolingLayer::destroyDesc()
{
	if( desc != nullptr ) {
		delete desc;
		desc = nullptr;
	}
}

} // namespace NeoML
