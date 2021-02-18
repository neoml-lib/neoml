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
	dimension( BD_Width ),
	restoreOriginalImageSize( false ),
	desc( nullptr )
{
}

CProjectionPoolingLayer::~CProjectionPoolingLayer()
{
	destroyDesc();
}

void CProjectionPoolingLayer::SetDimension( TBlobDim _dimension )
{
	if( dimension == _dimension ) {
		return;
	}

	dimension = _dimension;
	ForceReshape();
}

void CProjectionPoolingLayer::SetRestoreOriginalImageSize( bool flag )
{
	if( restoreOriginalImageSize == flag ) {
		return;
	}

	restoreOriginalImageSize = flag;
	ForceReshape();
}

static const int currentVersion = 1;

void CProjectionPoolingLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( currentVersion );
	CBaseLayer::Serialize( archive );

	if( version < 1 ) {
		// (legacy) Projection direction
		enum TDirection {
			D_ByRows, // Along BD_Width
			D_ByColumns, // Along BD_Height
			D_EnumSize
		};

		TDirection direction = D_ByRows;
		archive.SerializeEnum( direction );

		switch( direction ) {
			case D_ByRows:
				dimension = BD_Width;
				break;
			case D_ByColumns:
				dimension = BD_Height;
				break;
			default:
				NeoAssert( false );
		}
	} else {
		int intDimension = static_cast<int>( dimension );
		archive.Serialize( intDimension );
		dimension = static_cast<TBlobDim>( intDimension );
	}
	archive.Serialize( restoreOriginalImageSize );
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
	if( restoreOriginalImageSize ) {
		CBlobDesc projectionResultBlobDesc = inputDescs[0];
		projectionResultBlobDesc.SetDimSize( dimension, 1 );
		projectionResultBlob = CDnnBlob::CreateBlob( MathEngine(), projectionResultBlobDesc );

		RegisterRuntimeBlob( projectionResultBlob );
	} else {
		outputDescs[0].SetDimSize( dimension, 1 );
	}

	destroyDesc();
}

void CProjectionPoolingLayer::RunOnce()
{
	const CBlobDesc& inputDesc = inputBlobs[0]->GetDesc();
	initDesc( inputDesc );

	if( restoreOriginalImageSize ) {
		NeoAssert( projectionResultBlob != nullptr );
		// Calculate pooling result into the temporary blob
		MathEngine().BlobMeanPooling( *desc, inputBlobs[0]->GetData(), projectionResultBlob->GetData() );
		// Broadcatst pooling result along whole result blob
		outputBlobs[0]->Clear();

		int batchSize = 1;
		int matrixHeight = 1;
		int matrixWidth = 1;
		for( TBlobDim d = TBlobDim( 0 ); d < BD_Count; ++d ) {
			if( d < dimension ) {
				batchSize *= inputDesc.DimSize( d );
			} else if( d == dimension ) {
				matrixHeight *= inputDesc.DimSize( d );
			} else {
				matrixWidth *= inputDesc.DimSize( d );
			}
		}

		MathEngine().AddVectorToMatrixRows( batchSize, outputBlobs[0]->GetData(), outputBlobs[0]->GetData(),
			matrixHeight, matrixWidth, projectionResultBlob->GetData() );
	} else {
		// Calculate pooling result straight into the output blob
		MathEngine().BlobMeanPooling( *desc, inputBlobs[0]->GetData(), outputBlobs[0]->GetData() );
	}
}

void CProjectionPoolingLayer::BackwardOnce()
{
	const CBlobDesc& outputDesc = outputDiffBlobs[0]->GetDesc();

	if( restoreOriginalImageSize ) {
		NeoAssert( projectionResultBlob != nullptr );
		// Sum output diff's into the temporary blob
		int batchSize = 1;
		int matrixHeight = 1;
		int matrixWidth = 1;
		for( TBlobDim d = TBlobDim( 0 ); d < BD_Count; ++d ) {
			if( d < dimension ) {
				batchSize *= outputDesc.DimSize( d );
			} else if( d == dimension ) {
				matrixHeight *= outputDesc.DimSize( d );
			} else {
				matrixWidth *= outputDesc.DimSize( d );
			}
		}
		MathEngine().SumMatrixRows( batchSize, projectionResultBlob->GetData(),
			outputDiffBlobs[0]->GetData(), matrixHeight, matrixWidth );

		// Calculate backprop of pooling
		MathEngine().BlobMeanPoolingBackward( *desc, projectionResultBlob->GetData(), inputDiffBlobs[0]->GetData() );
	} else {
		// Calculate backprop of pooling
		MathEngine().BlobMeanPoolingBackward( *desc, outputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetData() );
	}
}

void CProjectionPoolingLayer::initDesc( const CBlobDesc& inputDesc )
{
	if( desc == nullptr ) {
		// Emulating mean pooling along given dimension by 2dPooling with projection dimension as height
		// Every dimension before projection dimension is interpreted as batchWidth
		// Every dimension after projection dimension is interpreted as channels
		int poolBatchSize = 1;
		int poolHeight = 1;
		int poolChannels = 1;

		for( TBlobDim d = TBlobDim( 0 ); d < BD_Count; ++d ) {
			if( d < dimension ) {
				poolBatchSize *= inputDesc.DimSize( d );
			} else if( d == dimension ) {
				poolHeight *= inputDesc.DimSize( d );
			} else {
				poolChannels *= inputDesc.DimSize( d );
			}
		}

		CBlobDesc poolOutputDesc( CT_Float );
		poolOutputDesc.SetDimSize( BD_BatchWidth, poolBatchSize );
		poolOutputDesc.SetDimSize( BD_Channels, poolChannels );
		CBlobDesc poolInputDesc( poolOutputDesc );
		poolInputDesc.SetDimSize( BD_Height, poolHeight );

		desc = MathEngine().InitMeanPooling( poolInputDesc, poolHeight, 1, poolHeight, 1, poolOutputDesc );
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
