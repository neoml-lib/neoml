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

#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/ConcatLayer.h>

namespace NeoML {

CBaseConcatLayer::CBaseConcatLayer( IMathEngine& mathEngine, TBlobDim _dimension, const char* name ) :
	CBaseLayer( mathEngine, name, false ),
	dimension( _dimension )
{
}

void CBaseConcatLayer::Reshape()
{
	CheckInputs();

	// Calculate the output blob size
	int outputDimSize = 0;
	for(int i = 0; i < inputDescs.Size(); ++i) {
		outputDimSize += inputDescs[i].DimSize(dimension);
	}

	// Create the output blob
	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize(dimension, outputDimSize);

	// Check compatibility
	for( int i = 1; i < inputDescs.Size(); ++i ) {
		CBlobDesc pattern1 = inputDescs[i];
		pattern1.SetDimSize(dimension, outputDimSize);
		CheckArchitecture( outputDescs[0].HasEqualDimensions(pattern1), GetName(), "Incompatible blobs size" );
	}
}

void CBaseConcatLayer::RunOnce()
{
	CDnnBlob::MergeByDim( MathEngine(), dimension, inputBlobs, outputBlobs[0] );
}

void CBaseConcatLayer::BackwardOnce()
{
	CDnnBlob::SplitByDim( MathEngine(), dimension, outputDiffBlobs[0], inputDiffBlobs );
}

static const int BaseConcatLayerVersion = 2000;

void CBaseConcatLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( BaseConcatLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

// ====================================================================================================================

static const int ConcatChannelsLayerVersion = 2000;

void CConcatChannelsLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ConcatChannelsLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseConcatLayer::Serialize( archive );
}

CLayerWrapper<CConcatChannelsLayer> ConcatChannels()
{
	return CLayerWrapper<CConcatChannelsLayer>( "ConcatChannels" );
}

// ====================================================================================================================

static const int ConcatDepthLayerVersion = 2000;

void CConcatDepthLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ConcatDepthLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseConcatLayer::Serialize( archive );
}

CLayerWrapper<CConcatDepthLayer> ConcatDepth()
{
	return CLayerWrapper<CConcatDepthLayer>( "ConcatDepth" );
}


// ====================================================================================================================

static const int ConcatWidthLayerVersion = 2000;

void CConcatWidthLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ConcatWidthLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseConcatLayer::Serialize( archive );
}

CLayerWrapper<CConcatWidthLayer> ConcatWidth()
{
	return CLayerWrapper<CConcatWidthLayer>( "ConcatWidth" );
}

// ====================================================================================================================

static const int ConcatHeightLayerVersion = 2000;

void CConcatHeightLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ConcatHeightLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseConcatLayer::Serialize( archive );
}

CLayerWrapper<CConcatHeightLayer> ConcatHeight()
{
	return CLayerWrapper<CConcatHeightLayer>( "ConcatHeight" );
}

// ====================================================================================================================

static const int ConcatBatchWidthLayerVersion = 2000;

void CConcatBatchWidthLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ConcatBatchWidthLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseConcatLayer::Serialize( archive );
}

CLayerWrapper<CConcatBatchWidthLayer> ConcatBatchWidth()
{
	return CLayerWrapper<CConcatBatchWidthLayer>( "ConcatBatchWidth" );
}

// ====================================================================================================================

static const int ConcatObjectLayerVersion = 2000;

void CConcatObjectLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ConcatObjectLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}


CLayerWrapper<CConcatObjectLayer> ConcatObject()
{
	return CLayerWrapper<CConcatObjectLayer>( "ConcatObject" );
}

} // namespace NeoML
