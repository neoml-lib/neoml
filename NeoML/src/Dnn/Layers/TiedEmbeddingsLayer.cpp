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

#include <NeoML/Dnn/Layers/TiedEmbeddingsLayer.h>
#include <NeoML/Dnn/Layers/MultichannelLookupLayer.h>

namespace NeoML {

void CTiedEmbeddingsLayer::SetChannelIndex( int val )
{
	NeoAssert( val >= 0 );

	channelIndex = val;
}

constexpr int TiedEmbeddingsLayerVersion = 2001;

void CTiedEmbeddingsLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( TiedEmbeddingsLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	if (version < 2001 && archive.IsLoading()) {
		CString embeddingLayerName;
		archive.Serialize(embeddingLayerName);
		embeddingPath = { embeddingLayerName };
	}
	else {
		archive.Serialize(embeddingPath);
	}

	archive.Serialize( channelIndex );
}

void CTiedEmbeddingsLayer::Reshape()
{
	CheckInputs();
	const CMultichannelLookupLayer* embeddingsLayer = getLookUpLayer();

	CheckLayerArchitecture( embeddingsLayer != 0, "The layer is not an embedding layer." );

	const int embeddingsChannelsCount = embeddingsLayer->GetDimensions().Size();
	CheckLayerArchitecture( channelIndex < embeddingsChannelsCount,
		"Wrong channgel index for embeddings" );

	outputDescs.SetSize( inputDescs.Size() );

	const CDnnBlob* embeddingsTable = getEmbeddingsTable();
	const int vectorsCount = embeddingsTable->GetBatchWidth();
	const int vectorSize = embeddingsTable->GetChannelsCount();

	for( int i = 0; i < inputDescs.Size(); i++ ) {
		const CBlobDesc inputDesc = inputDescs[i];
		CheckLayerArchitecture( inputDesc.Channels() == vectorSize,
			"The number of channels in the input layer is incorrect." );
		CheckLayerArchitecture( inputDesc.Width() == 1, "The width in the input layer must be 1." );
		CheckLayerArchitecture( inputDesc.Height() == 1, "The height in the input layer must be 1." );
		CheckLayerArchitecture( inputDesc.Depth() == 1, "The depth in the input layer must be 1." );

		CBlobDesc outputDesc = inputDesc;
		outputDesc.SetDimSize( BD_Channels, vectorsCount );
		outputDescs[i] = outputDesc;
	}
}

void CTiedEmbeddingsLayer::RunOnce()
{
	const CDnnBlob* embeddingsTable = getEmbeddingsTable();
	const int vectorsCount = embeddingsTable->GetBatchWidth();
	const int vectorSize = embeddingsTable->GetChannelsCount();

	for( int i = 0; i < inputBlobs.Size(); i++ ) {
		MathEngine().MultiplyMatrixByTransposedMatrix( 1, inputBlobs[i]->GetData(),
			inputBlobs[i]->GetObjectCount(), vectorSize, embeddingsTable->GetData(),
			vectorsCount, outputBlobs[i]->GetData(), outputBlobs[i]->GetDataSize() );
	}
}

void CTiedEmbeddingsLayer::BackwardOnce()
{
	const CDnnBlob* embeddingsTable = getEmbeddingsTable();
	const int vectorsCount = embeddingsTable->GetBatchWidth();
	const int vectorSize = embeddingsTable->GetChannelsCount();

	for( int i = 0; i < inputBlobs.Size(); i++ ) {
		MathEngine().MultiplyMatrixByMatrix( 1, outputDiffBlobs[i]->GetData(),
			outputDiffBlobs[i]->GetObjectCount(), vectorsCount, embeddingsTable->GetData(),
			vectorSize, inputDiffBlobs[i]->GetData(), inputDiffBlobs[i]->GetDataSize() );
	}
}

void CTiedEmbeddingsLayer::LearnOnce()
{
	const CDnnBlob* embeddingsTable = getEmbeddingsTable();
	const int vectorsCount = embeddingsTable->GetBatchWidth();
	const int vectorSize = embeddingsTable->GetChannelsCount();

	CBlobDesc diffBlobDesc = embeddingsTable->GetDesc();
	CPtr<CDnnBlob> totalDiffBlob = CDnnBlob::CreateBlob( MathEngine(), diffBlobDesc );
	totalDiffBlob->Fill( 0 );
	CPtr<CDnnBlob> diffBlob = CDnnBlob::CreateBlob( MathEngine(), diffBlobDesc );
	
	for( int i = 0; i < inputBlobs.Size(); i++ ) {
		MathEngine().MultiplyTransposedMatrixByMatrix( 1, outputDiffBlobs[i]->GetData(),
			outputDiffBlobs[i]->GetObjectCount(), vectorsCount, inputBlobs[i]->GetData(),
			vectorSize, diffBlob->GetData(), diffBlob->GetDataSize() );

		MathEngine().VectorAdd( totalDiffBlob->GetData(), diffBlob->GetData(),
			totalDiffBlob->GetData(), totalDiffBlob->GetDataSize() );
		diffBlob->Clear();
	}

	const CMultichannelLookupLayer* embeddingsLayer = getLookUpLayer();
	CObjectArray<CDnnBlob> totalDiffBlobs;
	const int channelsCount = embeddingsLayer->GetDimensions().Size();
	for( int i = 0; i < channelsCount; i++ ) {
		if( i == channelIndex ) {
			totalDiffBlobs.Add( totalDiffBlob );
		} else {
			CPtr<CDnnBlob> nullBlob = embeddingsLayer->GetEmbeddings( i )->GetClone();
			nullBlob->Fill( 0 );
			totalDiffBlobs.Add( nullBlob );
		}
	}

	GetDnn()->GetSolver()->AddDiff( embeddingsLayer, totalDiffBlobs, /*sharedWeights*/true );
}

// Embeddings matrix
const CDnnBlob* CTiedEmbeddingsLayer::getEmbeddingsTable() const
{
	NeoAssert( channelIndex >= 0 );

	return getLookUpLayer()->GetEmbeddings( channelIndex );
}

const CMultichannelLookupLayer* CTiedEmbeddingsLayer::getLookUpLayer() const
{
	const CMultichannelLookupLayer* embeddingsLayer
		= CheckCast<CMultichannelLookupLayer>( GetDnn()->GetLayer( embeddingPath ).Ptr() );
	return embeddingsLayer;
}

CLayerWrapper<CTiedEmbeddingsLayer> TiedEmbeddings( const char* name, int channel, CArray<CString>&& embeddingPath )
{
	return CLayerWrapper<CTiedEmbeddingsLayer>( "TiedEmbeddings",
		[=, path=std::move( embeddingPath )]( CTiedEmbeddingsLayer* result )
		{
			result->SetEmbeddingsLayerName( name );
			result->SetChannelIndex( channel );
			result->SetEmbeddingsLayerPath( path );
		}
	);
}

} // namespace NeoML
