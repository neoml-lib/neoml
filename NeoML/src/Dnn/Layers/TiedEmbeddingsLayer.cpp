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

#include <NeoML/Dnn/Layers/TiedEmbeddingsLayer.h>
#include <NeoML/Dnn/Layers/MultichannelLookupLayer.h>

namespace NeoML {

////////////////////////////////////////////////////////////////////////////////////////////////////

CTiedEmbeddingsLayer::CTiedEmbeddingsLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CTiedEmbeddingsLayer", true ),
	channelIndex( 0 )
{
}

void CTiedEmbeddingsLayer::SetChannelIndex( int val )
{
	NeoAssert( val >= 0 );

	channelIndex = val;
}

static const int CnnTiedEmbeddingsLayerVersion = 2000;

void CTiedEmbeddingsLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CnnTiedEmbeddingsLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( embeddingsLayerName );
	archive.Serialize( channelIndex );
}

void CTiedEmbeddingsLayer::Reshape()
{
	CheckInputs();

	CheckArchitecture( GetDnn()->HasLayer( embeddingsLayerName ), GetName(),
		"Wrong name for embeddings layer." );

	const int embeddingsChannelsCount = CheckCast<CMultichannelLookupLayer>(
		GetDnn()->GetLayer( embeddingsLayerName ) )->GetDimensions().Size();
	CheckArchitecture( channelIndex < embeddingsChannelsCount, GetName(),
		"Wrong channgel index for embeddings" );

	outputDescs.SetSize( inputDescs.Size() );

	const CDnnBlob* embeddingsTable = getEmbeddingsTable();
	const int vectorsCount = embeddingsTable->GetBatchWidth();
	const int vectorSize = embeddingsTable->GetChannelsCount();

	for( int i = 0; i < inputDescs.Size(); i++ ) {
		const CBlobDesc inputDesc = inputDescs[i];
		assert( inputDesc.Channels() == vectorSize );

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

		MathEngine().VectorAdd( totalDiffBlob->GetData(), diffBlob->GetData(), totalDiffBlob->GetData(),
			totalDiffBlob->GetDataSize() );
		diffBlob->Clear();
	}

	CObjectArray<CDnnBlob> totalDiffBlobs;
	totalDiffBlobs.Add( totalDiffBlob );

	CMultichannelLookupLayer* embeddingsLayer =
		CheckCast<CMultichannelLookupLayer>( GetDnn()->GetLayer( embeddingsLayerName ) );
	GetDnn()->GetSolver()->AddDiff( embeddingsLayer, totalDiffBlobs );
}

// Embeddings matrix
const CDnnBlob* CTiedEmbeddingsLayer::getEmbeddingsTable() const
{
	NeoAssert( channelIndex >= 0 );

	const CMultichannelLookupLayer* embeddingsLayer =
		CheckCast<CMultichannelLookupLayer>( GetDnn()->GetLayer( embeddingsLayerName ) );
	return embeddingsLayer->GetEmbeddings( channelIndex );
}

CLayerWrapper<CTiedEmbeddingsLayer> TiedEmbeddings( const char* name, int channel )
{
	return CLayerWrapper<CTiedEmbeddingsLayer>( "TiedEmbeddings", [=]( CTiedEmbeddingsLayer* result ) {
		result->SetEmbeddingsLayerName( name );
		result->SetChannelIndex( channel );
	} );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace NeoML
