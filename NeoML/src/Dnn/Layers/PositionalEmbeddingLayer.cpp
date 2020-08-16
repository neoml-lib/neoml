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

#include <NeoML/Dnn/Layers/PositionalEmbeddingLayer.h>

namespace NeoML {

CPositionalEmbeddingLayer::CPositionalEmbeddingLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CPositionalEmbeddingLayer", true ),
	type( PET_LearnableAddition )
{
}

void CPositionalEmbeddingLayer::Reshape()
{
	checkDimensions();

	const CBlobDesc& inputDesc = inputDescs[0];
	CBlobDesc paramsDesc = inputDesc;
	paramsDesc.SetDimSize( BD_BatchWidth, 1 );

	if( paramBlobs.IsEmpty() || !paramBlobs.First()->GetDesc().HasEqualDimensions( paramsDesc ) ) {
		switch( type ) {
			case PET_LearnableAddition:
			{
				paramBlobs.SetSize( 1 );
				paramBlobs[0] = CDnnBlob::CreateBlob( MathEngine(), paramsDesc );

				initializeLearnableAddition();
				break;
			}
			case PET_LearnableMultAddition:
			{
				paramBlobs.SetSize( 2 );
				paramBlobs[0] = CDnnBlob::CreateBlob( MathEngine(), paramsDesc );
				initializeLearnableAddition();

				paramBlobs[1] = CDnnBlob::CreateBlob( MathEngine(), paramsDesc );
				paramBlobs[1]->Fill( 1 );
				break;
			}
			case PET_Transformers:
			{
				paramBlobs.SetSize( 1 );
				paramBlobs[0] = CDnnBlob::CreateBlob( MathEngine(), paramsDesc );

				fillPositionalEmbedding( paramBlobs[0] );
				break;
			}
			default:
				break;

		}
	}

	outputDescs.SetSize( 1 );
	outputDescs[0] = inputDesc;
}

void CPositionalEmbeddingLayer::RunOnce()
{
	const int objectsCount = inputBlobs[0]->GetBatchWidth();
	const int objectSize = inputBlobs[0]->GetDataSize() / objectsCount;

	switch( type ) {
		case PET_LearnableAddition:
		case PET_Transformers:
			MathEngine().AddVectorToMatrixRows( 1, inputBlobs[0]->GetData(), outputBlobs[0]->GetData(),
				objectsCount, objectSize, paramBlobs[0]->GetData() );
			break;
		case PET_LearnableMultAddition:
			for( int i = 0; i < objectsCount; i++ ) {
				MathEngine().VectorEltwiseMultiply( inputBlobs[0]->GetObjectData( i ),
					paramBlobs[1]->GetData(), outputBlobs[0]->GetObjectData( i ), objectSize );
			}
			MathEngine().AddVectorToMatrixRows( 1, outputBlobs[0]->GetData(), outputBlobs[0]->GetData(),
				objectsCount, objectSize, paramBlobs[0]->GetData() );
			break;
		default:
			NeoAssert( false );
	}
}

void CPositionalEmbeddingLayer::BackwardOnce()
{
	const int objectsCount = inputBlobs[0]->GetBatchWidth();
	const int objectSize = inputBlobs[0]->GetDataSize() / objectsCount;

	switch( type ) {
		case PET_Transformers:
		case PET_LearnableAddition:
			MathEngine().VectorCopy( inputDiffBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
				objectsCount * objectSize );
			break;
		case PET_LearnableMultAddition:
			MathEngine().VectorCopy( inputDiffBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
				objectsCount * objectSize );
			for( int i = 0; i < objectsCount; i++ ) {
				MathEngine().VectorEltwiseMultiply( outputDiffBlobs[0]->GetObjectData( i ),
					paramBlobs[1]->GetData(), inputDiffBlobs[0]->GetObjectData( i ), objectSize );
			}
			break;
		default:
			NeoAssert( false );
	}
}

void CPositionalEmbeddingLayer::LearnOnce()
{
	const int objectsCount = inputBlobs[0]->GetBatchWidth();
	const int objectSize = inputBlobs[0]->GetDataSize() / objectsCount;

	static_assert( PET_EnumCount == 3, "PET_EnumCount != 3" );
	switch( type ) {
		case PET_LearnableAddition:
			for( int i = 0; i < objectsCount; i++ ) {
				MathEngine().VectorAdd( outputDiffBlobs[0]->GetObjectData( i ),
					paramDiffBlobs[0]->GetData(), paramDiffBlobs[0]->GetData(), objectSize );
			}
			break;
		case PET_LearnableMultAddition:
			for( int i = 0; i < objectsCount; i++ ) {
				MathEngine().VectorEltwiseMultiplyAdd( outputDiffBlobs[0]->GetObjectData( i ),
					inputBlobs[0]->GetObjectData( i ), paramDiffBlobs[1]->GetData(), objectSize );
				MathEngine().VectorAdd( outputDiffBlobs[0]->GetObjectData( i ),
					paramDiffBlobs[0]->GetData(), paramDiffBlobs[0]->GetData(), objectSize );
			}
			break;
		case PET_Transformers:
			break;
		default:
			NeoAssert( false );
	}
}

static const int PositionalEmbeddingLayerVersion = 0;

void CPositionalEmbeddingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( PositionalEmbeddingLayerVersion );
	CBaseLayer::Serialize( archive );
	archive.SerializeEnum( type );
}

// Checks input dimensions
void CPositionalEmbeddingLayer::checkDimensions()
{
	CheckInputs();
	NeoAssert( inputDescs.Size() == 1 );

	const CBlobDesc& inputDesc = inputDescs[0];

	CheckArchitecture( inputDesc.BatchLength() == 1, GetName(), "CPositionalEmbeddingLayer wrong input BatchLength dimension" );

	if( type == PET_Transformers ) {
		CheckArchitecture( inputDesc.Height() == 1, GetName(), "CPositionalEmbeddingLayer wrong input Height dimension" );
		CheckArchitecture( inputDesc.Width() == 1, GetName(), "CPositionalEmbeddingLayer wrong input Width dimension" );
		CheckArchitecture( inputDesc.Depth() == 1, GetName(), "CPositionalEmbeddingLayer wrong input Depth dimension" );
	}
}

// Initializes learnable addition
void CPositionalEmbeddingLayer::initializeLearnableAddition()
{
	NeoAssert( paramBlobs.Size() >= 1 );
	// This initialization is used in BERT
	CPtr<CDnnUniformInitializer> uniformInitializer = new CDnnUniformInitializer( GetDnn()->Random(), -0.02f, 0.02f );
	uniformInitializer->InitializeLayerParams( *paramBlobs[0], 0 );
}

// Fills positional embeddings (used in case of PET_Transformers)
// https://arxiv.org/abs/1807.03819
void CPositionalEmbeddingLayer::fillPositionalEmbedding( CDnnBlob* blob )
{
	NeoAssert( blob != 0 );

	const int seqLength = blob->GetListSize();
	const int hiddenSize = blob->GetChannelsCount();
	NeoAssert( blob->GetDataSize() == seqLength * hiddenSize );

	CArray<float> data;
	data.SetBufferSize( blob->GetDataSize() );

	for ( int pos = 0; pos < seqLength; pos++ )	{
		for( int i = 0; i < hiddenSize; i++ ) {
			double value = 0;
			if( i % 2 == 0 ) {
				value = sin( 1.0 * pos / pow( 10000.0, static_cast<double>( i ) / hiddenSize ) );
			} else {
				value = cos( 1.0 * pos / pow( 10000.0, static_cast<double>( i - 1.0 ) / hiddenSize ) );
			}
			data.Add( static_cast<float>( value ) );
		}
	}
	blob->CopyFrom( data.GetPtr() );
}

} // namespace NeoML
