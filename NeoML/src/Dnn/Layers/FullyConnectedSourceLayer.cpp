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

#include <NeoML/Dnn/Layers/FullyConnectedSourceLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CFullyConnectedSourceLayer::CFullyConnectedSourceLayer( IMathEngine& mathEngine ) :
	CFullyConnectedLayer( mathEngine, "CCnnFullyConnectedSourceLayer" ),
	problem( nullptr ),
	batchData( 0 ),
	batchSize( 1 ),
	batchCount( 0 ),
	batchIndex( NotFound ),
	batchFirstLoadedIndex( NotFound ),
	batchLastLoadedIndex( NotFound ),
	firstVectorInBatchIndex( NotFound ),
	labelType( CT_Float )
{
}

void CFullyConnectedSourceLayer::SetBatchSize( int newBatchSize )
{
	NeoAssert( newBatchSize > 0 );

	batchSize = newBatchSize;
	batchIndex = NotFound;
	batchFirstLoadedIndex = NotFound;
	batchLastLoadedIndex = NotFound;
	if( batchData != 0 ) {
		delete batchData;
		batchData = 0;
	}

	ForceReshape();
}

void CFullyConnectedSourceLayer::SetMaxBatchCount( int newBatchCount )
{
	NeoAssert( newBatchCount >= 0 );

	batchCount = newBatchCount;
}

void CFullyConnectedSourceLayer::SetProblem( IProblem* newProblem )
{
	NeoAssert( GetDnn() == 0 || problem == 0 || newProblem == 0
		|| ( problem->GetFeatureCount() == newProblem->GetFeatureCount()
			&& problem->GetClassCount() == newProblem->GetClassCount() ) );

	problem = newProblem;
	batchIndex = NotFound;
	batchFirstLoadedIndex = NotFound;
	batchLastLoadedIndex = NotFound;
	if( batchData != 0 ) {
		delete batchData;
		batchData = 0;
	}
	firstVectorInBatchIndex = 0;
}

void CFullyConnectedSourceLayer::SetLabelType( TBlobType newLabelType )
{
	NeoAssert( newLabelType == CT_Float || newLabelType == CT_Int );

	if( labelType == newLabelType ) {
		return;
	}

	labelType = newLabelType;

	ForceReshape();
}

static const int FullyConnectedSourceLayerVersion = 2000;

void CFullyConnectedSourceLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( FullyConnectedSourceLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CFullyConnectedLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		archive << batchSize;
		archive << batchCount;
		archive << static_cast<int>( labelType );
	} else if( archive.IsLoading() ) {
		problem = 0;
		delete batchData;
		batchData = 0;
		archive >> batchSize;
		archive >> batchCount;
		batchIndex = NotFound;
		batchFirstLoadedIndex = NotFound;
		batchLastLoadedIndex = NotFound;
		firstVectorInBatchIndex = 0;
		int labelTypeInt = 0;
		archive >> labelTypeInt;
		labelType = static_cast<TBlobType>( labelTypeInt );
	} else {
		NeoAssert( false );
	}
}

CFullyConnectedSourceLayer::~CFullyConnectedSourceLayer()
{
	if( batchData != 0 ) {
		delete batchData;
	}
}

void CFullyConnectedSourceLayer::Reshape()
{
	CheckArchitecture( inputDescs.IsEmpty(), GetName(), "layer has input" );
	CheckArchitecture( GetOutputCount() >= 3, GetName(), "fully connected source layer has less than 3 outputs" );
	CheckArchitecture( problem.Ptr() != 0, GetName(), "source problem is null" );

	if( Weights() == 0 ) {
		// Create and initialize a weights matrix
		Weights() = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, GetNumberOfElements(), problem->GetFeatureCount() );
		InitializeParamBlob( 0, *Weights(), batchSize );
	} else {
		CheckArchitecture( Weights()->GetObjectCount() == GetNumberOfElements(),
			GetName(), "weights number is not equal to number of elements" );
		CheckArchitecture( Weights()->GetObjectSize() == problem->GetFeatureCount(),
			GetName(), "weights size mismatch" );
	}

	if( FreeTerms() == 0 ) {
		FreeTerms() = CDnnBlob::CreateVector( MathEngine(), CT_Float, GetNumberOfElements() );
		// Initialize
		FreeTerms()->Fill( 0 );
	} else {
		CheckArchitecture( FreeTerms()->GetDataSize() == GetNumberOfElements(),
			GetName(), "free terms num is not equal to number of elements" );
	}

	// The data
	outputDescs[0] = CBlobDesc( CT_Float );
	outputDescs[0].SetDimSize( BD_BatchWidth, batchSize );
	outputDescs[0].SetDimSize( BD_Channels, GetNumberOfElements() );

	// The labels
	int labelSize = problem->GetClassCount();
	if( labelSize == 2 ) {
		labelSize = 1;
	}
	outputDescs[1] = CBlobDesc( labelType );
	outputDescs[1].SetDimSize( BD_BatchWidth, batchSize );
	if( labelType != CT_Int ) {
		outputDescs[1].SetDimSize( BD_Channels, labelSize );
	}

	// The weights
	outputDescs[2] = CBlobDesc( CT_Float );
	outputDescs[2].SetDimSize( BD_BatchWidth, batchSize );
}

void CFullyConnectedSourceLayer::RunOnce()
{
	loadBatchData();

	// Update the data
	CSparseMatrixDesc batchDataDesc = batchData->GetBatchDesc( batchIndex - batchFirstLoadedIndex );
	MathEngine().MultiplySparseMatrixByTransposedMatrix( batchSize,
		problem->GetFeatureCount(), GetNumberOfElements(),
		batchDataDesc, Weights()->GetData(), outputBlobs[0]->GetData() );

	if( !IsZeroFreeTerm() ) {
		MathEngine().AddVectorToMatrixRows(1, outputBlobs[0]->GetObjectData( 0 ), outputBlobs[0]->GetData(),
			batchSize, outputBlobs[0]->GetObjectSize(), FreeTerms()->GetData());
	}

	// Update the labels and weights

	float* labels = batchLabels.GetPtr();
	float* weights = batchWeights.GetPtr();

	const int vectorCount = problem->GetVectorCount();
	for( int i = 0; i < batchSize; ++i ) {
		const int vectorIndex = ( firstVectorInBatchIndex + i ) % vectorCount;
		// Update the labels
		if( labelType == CT_Float ) {
			if( outputBlobs[1]->GetChannelsCount() == 1 ) {
				*labels = static_cast< float >( problem->GetBinaryClass( vectorIndex ) );
			} else {
				int classLabel = problem->GetClass( vectorIndex );
				NeoAssert( 0 <= classLabel && classLabel < outputBlobs[1]->GetChannelsCount() );
				::memset( labels, 0, outputBlobs[1]->GetChannelsCount() * sizeof( float ) );
				labels[classLabel] = 1;
			}
		} else {
			static_assert( sizeof( float ) == sizeof( int ), "sizeof( float ) != sizeof( int )" );
			NeoAssert( outputBlobs[1]->GetChannelsCount() == 1 );
			*reinterpret_cast<int*>( labels ) = problem->GetClass( vectorIndex );
		}

		// Update the weights
		*weights = static_cast<float>( problem->GetVectorWeight( vectorIndex ) );

		labels += outputBlobs[1]->GetObjectSize();
		weights += outputBlobs[2]->GetObjectSize();
	}

	if( labelType == CT_Float ) {
		outputBlobs[1]->CopyFrom( batchLabels.GetPtr() );
	} else {
		outputBlobs[1]->CopyFrom( reinterpret_cast<int*>( batchLabels.GetPtr() ) );
	}
	outputBlobs[2]->CopyFrom( batchWeights.GetPtr() );
}

void CFullyConnectedSourceLayer::BackwardOnce()
{
	NeoAssert( false );
}

void CFullyConnectedSourceLayer::LearnOnce()
{
	NeoAssert( batchData != 0 );

	CSparseMatrixDesc batchDataDesc = batchData->GetBatchDesc( batchIndex );
	MathEngine().MultiplyTransposedMatrixBySparseMatrixAndAdd( outputDiffBlobs[0]->GetObjectCount(),
		GetNumberOfElements(), problem->GetFeatureCount(),
		outputDiffBlobs[0]->GetData(), batchDataDesc, WeightsDiff()->GetData() );

	if( !IsZeroFreeTerm() ) {
		MathEngine().SumMatrixRowsAdd( 1, FreeTermsDiff()->GetData(),
			outputDiffBlobs[0]->GetData(), outputDiffBlobs[0]->GetObjectCount(), GetNumberOfElements() );
	}
}

// Load the batch
void CFullyConnectedSourceLayer::loadBatchData()
{
	NeoAssert( problem != 0 );

	const int totalBatchCount = Ceil( problem->GetVectorCount(), batchSize );

	// Initialize the data
	if( batchData == 0 ) {
		NeoAssert( batchIndex == NotFound );
		NeoAssert( batchFirstLoadedIndex == NotFound );
		NeoAssert( batchLastLoadedIndex == NotFound );
		batchData = FINE_DEBUG_NEW CDnnSparseMatrix( MathEngine(), batchSize, problem->GetFeatureCount() );
		batchLabels.SetSize( outputBlobs[1]->GetDataSize() );
		batchWeights.SetSize( outputBlobs[2]->GetDataSize() );
		firstVectorInBatchIndex = 0;
	}

	bool requiresReload = false;

	// Increment the current batch index
	if( batchIndex == NotFound ) {
		batchIndex = 0;
		firstVectorInBatchIndex = 0;
	} else {
		batchIndex++;
		firstVectorInBatchIndex += batchSize;
		firstVectorInBatchIndex %= problem->GetVectorCount();
		if( batchIndex == totalBatchCount ) {
			batchIndex = 0;
			if( firstVectorInBatchIndex != 0 ) {
				// The data size is not divisible by the batch size
				// So the data should be reloaded with offset
				requiresReload = true;
			}
		}
	}

	// Load the current batch
	if( !isBatchLoaded( batchIndex ) || requiresReload ) {
		batchData->Destroy();
		batchFirstLoadedIndex = NotFound;
		batchLastLoadedIndex = NotFound;
		if( batchCount == 0 ) {
			batchData->Create( problem, firstVectorInBatchIndex, totalBatchCount );
			batchFirstLoadedIndex = 0;
			batchLastLoadedIndex = totalBatchCount - 1;
		} else {
			batchData->Create( problem, firstVectorInBatchIndex, min( batchCount, totalBatchCount - batchIndex ) );
			batchFirstLoadedIndex = batchIndex;
			batchLastLoadedIndex = batchIndex + min( batchCount, totalBatchCount - batchIndex ) - 1;
		}
	}
}

// Checks if a batch with this index has been loaded
bool CFullyConnectedSourceLayer::isBatchLoaded( int index ) const
{
	if( batchFirstLoadedIndex == NotFound || batchLastLoadedIndex == NotFound ) {
		return false;
	}
	return ( batchFirstLoadedIndex <= index && index <= batchLastLoadedIndex );
}

} // namespace NeoML
