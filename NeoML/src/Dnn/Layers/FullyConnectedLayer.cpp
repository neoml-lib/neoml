/* Copyright © 2017-2023 ABBYY

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

#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CFullyConnectedLayer::CFullyConnectedLayer( IMathEngine& mathEngine, const char* name ) :
	CBaseLayer( mathEngine, name == nullptr ? "CCnnFullyConnectedLayer" : name, /*isLearnable*/true ),
	numberOfElements( 0 ),
	isZeroFreeTerm( false )
{
	paramBlobs.SetSize(2);
}

void CFullyConnectedLayer::Reshape()
{
	CheckInputs();
	CheckLayerArchitecture( GetInputCount() == GetOutputCount(),
		"fully connected layer with different numbers of input and output" );
	for( int i = 0; i < GetInputCount(); ++i ) {
		if( Weights() == nullptr ) {
			// Create a weights matrix
			CBlobDesc weightsDesc = inputDescs[i];
			weightsDesc.SetDimSize( BD_BatchLength, 1 );
			weightsDesc.SetDimSize( BD_BatchWidth, numberOfElements );
			weightsDesc.SetDimSize( BD_ListSize, 1 );
			Weights() = CDnnBlob::CreateBlob( MathEngine(), CT_Float, weightsDesc );
			// Initialize
			InitializeParamBlob( i, *Weights() );
		} else {
			CheckLayerArchitecture( Weights()->GetObjectCount() == numberOfElements,
				"weights number is not equal to number of elements" );
			CheckLayerArchitecture( Weights()->GetObjectSize() == inputDescs[i].ObjectSize(),
				"weights size mismatch" );
		}

		if( FreeTerms() == nullptr ) {
			FreeTerms() = CDnnBlob::CreateVector( MathEngine(), CT_Float, numberOfElements );
			// Initialize
			FreeTerms()->Fill( 0 );
		} else {
			CheckLayerArchitecture( FreeTerms()->GetDataSize() == numberOfElements,
				"free terms num is not equal to number of elements" );
		}

		// For each layer element there is a channel in the output blob
		outputDescs[i] = inputDescs[i];
		outputDescs[i].SetDimSize( BD_Height, 1 );
		outputDescs[i].SetDimSize( BD_Width, 1 );
		outputDescs[i].SetDimSize( BD_Depth, 1 );
		outputDescs[i].SetDimSize( BD_Channels, numberOfElements );
	}
}

void CFullyConnectedLayer::RunOnce()
{
	const int inputCount = GetInputCount();
	const int secondHeight = numberOfElements;
	const int secondWidth = Weights()->GetObjectSize();

	CConstFloatHandle weightData = Weights()->GetData();
	CConstFloatHandle FreeTermsData = FreeTerms()->GetData();

	for( int inputNumber = 0; inputNumber < inputCount; ++inputNumber ) {
		CConstFloatHandle inputData = inputBlobs[inputNumber]->GetData();
		CFloatHandle outputData = outputBlobs[inputNumber]->GetData();

		const int firstHeight = inputBlobs[inputNumber]->GetObjectCount();
		const int firstWidth = inputBlobs[inputNumber]->GetObjectSize();
		const int resultWidth = outputBlobs[inputNumber]->GetObjectSize();
		NeoPresume( firstWidth == secondWidth );
		NeoPresume( resultWidth == secondHeight );

		MathEngine().MultiplyMatrixByTransposedMatrix(
			/*first*/inputData, firstHeight, firstWidth, firstWidth,
			/*second*/weightData, secondHeight, secondWidth,
			/*result*/outputData, resultWidth, /*unused*/0 );

		if( !isZeroFreeTerm ) {
			MathEngine().AddVectorToMatrixRows( /*batchSize*/1, outputData,
				outputData, firstHeight, resultWidth, FreeTermsData );
		}
	}
}

void CFullyConnectedLayer::BackwardOnce()
{
	const int outputDiffCount = outputDiffBlobs.Size();
	const int secondWidth = Weights()->GetObjectSize();

	CConstFloatHandle weightData = Weights()->GetData();

	for( int outputDiffNumber = 0; outputDiffNumber < outputDiffCount; ++outputDiffNumber ) {
		CConstFloatHandle outputDiffData = outputDiffBlobs[outputDiffNumber]->GetData();
		CFloatHandle inputDiffData = inputDiffBlobs[outputDiffNumber]->GetData();

		const int firstHeight = outputDiffBlobs[outputDiffNumber]->GetObjectCount();
		const int firstWidth = outputDiffBlobs[outputDiffNumber]->GetObjectSize();
		const int resultBufferSize = inputDiffBlobs[outputDiffNumber]->GetDataSize();

		MathEngine().MultiplyMatrixByMatrix( /*batchSize*/1,
			/*first*/outputDiffData, firstHeight, firstWidth,
			/*second*/weightData, secondWidth,
			/*result*/inputDiffData, resultBufferSize );
	}
}

void CFullyConnectedLayer::LearnOnce()
{
	const int outputDiffCount = outputDiffBlobs.Size();
	const int firstWidth = numberOfElements;
	const int resultWidth = WeightsDiff()->GetObjectSize();
	const int resultBufferSize = WeightsDiff()->GetDataSize();

	CFloatHandle weightsDiffData = WeightsDiff()->GetData();
	CFloatHandle FreeTermsDiffData = FreeTermsDiff()->GetData();

	for( int outputDiffNumber = 0; outputDiffNumber < outputDiffCount; ++outputDiffNumber ) {
		CConstFloatHandle outputDiffData = outputDiffBlobs[outputDiffNumber]->GetData();
		CConstFloatHandle inputData = inputBlobs[outputDiffNumber]->GetData();

		const int firstHeight = outputDiffBlobs[outputDiffNumber]->GetObjectCount();
		const int secondWidth = inputBlobs[outputDiffNumber]->GetObjectSize();
		NeoPresume( resultWidth == secondWidth );

		MathEngine().MultiplyTransposedMatrixByMatrixAndAdd(
			/*first*/outputDiffData, firstHeight, firstWidth, firstWidth,
			/*second*/inputData, secondWidth, secondWidth,
			/*result*/weightsDiffData, resultWidth, resultBufferSize );

		if( !isZeroFreeTerm ) {
			MathEngine().SumMatrixRowsAdd( /*batchSize*/1, FreeTermsDiffData,
				outputDiffData, firstHeight, firstWidth );
		}
	}
}

void CFullyConnectedLayer::FilterLayerParams( float threshold )
{
	for( int blobIndex = 0; blobIndex < paramBlobs.Size(); ++blobIndex ) {
		if( paramBlobs[blobIndex] != nullptr ) {
			MathEngine().FilterSmallValues( paramBlobs[blobIndex]->GetData(),
				paramBlobs[blobIndex]->GetDataSize(), threshold );
		}
	}
}

void CFullyConnectedLayer::SetNumberOfElements( int newNumberOfElements )
{
	NeoAssert( ( Weights() == nullptr && FreeTerms() == nullptr ) || numberOfElements == newNumberOfElements );
	numberOfElements = newNumberOfElements;
}

CPtr<CDnnBlob> CFullyConnectedLayer::GetWeightsData() const
{
	if( Weights() == nullptr ) {
		return nullptr;
	}
	return Weights()->GetCopy();
}

void CFullyConnectedLayer::SetWeightsData( const CDnnBlob* newWeights )
{
	if( newWeights == nullptr ) {
		NeoAssert( Weights() == nullptr || GetDnn() == nullptr );
		Weights() = nullptr;
	} else if( Weights() != nullptr && GetDnn() != nullptr ) {
		NeoAssert( Weights()->GetObjectCount() == newWeights->GetObjectCount() );
		NeoAssert( Weights()->GetObjectSize() == newWeights->GetObjectSize() );
		Weights()->CopyFrom( newWeights );
	} else {
		Weights() = newWeights->GetCopy();
	}

	if( Weights() != nullptr ) {
		numberOfElements = Weights()->GetObjectCount();
	}
}

CPtr<CDnnBlob> CFullyConnectedLayer::GetFreeTermData() const
{
	if( FreeTerms() == nullptr ) {
		return nullptr;
	}
	return FreeTerms()->GetCopy();
}

void CFullyConnectedLayer::SetFreeTermData( const CDnnBlob* newFreeTerms )
{
	if( newFreeTerms == nullptr ) {
		NeoAssert( FreeTerms() == nullptr || GetDnn() == nullptr );
		FreeTerms() = nullptr;
	} else {
		if( FreeTerms() != nullptr && GetDnn() != nullptr ) {
			NeoAssert( FreeTerms()->GetDataSize() == newFreeTerms->GetDataSize() );

			FreeTerms()->CopyFrom( newFreeTerms );
		} else {
			FreeTerms() = newFreeTerms->GetCopy();
		}
	}

	if( FreeTerms() != nullptr ) {
		numberOfElements = FreeTerms()->GetDataSize();
	}
}

void CFullyConnectedLayer::SetZeroFreeTerm( bool _isZeroFreeTerm )
{
	isZeroFreeTerm = _isZeroFreeTerm;
}

void CFullyConnectedLayer::ApplyBatchNormalization( CBatchNormalizationLayer& batchNorm )
{
	CPtr<CDnnBlob> params = batchNorm.GetFinalParams();
	if( params.Ptr() == nullptr || Weights().Ptr() == nullptr ) {
		return;
	}
	NeoAssert( params->GetObjectSize() == numberOfElements );
	CConstFloatHandle gamma = params->GetObjectData( 0 );
	CConstFloatHandle beta = params->GetObjectData( 1 );

	CFloatHandle weightData = Weights()->GetData();
	CFloatHandle freeTermData = FreeTerms()->GetData();
	int wieghtCount = Weights()->GetObjectSize();
	MathEngine().VectorEltwiseMultiply( freeTermData, gamma, freeTermData, numberOfElements );
	MathEngine().VectorAdd( freeTermData, beta, freeTermData, numberOfElements );
	for( int i = 0; i < numberOfElements; ++i ) {
		MathEngine().VectorMultiply( weightData, weightData, wieghtCount, gamma++ );
		weightData += wieghtCount;
	}
}

static const int FullyConnectedLayerVersion = 2000;

void CFullyConnectedLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( FullyConnectedLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( numberOfElements );
	archive.Serialize( isZeroFreeTerm );

	if( archive.IsLoading() ) {
		// Converts the free terms blob into a new tensor with the length in the first dimension not Channels
		CDnnBlob* freeTerms = FreeTerms();
		if( freeTerms != nullptr && freeTerms->DimSize( 0 ) != freeTerms->GetDataSize() ) {
			NeoAssert( freeTerms->GetChannelsCount() == freeTerms->GetDataSize() );
			CBlobDesc desc( CT_Float );
			desc.SetDimSize( 0, freeTerms->GetDataSize() );
			freeTerms->ReinterpretDimensions( desc );
		}
	}
}

CLayerWrapper<CFullyConnectedLayer> FullyConnected( int numberOfElements, bool isZeroFreeTerm )
{
	return CLayerWrapper<CFullyConnectedLayer>( "FullyConnected", [=]( CFullyConnectedLayer* result ) {
		result->SetNumberOfElements( numberOfElements );
		result->SetZeroFreeTerm( isZeroFreeTerm );
	} );
}

} // namespace NeoML
