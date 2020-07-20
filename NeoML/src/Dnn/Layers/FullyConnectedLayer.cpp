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

#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CFullyConnectedLayer::CFullyConnectedLayer( IMathEngine& mathEngine, const char* name ) :
	CBaseLayer( mathEngine, name == nullptr ? "CCnnFullyConnectedLayer" : name, true ),
	numberOfElements(0),
	isZeroFreeTerm(false)
{
	paramBlobs.SetSize(2);
}

CFullyConnectedLayer::~CFullyConnectedLayer()
{
}

void CFullyConnectedLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( GetInputCount() == GetOutputCount(),
		GetName(), "fully connected layer with different numbers of input and output" );
	for(int i = 0; i < GetInputCount(); i++) {
		if(Weights() == 0) {
			// Create a weights matrix
			CBlobDesc weightsDesc = inputDescs[i];
			weightsDesc.SetDimSize(BD_BatchLength, 1);
			weightsDesc.SetDimSize(BD_BatchWidth, numberOfElements);
			weightsDesc.SetDimSize(BD_ListSize, 1);
			Weights() = CDnnBlob::CreateBlob(MathEngine(), CT_Float, weightsDesc);
			// Initialize
			InitializeParamBlob(i, *Weights());
		} else {
			CheckArchitecture( Weights()->GetObjectCount() == numberOfElements,
				GetName(), "weights number is not equal to number of elements" );
			CheckArchitecture( Weights()->GetObjectSize() == inputDescs[i].ObjectSize(),
				GetName(), "weights size mismatch" );
		}

		if(FreeTerms() == 0) {
			FreeTerms() = CDnnBlob::CreateVector(MathEngine(), CT_Float, numberOfElements);
			// Initialize
			FreeTerms()->Fill(0);
		} else {
			CheckArchitecture( FreeTerms()->GetDataSize() == numberOfElements,
				GetName(), "free terms num is not equal to number of elements" );
		}

		// For each layer element there is a channel in the output blob
		outputDescs[i] = inputDescs[i];
		outputDescs[i].SetDimSize(BD_Height, 1);
		outputDescs[i].SetDimSize(BD_Width, 1);
		outputDescs[i].SetDimSize(BD_Depth, 1);
		outputDescs[i].SetDimSize(BD_Channels, numberOfElements);
	}
}

void CFullyConnectedLayer::RunOnce()
{
	for( int i = 0; i < GetInputCount(); i++ ) {
		CConstFloatHandle inputData = inputBlobs[i]->GetData();
		CFloatHandle outputData = outputBlobs[i]->GetData();
		CConstFloatHandle weightData = Weights()->GetData();

		MathEngine().MultiplyMatrixByTransposedMatrix(inputData, inputBlobs[i]->GetObjectCount(),
			inputBlobs[i]->GetObjectSize(), inputBlobs[i]->GetObjectSize(),
			weightData, numberOfElements, Weights()->GetObjectSize(),
			outputData, outputBlobs[i]->GetObjectSize(), outputBlobs[i]->GetObjectSize() * inputBlobs[i]->GetObjectCount());

		if( !isZeroFreeTerm ) {
			MathEngine().AddVectorToMatrixRows(1, outputData, outputData, inputBlobs[i]->GetObjectCount(),
				outputBlobs[i]->GetObjectSize(), FreeTerms()->GetData());
		}
	}
}

void CFullyConnectedLayer::BackwardOnce()
{
	for( int i = 0; i < outputDiffBlobs.Size(); i++ ) {
		MathEngine().MultiplyMatrixByMatrix(1, outputDiffBlobs[i]->GetData(), inputBlobs[i]->GetObjectCount(),
			outputDiffBlobs[i]->GetObjectSize(), Weights()->GetData(), Weights()->GetObjectSize(),
			inputDiffBlobs[i]->GetData(), inputDiffBlobs[i]->GetDataSize());
	}
}

void CFullyConnectedLayer::LearnOnce()
{
	for( int out = 0; out < outputDiffBlobs.Size(); out++ ) {
		MathEngine().MultiplyTransposedMatrixByMatrixAndAdd(outputDiffBlobs[out]->GetData(),
			outputDiffBlobs[out]->GetObjectCount(), numberOfElements, numberOfElements,
			inputBlobs[out]->GetData(), inputBlobs[out]->GetObjectSize(), inputBlobs[out]->GetObjectSize(),
			WeightsDiff()->GetData(), WeightsDiff()->GetObjectSize(), WeightsDiff()->GetDataSize());

		if( !isZeroFreeTerm ) {
			MathEngine().SumMatrixRowsAdd(1, FreeTermsDiff()->GetData(),
				outputDiffBlobs[out]->GetData(), outputDiffBlobs[out]->GetObjectCount(), numberOfElements);
		}
	}
}

void CFullyConnectedLayer::FilterLayerParams( float threshold )
{
	for( int blobIndex = 0; blobIndex < paramBlobs.Size(); ++blobIndex ) {
		if( paramBlobs[blobIndex] != 0 ) {
			MathEngine().FilterSmallValues( paramBlobs[blobIndex]->GetData(),
				paramBlobs[blobIndex]->GetDataSize(), threshold );
		}
	}
}

void CFullyConnectedLayer::SetNumberOfElements(int newNumberOfElements)
{
	NeoAssert( ( Weights() == 0 && FreeTerms() == 0 ) || numberOfElements == newNumberOfElements );
	numberOfElements = newNumberOfElements;
}

CPtr<CDnnBlob> CFullyConnectedLayer::GetWeightsData() const
{
	if(Weights() == 0) {
		return 0;
	}

	return Weights()->GetCopy();
}

void CFullyConnectedLayer::SetWeightsData(CDnnBlob* newWeights)
{
	if(newWeights == 0) {
		NeoAssert(Weights() == 0 || GetDnn() == 0);
		Weights() = 0;
	} else if(Weights() != 0 && GetDnn() != 0) {
		NeoAssert(Weights()->GetObjectCount() == newWeights->GetObjectCount());
		NeoAssert(Weights()->GetObjectSize() == newWeights->GetObjectSize());
		Weights()->CopyFrom(newWeights);
	} else {
		Weights() = newWeights->GetCopy();
	}

	if(Weights() != 0) {
		numberOfElements = Weights()->GetObjectCount();
	}
}

CPtr<CDnnBlob> CFullyConnectedLayer::GetFreeTermData() const
{
	if(FreeTerms() == 0) {
		return 0;
	}

	return FreeTerms()->GetCopy();
}

void CFullyConnectedLayer::SetFreeTermData(CDnnBlob* newFreeTerms)
{
	if(newFreeTerms == 0) {
		NeoAssert(FreeTerms() == 0 || GetDnn() == 0);
		FreeTerms() = 0;
	} else {
		if(FreeTerms() != 0 && GetDnn() != 0) {
			NeoAssert(FreeTerms()->GetDataSize() == newFreeTerms->GetDataSize());

			FreeTerms()->CopyFrom(newFreeTerms);
		} else {
			FreeTerms() = newFreeTerms->GetCopy();
		}
	}

	if(FreeTerms() != 0) {
		numberOfElements = FreeTerms()->GetDataSize();
	}
}

void CFullyConnectedLayer::SetZeroFreeTerm(bool _isZeroFreeTerm)
{
	isZeroFreeTerm = _isZeroFreeTerm;
}

void CFullyConnectedLayer::ApplyBatchNormalization(CBatchNormalizationLayer& batchNorm)
{
	CPtr<CDnnBlob> params = batchNorm.GetFinalParams();
	if(params.Ptr() == 0 || Weights().Ptr() == 0) {
		return;
	}
	NeoAssert(params->GetObjectSize() == numberOfElements);
	CConstFloatHandle gamma = params->GetObjectData( 0 );
	CConstFloatHandle beta = params->GetObjectData( 1 );

	CFloatHandle weightData = Weights()->GetData();
	CFloatHandle freeTermData = FreeTerms()->GetData();
	int wieghtCount = Weights()->GetObjectSize();
	MathEngine().VectorEltwiseMultiply(freeTermData, gamma, freeTermData, numberOfElements);
	MathEngine().VectorAdd(freeTermData, beta, freeTermData, numberOfElements);
	for(int i = 0; i < numberOfElements; ++i) {
		MathEngine().VectorMultiply(weightData, weightData, wieghtCount, gamma++);
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
		if( freeTerms != 0 && freeTerms->DimSize(0) != freeTerms->GetDataSize() ) {
			NeoAssert( freeTerms->GetChannelsCount() == freeTerms->GetDataSize() );
			CBlobDesc desc( CT_Float );
			desc.SetDimSize( 0, freeTerms->GetDataSize() );
			freeTerms->ReinterpretDimensions( desc );
		}
	}
}

} // namespace NeoML
