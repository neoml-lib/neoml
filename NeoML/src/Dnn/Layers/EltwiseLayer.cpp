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
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

void CEltwiseBaseLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( inputDescs.Size() > 1, GetName(), "eltwise layer with single input" );
	CheckArchitecture( !IsBackwardPerformed() || inputDescs[0].GetDataType() == CT_Float, GetName(), "integer eltwise backward" );

	for( int i = 1; i < inputDescs.Size(); ++i ) {
		CheckArchitecture( inputDescs[i].HasEqualDimensions(inputDescs[0]),
			GetName(), "eltwise input size mismatch (batchSize mismatch)" );
		const CBlobDesc& blobDesc = inputDescs[i];
		CheckArchitecture( blobDesc.GetDataType() == inputDescs[0].GetDataType(),
			GetName(), "input types mismatch" );
		CheckArchitecture( inputDescs[i].GetDataType() == inputDescs[0].GetDataType(),
			GetName(), "input types mismatch" );
	}

	outputDescs[0] = inputDescs[0];
}

static const int EltwiseBaseLayerVersion = 2000;

void CEltwiseBaseLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( EltwiseBaseLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
static void eltwiseSumRunOnce( const CObjectArray<CDnnBlob>& inputBlobs, const CObjectArray<CDnnBlob>& outputBlobs )
{
	IMathEngine& mathEngine = inputBlobs[0]->GetMathEngine();
	CTypedMemoryHandle<T> output = outputBlobs[0]->GetData<T>();
	const int dataSize = outputBlobs[0]->GetDataSize();

	mathEngine.VectorAdd( inputBlobs[0]->GetData<T>(), inputBlobs[1]->GetData<T>(), output, dataSize );
	for( int i = 2; i < inputBlobs.Size(); ++i ) {
		mathEngine.VectorAdd( output, inputBlobs[i]->GetData<T>(), output, dataSize );
	}
}

void CEltwiseSumLayer::RunOnce()
{
	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		eltwiseSumRunOnce<float>( inputBlobs, outputBlobs );
	} else {
		eltwiseSumRunOnce<int>( inputBlobs, outputBlobs );
	}
}

void CEltwiseSumLayer::BackwardOnce()
{
	NeoAssert( inputBlobs[0]->GetDataType() == CT_Float );
	for( int i = 0; i < inputDiffBlobs.Size(); ++i ) {
		MathEngine().VectorCopy( inputDiffBlobs[i]->GetData(), outputDiffBlobs[0]->GetData(),
			inputDiffBlobs[i]->GetDataSize() );
	}
}

static const int EltwiseSumLayerVersion = 2000;

void CEltwiseSumLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( EltwiseSumLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CEltwiseBaseLayer::Serialize( archive );
}

CLayerWrapper<CEltwiseSumLayer> Sum()
{
	return CLayerWrapper<CEltwiseSumLayer>( "Sum" );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

void CEltwiseSubLayer::Reshape()
{
	// This layer must have 2 inputs
	CheckArchitecture( inputDescs.Size() == 2, GetName(), "EltwiseSub layer must have 2 inputs" );
	CEltwiseBaseLayer::Reshape();
}

template<class T>
static void eltwiseSubRunOnce( const CObjectArray<CDnnBlob>& inputBlobs, const CObjectArray<CDnnBlob>& outputBlobs )
{
	IMathEngine& mathEngine = inputBlobs[0]->GetMathEngine();
	CTypedMemoryHandle<T> output = outputBlobs[0]->GetData<T>();
	const int dataSize = outputBlobs[0]->GetDataSize();

	mathEngine.VectorSub( inputBlobs[0]->GetData<T>(), inputBlobs[1]->GetData<T>(), output, dataSize );
	for( int i = 2; i < inputBlobs.Size(); ++i ) {
		mathEngine.VectorSub( output, inputBlobs[i]->GetData<T>(), output, dataSize );
	}
}

void CEltwiseSubLayer::RunOnce()
{
	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		eltwiseSubRunOnce<float>( inputBlobs, outputBlobs );
	} else {
		eltwiseSubRunOnce<int>( inputBlobs, outputBlobs );
	}
}

void CEltwiseSubLayer::BackwardOnce()
{
	for( int i = 0; i < inputDiffBlobs.Size(); ++i ) {
		MathEngine().VectorCopy( inputDiffBlobs[i]->GetData(), outputDiffBlobs[0]->GetData(),
			inputDiffBlobs[i]->GetDataSize() );
	}
}

static const int EltwiseSubLayerVersion = 0;

void CEltwiseSubLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( EltwiseSubLayerVersion );
	CEltwiseBaseLayer::Serialize( archive );
}

CLayerWrapper<CEltwiseSubLayer> Sub()
{
	return CLayerWrapper<CEltwiseSubLayer>( "Sub" );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void CEltwiseMulLayer::RunOnce()
{
	CFloatHandle output = outputBlobs[0]->GetData();
	int dataSize = outputBlobs[0]->GetDataSize();

	MathEngine().VectorEltwiseMultiply( inputBlobs[0]->GetData(), inputBlobs[1]->GetData(), output, dataSize );

	for( int i = 2; i < inputBlobs.Size(); ++i ) {
		MathEngine().VectorEltwiseMultiply( output, inputBlobs[i]->GetData(), output, dataSize );
	}
}

void CEltwiseMulLayer::BackwardOnce()
{
	int dataSize = inputDiffBlobs[0]->GetDataSize();
	for( int i = 0; i < inputDiffBlobs.Size(); ++i ) {
		for( int j = 0; j < inputBlobs.Size(); ++j ) {
			if( i == j ) {
				continue;
			}

			if( j == 0 || ( i == 0 && j == 1 ) ) {
				MathEngine().VectorEltwiseMultiply( outputDiffBlobs[0]->GetData(), inputBlobs[j]->GetData(),
					inputDiffBlobs[i]->GetData(), dataSize );
			} else {
				MathEngine().VectorEltwiseMultiply( inputDiffBlobs[i]->GetData(), inputBlobs[j]->GetData(),
					inputDiffBlobs[i]->GetData(), dataSize );
			}
		}
	}
}

static const int EltwiseMulLayerVersion = 2000;

void CEltwiseMulLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( EltwiseMulLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CEltwiseBaseLayer::Serialize( archive );
}

CLayerWrapper<CEltwiseMulLayer> Mul()
{
	return CLayerWrapper<CEltwiseMulLayer>( "Mul" );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void CEltwiseNegMulLayer::Reshape()
{
	CEltwiseBaseLayer::Reshape();
	oneVector = 0;
	negInputBlob = 0;
	if(IsBackwardPerformed()) {
		negInputBlob = CDnnBlob::CreateBlob( MathEngine(), inputDescs[0] );
		RegisterRuntimeBlob(negInputBlob);
	}
}

void CEltwiseNegMulLayer::RunOnce()
{
	CFloatHandle output = outputBlobs[0]->GetData();
	int dataSize = outputBlobs[0]->GetDataSize();

	if(oneVector == 0) {
		oneVector = inputBlobs[0]->GetClone();
		oneVector->Fill(1);
	}
	CFloatHandle negInput = negInputBlob != 0 ? negInputBlob->GetData() : output;
	MathEngine().VectorSub(oneVector->GetData(), inputBlobs[0]->GetData(), negInput, dataSize);
	MathEngine().VectorEltwiseMultiply(negInput, inputBlobs[1]->GetData(), output, dataSize);

	for(int i = 2; i < inputBlobs.Size(); ++i) {
		MathEngine().VectorEltwiseMultiply(output, inputBlobs[i]->GetData(), output, dataSize);
	}
}

void CEltwiseNegMulLayer::BackwardOnce()
{
	int dataSize = inputDiffBlobs[0]->GetDataSize();
	for(int i = 0; i < inputDiffBlobs.Size(); ++i) {
		for(int j = 0; j < inputBlobs.Size(); ++j) {
			if(i == j) {
				continue;
			}
			if(j == 0) {
				MathEngine().VectorEltwiseMultiply(outputDiffBlobs[0]->GetData(), negInputBlob->GetData(),
					inputDiffBlobs[i]->GetData(), dataSize);
			} else if(i == 0 && j == 1) {
				MathEngine().VectorEltwiseNegMultiply(outputDiffBlobs[0]->GetData(), inputBlobs[j]->GetData(),
					inputDiffBlobs[i]->GetData(), dataSize);
			} else {
				MathEngine().VectorEltwiseMultiply(inputDiffBlobs[i]->GetData(), inputBlobs[j]->GetData(),
					inputDiffBlobs[i]->GetData(), dataSize);
			}
		}
	}
}

static const int EltwiseNegMulLayerVersion = 2000;

void CEltwiseNegMulLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( EltwiseNegMulLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CEltwiseBaseLayer::Serialize( archive );
}

CLayerWrapper<CEltwiseNegMulLayer> NegMul()
{
	return CLayerWrapper<CEltwiseNegMulLayer>( "NegMul" );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void CEltwiseMaxLayer::Reshape()
{
	CEltwiseBaseLayer::Reshape();

	vectorsArray.DeleteAll();
	vectorsArray.SetSize(GetDnn()->GetMaxSequenceLength());
	diffVectorsArray.DeleteAll();
	diffVectorsArray.SetSize(GetDnn()->GetMaxSequenceLength());

	maxIndices = 0;
	if(IsBackwardPerformed()) {
		maxIndices = CDnnBlob::CreateBlob( MathEngine(), CT_Int, outputDescs[0] );
		RegisterRuntimeBlob(maxIndices);
	}
}

void CEltwiseMaxLayer::RunOnce()
{
	int dataSize = outputBlobs[0]->GetDataSize();
	CFloatHandle output = outputBlobs[0]->GetData();

	CArray<CConstFloatHandle>& vectors = vectorsArray[GetDnn()->GetCurrentSequencePos()];

	if( vectors.Size() == 0 ) {
		vectors.SetSize(inputBlobs.Size());
		for(int i = 0; i < inputBlobs.Size(); ++i) {
			vectors[i] = inputBlobs[i]->GetData();
		}
	}

	if( IsBackwardPerformed() ) {
		MathEngine().VectorFindMaxValueInSet(vectors.GetPtr(), vectors.Size(), output, maxIndices->GetData<int>(), dataSize);
	} else {
		MathEngine().VectorFindMaxValueInSet(vectors.GetPtr(), vectors.Size(), output, dataSize);
	}
}

void CEltwiseMaxLayer::BackwardOnce()
{
	CArray<CFloatHandle>& diffVectors = diffVectorsArray[GetDnn()->GetCurrentSequencePos()];

	if( diffVectors.Size() == 0 ) {
		diffVectors.SetSize(inputDiffBlobs.Size());
		for(int j = 0; j < inputDiffBlobs.Size(); ++j) {
			diffVectors[j] = inputDiffBlobs[j]->GetData();
		}
	}

	for( int j = 0; j < inputDiffBlobs.Size(); ++j ) {
		inputDiffBlobs[j]->Clear();
	}

	MathEngine().VectorSpreadValues( outputDiffBlobs[0]->GetData(), diffVectors.GetPtr(), diffVectors.Size(),
		maxIndices->GetData<int>(), outputDiffBlobs[0]->GetDataSize() );
}

static const int EltwiseMaxLayerVersion = 2000;

void CEltwiseMaxLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( EltwiseMaxLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CEltwiseBaseLayer::Serialize( archive );
}

CLayerWrapper<CEltwiseMaxLayer> Max()
{
	return CLayerWrapper<CEltwiseMaxLayer>( "Max" );
}

}
