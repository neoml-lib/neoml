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

#include <NeoML/Dnn/Layers/LossLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

static const float lossDefaultMaxGradientValue = 1e+6;

// The base loss function layer
// Can have 2 to 3 inputs: #0 - the network response, #1 - the correct result, #2 - vector weights (optional)
CLossLayer::CLossLayer( IMathEngine& mathEngine, const char* name, bool trainLabels ) :
	CBaseLayer( mathEngine, name, false ),
	lossDivider( 0.f ),
	lossWeight( 1.f ),
	minGradient( -lossDefaultMaxGradientValue ),
	maxGradient( lossDefaultMaxGradientValue ),
	trainLabels( trainLabels ),
	lossParam( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) )
{
	lossParam->GetData().SetValue( 0 );
}

void CLossLayer::SetTrainLabels( bool toSet )
{
	if( trainLabels != toSet ) {
		trainLabels = toSet;
		ForceReshape();
	}
}

void CLossLayer::SetMaxGradientValue(float maxValue)
{
	NeoAssert(maxValue > 0);

	minGradient = -maxValue;
	maxGradient = maxValue;
}

void CLossLayer::Reshape()
{
	CheckInputs();
	CheckLayerArchitecture( inputDescs.Size() >= 2, "loss layer with 1 input" );
	CheckLayerArchitecture( inputDescs.Size() <= 3, "loss layer with more than 3 inputs" );
	CheckLayerArchitecture( outputDescs.IsEmpty(), "loss layer has no output" );
	CheckLayerArchitecture( inputDescs[0].ObjectCount() == inputDescs[1].ObjectCount(), "object count mismatch" );
	CheckLayerArchitecture( !(trainLabels && inputDescs[1].GetDataType() == CT_Int), "can't train integer labels" );

	if( inputDescs.Size() > 2 ) {
		CheckLayerArchitecture( inputDescs[0].BatchWidth() == inputDescs[2].BatchWidth(),
			"weights batch width doesn't match result batch width" );
	}

	lossDivider = ( 1.f / inputDescs[0].ObjectCount() );
	resultBuffer = nullptr;
	weights = nullptr;

	lossGradientBlobs.DeleteAll();
	if( IsBackwardPerformed() ) {
		lossGradientBlobs.SetSize(trainLabels ? 2 : 1);
		lossGradientBlobs[0] = CDnnBlob::CreateBlob( MathEngine(), inputDescs[0] );
		RegisterRuntimeBlob(lossGradientBlobs[0]);
		if(trainLabels) {
			lossGradientBlobs[1] = CDnnBlob::CreateBlob( MathEngine(), inputDescs[0] );
			RegisterRuntimeBlob(lossGradientBlobs[1]);
		}
	}
}

static const int LossLayerVersion = 2000;

void CLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( LossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize(archive);

	if( archive.IsStoring() ) {
		archive << GetLossWeight();
	} else if( archive.IsLoading() ) {
		float tmp;
		archive >> tmp;
		SetLossWeight(tmp);
		lossParam->GetData().SetValue( 0 );
		weights = nullptr;
		resultBuffer = nullptr;
	} else {
		NeoAssert( false );
	}
}

void CLossLayer::BatchCalculateLossAndGradient(int, CConstFloatHandle, int, CConstFloatHandle, int, CFloatHandle, CFloatHandle)
{
	NeoAssert(false);
}

void CLossLayer::BatchCalculateLossAndGradient(int, CConstFloatHandle, int, CConstFloatHandle, int, CFloatHandle,
	CFloatHandle, CFloatHandle)
{
	NeoAssert(false);
}

void CLossLayer::BatchCalculateLossAndGradient(int, CConstFloatHandle, int, CConstIntHandle, int, CFloatHandle, CFloatHandle)
{
	NeoAssert(false);
}

void CLossLayer::RunOnce()
{
	// Set the weights
	if(inputBlobs.Size() <= 2) {
		if(weights == nullptr) {
			weights = CDnnBlob::CreateListBlob(MathEngine(), CT_Float, inputBlobs[0]->GetBatchLength(), 
				inputBlobs[0]->GetBatchWidth(), inputBlobs[0]->GetListSize(), 1);
			weights->Fill(1);
		}
	} else {
		// The weights blob already exists
		weights = inputBlobs[2];
	}
	// Calculate the loss value
	if(resultBuffer == nullptr) {
		resultBuffer = CDnnBlob::CreateListBlob(MathEngine(), CT_Float, inputBlobs[0]->GetBatchLength(), 
			inputBlobs[0]->GetBatchWidth(), inputBlobs[0]->GetListSize(), 1);
	}
	CFloatHandle dataLossGradient;
	if(lossGradientBlobs.Size() > 0) {
		dataLossGradient = lossGradientBlobs[0]->GetData();
	}
	CFloatHandle labelLossGradient;
	if(lossGradientBlobs.Size() > 1) {
		labelLossGradient = lossGradientBlobs[1]->GetData();
	}
	// When learning is turned off, dataLossGradient and labelLossGradient will be null
	// and only the loss function value will be calculated
	if(inputBlobs[1]->GetDataType() == CT_Int) {
		BatchCalculateLossAndGradient(inputBlobs[0]->GetObjectCount(),
			inputBlobs[0]->GetData(), inputBlobs[0]->GetObjectSize(),
			inputBlobs[1]->GetData<int>(), inputBlobs[1]->GetObjectSize(),
			resultBuffer->GetData(), dataLossGradient);
	} else {
		if(!trainLabels) {
			BatchCalculateLossAndGradient(inputBlobs[0]->GetObjectCount(),
				inputBlobs[0]->GetData(), inputBlobs[0]->GetObjectSize(),
				inputBlobs[1]->GetData(), inputBlobs[1]->GetObjectSize(),
				resultBuffer->GetData(), dataLossGradient);
		} else {
			BatchCalculateLossAndGradient(inputBlobs[0]->GetObjectCount(),
				inputBlobs[0]->GetData(), inputBlobs[0]->GetObjectSize(),
				inputBlobs[1]->GetData(), inputBlobs[1]->GetObjectSize(),
				resultBuffer->GetData(), dataLossGradient, labelLossGradient);
		}
	}
	// Take weights into account
	MathEngine().VectorDotProduct(weights->GetData(), resultBuffer->GetData(),
		resultBuffer->GetObjectCount(), lossParam->GetData() );
	MathEngine().VectorMultiply( lossParam->GetData(), lossParam->GetData(), 1, lossDivider );
}

void CLossLayer::BackwardOnce()
{
	// Averaging factor for calculating the loss gradient, takes lossWeight into account
	const float lossGradientDivider = lossDivider * lossWeight;

	for(int i = 0; i < lossGradientBlobs.Size(); i++) {
		// Take weights into account
		MathEngine().MultiplyDiagMatrixByMatrix( weights->GetData(), weights->GetDataSize(),
			lossGradientBlobs[i]->GetData(), inputDiffBlobs[i]->GetObjectSize(),
			inputDiffBlobs[i]->GetData(), inputDiffBlobs[i]->GetDataSize() );
		MathEngine().VectorMultiply( inputDiffBlobs[i]->GetData(), inputDiffBlobs[i]->GetData(),
			inputDiffBlobs[i]->GetDataSize(), lossGradientDivider );
		// The gradients might take "huge" values that will lead to incorrect behaviour
		// Cut these values down
		MathEngine().VectorMinMax( inputDiffBlobs[i]->GetData(), inputDiffBlobs[i]->GetData(),
			inputDiffBlobs[i]->GetDataSize(), minGradient, maxGradient );
	}
}

template<class T>
float CLossLayer::testImpl(int batchSize, CConstFloatHandle data, int vectorSize, CTypedMemoryHandle<const T> label,
	int labelSize, CConstFloatHandle dataDelta)
{
	int totalSize = batchSize * vectorSize;

	CFloatHandleVar lossValue(MathEngine(), batchSize);	// the function value in data point
	CFloatHandleVar lossGradient(MathEngine(), totalSize); // the function gradient in data point
	CFloatHandleVar dataShift(MathEngine(), totalSize); // the data + dataDelta point
	CFloatHandleVar lossValueShift(MathEngine(), batchSize); // the function value in data + dataDelta point
	CFloatHandleVar lossValueShiftApp(MathEngine(), batchSize); // the function approximation in data + dataDelta point
	CFloatHandleStackVar l2( MathEngine() ); // L2-measure (lossValueShiftApp - lossValueShift)

	CPtr<CDnnBlob> oldWeights = weights;
	weights = CDnnBlob::CreateVector(MathEngine(), CT_Float, batchSize);
	weights->Fill(1);

	// Estimate
	BatchCalculateLossAndGradient(batchSize, data, vectorSize,
		label, labelSize, lossValue.GetHandle(), lossGradient.GetHandle());

	MathEngine().VectorAdd(data, dataDelta, dataShift.GetHandle(), totalSize);
	BatchCalculateLossAndGradient(batchSize, dataShift.GetHandle(), vectorSize,
		label, labelSize, lossValueShift.GetHandle(), CFloatHandle());

	for(int i = 0; i < batchSize; ++i) {
		MathEngine().VectorDotProduct(lossGradient.GetHandle() + i * vectorSize,
			dataDelta + i * vectorSize, vectorSize, lossValueShiftApp.GetHandle() + i);
	}
	MathEngine().VectorAdd(lossValueShiftApp.GetHandle(), lossValue.GetHandle(),
		lossValueShiftApp.GetHandle(), batchSize);
	MathEngine().VectorSub(lossValueShiftApp.GetHandle(), lossValueShift.GetHandle(),
		lossValueShiftApp.GetHandle(), batchSize);
	MathEngine().VectorDotProduct(lossValueShiftApp.GetHandle(), lossValueShiftApp.GetHandle(),
		batchSize, l2.GetHandle());

	float res = l2.GetHandle().GetValue() / batchSize;

	weights = oldWeights; // restore the old weight values

	return res;
}

float CLossLayer::Test(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label, int labelSize,
	CConstFloatHandle dataDelta)
{
	return testImpl(batchSize, data, vectorSize, label, labelSize, dataDelta);
}

float CLossLayer::Test(int batchSize, CConstFloatHandle data, int vectorSize, CConstIntHandle label, int labelSize,
	CConstFloatHandle dataDelta)
{
	return testImpl(batchSize, data, vectorSize, label, labelSize, dataDelta);
}

float CLossLayer::TestRandom(CRandom& random, int batchSize, float dataLabelMin, float dataLabelMax, float deltaAbsMax,
	int vectorSize)
{
	int totalSize = batchSize * vectorSize;

	CArray<float> temp;

	CFloatHandleVar data( MathEngine(), totalSize );
	temp.SetSize(totalSize);
	for(int i = 0; i < totalSize; ++i) {
		temp[i] = (float)random.Uniform(dataLabelMin, dataLabelMax);
	}
	MathEngine().DataExchangeTyped(data.GetHandle(), temp.GetPtr(), totalSize);

	CFloatHandleVar label( MathEngine(), totalSize );
	temp.SetSize(totalSize);
	for(int i = 0; i < totalSize; ++i) {
		temp[i] = (float)random.Uniform(dataLabelMin, dataLabelMax);
	}
	MathEngine().DataExchangeTyped(label.GetHandle(), temp.GetPtr(), totalSize);

	NeoAssert(deltaAbsMax > 0);
	CFloatHandleVar delta( MathEngine(), totalSize );
	temp.SetSize(totalSize);
	for(int i = 0; i < totalSize; ++i) {
		temp[i] = (float)random.Uniform(-deltaAbsMax, deltaAbsMax);
	}
	MathEngine().DataExchangeTyped(delta.GetHandle(), temp.GetPtr(), totalSize);

	return Test(batchSize, data.GetHandle(), vectorSize, label.GetHandle(), vectorSize, delta.GetHandle());
}

float CLossLayer::TestRandom(CRandom& random, int batchSize, float dataMin, float dataMax, int labelMax, float deltaAbsMax,
	int vectorSize)
{
	int totalSize = batchSize * vectorSize;

	CArray<float> temp;

	CFloatHandleVar data( MathEngine(), totalSize );
	temp.SetSize(totalSize);
	for(int i = 0; i < totalSize; ++i) {
		temp[i] = (float)random.Uniform(dataMin, dataMax);
	}
	MathEngine().DataExchangeTyped(data.GetHandle(), temp.GetPtr(), totalSize);

	NeoAssert(labelMax > 0);
	CPtr<CDnnBlob> label = CDnnBlob::CreateVector(MathEngine(), CT_Int, batchSize);
	CArray<int> tempInt;
	tempInt.SetSize(batchSize);
	for(int i = 0; i < batchSize; ++i) {
		tempInt[i] = random.UniformInt(0, labelMax - 1);
	}
	MathEngine().DataExchangeTyped(label->GetData<int>(), tempInt.GetPtr(), batchSize);

	NeoAssert(deltaAbsMax > 0);
	CFloatHandleVar delta( MathEngine(), totalSize );
	temp.SetSize(totalSize);
	for(int i = 0; i < totalSize; ++i) {
		temp[i] = (float)random.Uniform(-deltaAbsMax, deltaAbsMax);
	}
	MathEngine().DataExchangeTyped(delta.GetHandle(), temp.GetPtr(), totalSize);

	return Test(batchSize, data.GetHandle(), vectorSize, label->GetData<int>(), 1, delta.GetHandle());
}

} // namespace NeoML
