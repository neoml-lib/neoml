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

#include <NeoML/Dnn/Layers/LossLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

static const float MaxGradientValue = 1e+6;

// The base loss function layer
// Can have 2 to 3 inputs: #0 - the network response, #1 - the correct result, #2 - vector weights (optional)
CLossLayer::CLossLayer( IMathEngine& mathEngine, const char* name, bool trainLabels ) :
	CBaseLayer( mathEngine, name, false ),
	trainLabels( trainLabels ),
	params( CDnnBlob::CreateVector( mathEngine, CT_Float, P_Count ) )
{
	params->GetData().SetValueAt( P_LossWeight, 1 );
	params->GetData().SetValueAt( P_Loss, 0 );
	params->GetData().SetValueAt( P_MinGradient, -MaxGradientValue );
	params->GetData().SetValueAt( P_MaxGradient, MaxGradientValue );
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

	params->GetData().SetValueAt( P_MinGradient, -maxValue );
	params->GetData().SetValueAt( P_MaxGradient, maxValue );
}

void CLossLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( inputDescs.Size() >= 2, GetName(), "loss layer with 1 input" );
	CheckArchitecture( inputDescs.Size() <= 3, GetName(), "loss layer with more than 3 inputs" );
	CheckArchitecture( inputDescs[0].ObjectCount() == inputDescs[1].ObjectCount(), GetName(), "object count mismatch" );
	CheckArchitecture( !(trainLabels && inputDescs[1].GetDataType() == CT_Int), GetName(), "can't train integer labels" );

	if( inputDescs.Size() > 2 ) {
		CheckArchitecture( inputDescs[0].BatchWidth() == inputDescs[2].BatchWidth(),
			GetName(), "weights batch width doesn't match result batch width" );
	}

	params->GetData().SetValueAt( P_LossDivider, 1.f / inputDescs[0].ObjectCount() );
	MathEngine().VectorEltwiseMultiply( params->GetData( {P_LossDivider} ), params->GetData( {P_LossWeight} ),
		params->GetData( {P_LossGradientDivider} ), 1 );

	resultBuffer = 0;
	weights = 0;

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
		params->GetData().SetValueAt( P_Loss, 0 );
		weights = 0;
		resultBuffer = 0;
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
		if(weights == 0) {
			weights = CDnnBlob::CreateListBlob(MathEngine(), CT_Float, inputBlobs[0]->GetBatchLength(), 
				inputBlobs[0]->GetBatchWidth(), inputBlobs[0]->GetListSize(), 1);
			weights->Fill(1);
		}
	} else {
		// The weights blob already exists
		weights = inputBlobs[2];
	}
	// Calculate the loss value
	if(resultBuffer == 0) {
		resultBuffer = CDnnBlob::CreateListBlob(MathEngine(), CT_Float, inputBlobs[0]->GetBatchLength(), 
			inputBlobs[0]->GetBatchWidth(), inputBlobs[0]->GetListSize(), 1);
	}
	CFloatHandle dataLossGradient, labelLossGradient;
	if(lossGradientBlobs.Size() > 0) {
		dataLossGradient = lossGradientBlobs[0]->GetData();
	}
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
		resultBuffer->GetObjectCount(), params->GetData( { P_Loss } ) );
	MathEngine().VectorMultiply( params->GetData( { P_Loss } ), params->GetData( { P_Loss } ),
		1, params->GetData( { P_LossDivider } ) );
}

void CLossLayer::BackwardOnce()
{
	for(int i = 0; i < lossGradientBlobs.Size(); i++) {
		// Take weights into account
		MathEngine().MultiplyDiagMatrixByMatrix( weights->GetData(), weights->GetDataSize(),
			lossGradientBlobs[i]->GetData(), inputDiffBlobs[i]->GetObjectSize(),
			inputDiffBlobs[i]->GetData(), inputDiffBlobs[i]->GetDataSize() );
		MathEngine().VectorMultiply( inputDiffBlobs[i]->GetData(), inputDiffBlobs[i]->GetData(),
			inputDiffBlobs[i]->GetDataSize(), params->GetData( { P_LossGradientDivider } ) );
		// The gradients might take "huge" values that will lead to incorrect behaviour
		// Cut these values down
		MathEngine().VectorMinMax( inputDiffBlobs[i]->GetData(), inputDiffBlobs[i]->GetData(),
			inputDiffBlobs[i]->GetDataSize(), params->GetData( { P_MinGradient } ), params->GetData( { P_MaxGradient } ) );
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
