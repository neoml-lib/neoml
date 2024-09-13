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
	CBaseLayer( mathEngine, name, /*isLearnable*/false ),
	lossParam( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	trainLabels( trainLabels )
{
	lossParam->GetData().SetValue( 0 );
	SetMaxGradientValue( lossDefaultMaxGradientValue );
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

static constexpr int lossLayerVersion = 2001;

void CLossLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( lossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize(archive);

	archive.Serialize( lossWeight );
	if( version >= 2001 ) {
		archive.Serialize( maxGradient );
		archive.Serialize( trainLabels );
	} else {
		maxGradient = lossDefaultMaxGradientValue;
		trainLabels = false;
	}

	if( archive.IsLoading() ) {
		lossParam->GetData().SetValue( 0 );
		weights = nullptr;
		resultBuffer = nullptr;
		lossDivider = 0.f;
		SetMaxGradientValue( maxGradient );
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
	const int totalSize = batchSize * vectorSize;
	CFloatHandleStackVar temp( MathEngine(), ( 2 * totalSize ) + ( 3 * batchSize ) + 1 );

	CFloatHandle lossValue = temp; // the function value in data point
	CFloatHandle lossGradient = lossValue + batchSize; // the function gradient in data point
	CFloatHandle dataShift = lossGradient + totalSize; // the data + dataDelta point
	CFloatHandle lossValueShift = dataShift + totalSize; // the function value in data + dataDelta point
	CFloatHandle lossValueShiftApp = lossValueShift + batchSize; // the function approximation in data + dataDelta point
	CFloatHandle l2 = lossValueShiftApp + batchSize; // L2-measure (lossValueShiftApp - lossValueShift)

	CPtr<CDnnBlob> oldWeights = weights;
	weights = CDnnBlob::CreateVector(MathEngine(), CT_Float, batchSize);
	weights->Fill(1);

	// Estimate
	BatchCalculateLossAndGradient(batchSize, data, vectorSize,
		label, labelSize, lossValue, lossGradient);

	MathEngine().VectorAdd(data, dataDelta, dataShift, totalSize);
	BatchCalculateLossAndGradient(batchSize, dataShift, vectorSize,
		label, labelSize, lossValueShift, CFloatHandle{});

	for(int i = 0; i < batchSize; ++i) {
		MathEngine().VectorDotProduct(lossGradient + i * vectorSize,
			dataDelta + i * vectorSize, vectorSize, lossValueShiftApp + i);
	}
	MathEngine().VectorAdd(lossValueShiftApp, lossValue,
		lossValueShiftApp, batchSize);
	MathEngine().VectorSub(lossValueShiftApp, lossValueShift,
		lossValueShiftApp, batchSize);
	MathEngine().VectorDotProduct(lossValueShiftApp, lossValueShiftApp,
		batchSize, l2);

	float res = l2.GetValue() / batchSize;

	weights = oldWeights; // restore the old weight values

	return res;
}

float CLossLayer::Test(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label, int labelSize,
	const CConstFloatHandle& dataDelta)
{
	return testImpl(batchSize, data, vectorSize, label, labelSize, dataDelta);
}

float CLossLayer::Test(int batchSize, CConstFloatHandle data, int vectorSize, CConstIntHandle label, int labelSize,
	const CConstFloatHandle& dataDelta)
{
	return testImpl(batchSize, data, vectorSize, label, labelSize, dataDelta);
}

float CLossLayer::TestRandom(CRandom& random, int batchSize, float dataLabelMin, float dataLabelMax, float deltaAbsMax,
	int vectorSize)
{
	NeoAssert( batchSize > 0 && vectorSize > 0 );
	NeoAssert( dataLabelMin < dataLabelMax && deltaAbsMax > 0 );

	const int totalSize = batchSize * vectorSize;
	CFloatHandleStackVar temp( MathEngine(), totalSize * 3 );

	CFloatHandle data = temp;
	CFloatHandle label = data + totalSize;
	CFloatHandle delta = label + totalSize;
	{
		CArray<float> buf;
		buf.SetSize( totalSize );

		for( int i = 0; i < totalSize; ++i ) {
			buf[i] = static_cast<float>( random.Uniform(dataLabelMin, dataLabelMax) );
		}
		MathEngine().DataExchangeTyped(data, buf.GetPtr(), totalSize);

		for( int i = 0; i < totalSize; ++i ) {
			buf[i] = static_cast<float>( random.Uniform(dataLabelMin, dataLabelMax) );
		}
		MathEngine().DataExchangeTyped(label, buf.GetPtr(), totalSize);

		for( int i = 0; i < totalSize; ++i ) {
			buf[i] = static_cast<float>( random.Uniform(-deltaAbsMax, deltaAbsMax) );
		}
		MathEngine().DataExchangeTyped(delta, buf.GetPtr(), totalSize);
	}
	return Test(batchSize, data, vectorSize, label, vectorSize, delta);
}

float CLossLayer::TestRandom( CRandom& random, int batchSize, float dataMin, float dataMax, int labelMax, float deltaAbsMax,
	int vectorSize )
{
	NeoAssert( batchSize > 0 && vectorSize > 0 );
	NeoAssert( dataMin < dataMax && labelMax > 0 && deltaAbsMax > 0 );

	const int totalSize = batchSize * vectorSize;
	CFloatHandleStackVar temp( MathEngine(), totalSize * 2 );

	CFloatHandle data = temp;
	CFloatHandle delta = data + totalSize;
	{
		CArray<float> buf;
		buf.SetSize( totalSize );

		for( int i = 0; i < totalSize; ++i ) {
			buf[i] = static_cast<float>( random.Uniform(dataMin, dataMax) );
		}
		MathEngine().DataExchangeTyped(data, buf.GetPtr(), totalSize);

		for( int i = 0; i < totalSize; ++i ) {
			buf[i] = static_cast<float>( random.Uniform(-deltaAbsMax, deltaAbsMax) );
		}
		MathEngine().DataExchangeTyped(delta, buf.GetPtr(), totalSize);
	}

	CIntHandleStackVar label(MathEngine(), batchSize);
	{
		CArray<int> bufInt;
		bufInt.SetSize( batchSize );
		for( int i = 0; i < batchSize; ++i ) {
			bufInt[i] = random.UniformInt(0, labelMax - 1);
		}
		MathEngine().DataExchangeTyped<int>(label, bufInt.GetPtr(), batchSize);
	}
	return Test(batchSize, data, vectorSize, label, 1, delta);
}

} // namespace NeoML
