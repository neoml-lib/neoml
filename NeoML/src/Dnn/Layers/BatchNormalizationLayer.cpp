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

#include <NeoML/Dnn/Layers/BatchNormalizationLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// The minimum batch size for correct operation of the algorithm
static const int MinBatchSize = 8;
// A small value to be added to the variance to avoid zero
static const float VarianceEpsilon = 1e-12f;

CBatchNormalizationLayer::CBatchNormalizationLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnBatchNormalizationLayer", true ),
	isChannelBased( true ),
	isZeroFreeTerm( false ),
	slowConvergenceRate( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	varianceEpsilon( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	fullBatchInv( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	varianceNorm( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	residual( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	varianceMult( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	useFinalParamsForInitialization( false ),
	isFinalParamDirty( false )
{
	SetSlowConvergenceRate(0.01f);
	varianceEpsilon->GetData().SetValue(VarianceEpsilon);
	paramBlobs.SetSize(1);
	SetBaseL1RegularizationMult(0); // by default, no regularization for batch normalization layer
	SetBaseL2RegularizationMult(0);
}

void CBatchNormalizationLayer::SetChannelBased(bool _isChannelBased)
{
	NeoAssert(GetDnn() == 0);
	isChannelBased = _isChannelBased;
}

void CBatchNormalizationLayer::ClearStatistics()
{
	updateFinalParams();
	paramBlobs[0] = 0;
	internalParams = 0;
}

// Initializes the statistics parameters if necessary
bool CBatchNormalizationLayer::checkAndCreateParams()
{
	bool isInit = false;

	if(paramBlobs[0] == 0) {
		paramBlobs[0] = finalParams->GetClone();
		CBlobDesc paramDesc = finalParams->GetDesc();
		paramDesc.SetDimSize(BD_BatchWidth, IPN_Count);
		internalParams = CDnnBlob::CreateBlob(MathEngine(), CT_Float, paramDesc);
		isInit = true;
	} else {
		NeoAssert(paramBlobs[0]->GetObjectCount() == PN_Count);
		NeoAssert(paramBlobs[0]->GetObjectSize() == finalParams->GetObjectSize());
		NeoAssert(internalParams->GetObjectCount() == IPN_Count);
		NeoAssert(internalParams->GetObjectSize() == finalParams->GetObjectSize());
	}
	if( useFinalParamsForInitialization ) {
		initializeFromFinalParams();
		useFinalParamsForInitialization = false;
		isInit = false; // All parameters have been set already in initializeFromFinalParams()
	}
	return isInit;
}

// Sets the parameters using the precalculated values from finalParams
void CBatchNormalizationLayer::initializeFromFinalParams()
{
	const int paramSize = finalParams->GetObjectSize();

	CPtr<CDnnBlob> params = finalParams;

	CConstFloatHandle finalBeta = params->GetObjectData( PN_Beta );
	CConstFloatHandle finalGamma = params->GetObjectData( PN_Gamma );

	CFloatHandle slowAverage = internalParams->GetObjectData( IPN_SlowAverage );
	CFloatHandle slowVariance = internalParams->GetObjectData( IPN_SlowVariance );
	CFloatHandle gamma = paramBlobs[0]->GetObjectData( PN_Gamma );
	CFloatHandle beta = paramBlobs[0]->GetObjectData(  PN_Beta );

	CPtr<CDnnBlob> ones = CDnnBlob::CreateVector( MathEngine(), CT_Float, paramSize );
	ones->Fill( 1.f );

	// Deduce the gamma, beta, slowVar, slowAvg values from the final data
	// We suppose the slow parameters to be equal to the current one because no history is available

	// gamma == slowVar == finalGamma ^ 2
	MathEngine().VectorEltwiseMultiply( finalGamma, finalGamma, gamma, paramSize );
	MathEngine().VectorCopy( slowVariance, gamma, paramSize );

	// beta == slowAvg = finalBeta / ( 1 - finalGamma )
	// No need to consider the IsZeroFreeTerm case because we will anyway be getting beta == slowAvg == 0
	MathEngine().VectorSub( ones->GetData(), finalGamma, slowAverage, paramSize );
	MathEngine().VectorInv( slowAverage, slowAverage, paramSize );
	MathEngine().VectorEltwiseMultiply( finalBeta, slowAverage, slowAverage, paramSize );
	MathEngine().VectorCopy( beta, slowAverage, paramSize );
}

void CBatchNormalizationLayer::getFullBatchAndObjectSize(int& fullBatchSize, int& objectSize)
{
	fullBatchSize = inputDescs[0].ObjectCount();
	if(isChannelBased) {
		fullBatchSize *= inputDescs[0].Width() * inputDescs[0].Height();
	}

	objectSize = inputDescs[0].BlobSize() / fullBatchSize;
}

void CBatchNormalizationLayer::SetSlowConvergenceRate(float rate)
{
	NeoAssert(0 < rate && rate <= 1);
	slowConvergenceRate->GetData().SetValue( rate );
	ForceReshape();
}

void CBatchNormalizationLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( inputDescs.Size() == 1, GetName(), "batch normalization with more than 1 input" );

	int fullBatchSize;
	int objectSize;
	getFullBatchAndObjectSize(fullBatchSize, objectSize);
	if(GetDnn() != 0 && GetDnn()->IsRecurrentMode()) {
		fullBatchSize /= GetDnn()->GetMaxSequenceLength();
	}

	CBlobDesc paramDesc = inputDescs[0];
	paramDesc.SetDimSize(BD_BatchLength, 1);
	paramDesc.SetDimSize(BD_BatchWidth, PN_Count);
	if(isChannelBased) {
		paramDesc.SetDimSize(BD_Height, 1);
		paramDesc.SetDimSize(BD_Width, 1);
		paramDesc.SetDimSize(BD_Depth, 1);
	}
	outputDescs[0] = inputDescs[0];

	if(finalParams == 0) {
		finalParams = CDnnBlob::CreateBlob(MathEngine(), CT_Float, paramDesc);
		MathEngine().VectorFill(finalParams->GetObjectData( PN_Gamma), 1.0, finalParams->GetObjectSize());
		MathEngine().VectorFill(finalParams->GetObjectData( PN_Beta), 0.0, finalParams->GetObjectSize());
	} else {
		CheckArchitecture( finalParams->GetObjectCount() == PN_Count,
			GetName(), "Parameters batch size must be 2" );
		CheckArchitecture( finalParams->GetObjectSize() == objectSize, 
			GetName(), "Object data size from params must be equal to actual object size" );
	}

	fullBatchInv->GetData().SetValue(1.f / fullBatchSize);
	float varianceNormValue  = (fullBatchSize > 1) ? (float)fullBatchSize / (fullBatchSize - 1) : 0;
	varianceNorm->GetData().SetValue(varianceNormValue);
	residual->GetData().SetValue(1);
	MathEngine().VectorSub(residual->GetData(), slowConvergenceRate->GetData(), residual->GetData(), 1);
	MathEngine().VectorEltwiseMultiply(slowConvergenceRate->GetData(), varianceNorm->GetData(), varianceMult->GetData(), 1);
	
	normalized  = 0;
	if( IsLearningPerformed() ) {
		normalized = CDnnBlob::CreateBlob( MathEngine(), inputDescs[0] );
		RegisterRuntimeBlob(normalized);
	}
}

void CBatchNormalizationLayer::RunOnce()
{
	if( IsLearningPerformed() ) {
		int fullBatchSize;
		int objectSize;
		getFullBatchAndObjectSize(fullBatchSize, objectSize);
		CheckArchitecture( fullBatchSize >= MinBatchSize,
			GetName(), "in batch normalization fullBatchSize is more than MinBatchSize" );

		runWhenLearning();
	} else {
		runWhenNoLearning();
	}
}

// Converts an input blob to an output blob using the parameters
void CBatchNormalizationLayer::processInput(const CPtr<CDnnBlob>& inputBlob, const CPtr<CDnnBlob>& paramBlob)
{
	CConstFloatHandle input = inputBlob->GetData();
	CFloatHandle output = outputBlobs[0]->GetData();
	CConstFloatHandle gammas = paramBlob->GetObjectData( PN_Gamma );
	CConstFloatHandle betas = paramBlob->GetObjectData( PN_Beta );

	int fullBatchSize;
	int objectSize;
	getFullBatchAndObjectSize(fullBatchSize, objectSize);

	MathEngine().MultiplyMatrixByDiagMatrix(input, fullBatchSize, objectSize, gammas,
		output, outputBlobs[0]->GetDataSize());

	if(!isZeroFreeTerm) {
		MathEngine().AddVectorToMatrixRows(1, output, output, fullBatchSize, objectSize, betas);
	}
}

// Calculates the average over the batch and the batch weight
void CBatchNormalizationLayer::calculateAverage()
{
	int fullBatchSize;
	int objectSize;
	getFullBatchAndObjectSize(fullBatchSize, objectSize);

	CFloatHandle averageData = internalParams->GetObjectData( IPN_Average );
	CConstFloatHandle input = inputBlobs[0]->GetData();

	MathEngine().SumMatrixRows(1, averageData, input, fullBatchSize, objectSize);
	MathEngine().VectorNegMultiply(averageData, averageData, objectSize, fullBatchInv->GetData());
}

void CBatchNormalizationLayer::calculateVariance()
{
	int fullBatchSize;
	int objectSize;
	getFullBatchAndObjectSize(fullBatchSize, objectSize);

	CConstFloatHandle averageData = internalParams->GetObjectData( IPN_Average );
	CFloatHandle varianceData = internalParams->GetObjectData( IPN_Variance );
	CFloatHandle invSqrtVarianceData = internalParams->GetObjectData( IPN_InvSqrtVariance );
	CConstFloatHandle input = inputBlobs[0]->GetData();

	CFloatHandleStackVar temp(MathEngine(), inputBlobs[0]->GetDataSize());

	MathEngine().AddVectorToMatrixRows(1, input, temp, fullBatchSize, objectSize, averageData);
	MathEngine().VectorEltwiseMultiply(temp, temp, temp, temp.Size());
	MathEngine().SumMatrixRows(1, varianceData, temp, fullBatchSize, objectSize);

	// Normalize the variance and calculate the inverse to the standard deviation
	MathEngine().VectorMultiply(varianceData, varianceData, objectSize, fullBatchInv->GetData());
	MathEngine().VectorAddValue(varianceData, invSqrtVarianceData, objectSize, varianceEpsilon->GetData());
	MathEngine().VectorInv(invSqrtVarianceData, invSqrtVarianceData, objectSize);
	MathEngine().VectorSqrt(invSqrtVarianceData, invSqrtVarianceData, objectSize);
}

void CBatchNormalizationLayer::calculateNormalized()
{
	int fullBatchSize;
	int objectSize;
	getFullBatchAndObjectSize(fullBatchSize, objectSize);

	CConstFloatHandle averageData = internalParams->GetObjectData( IPN_Average );
	CConstFloatHandle invSqrtVarianceData = internalParams->GetObjectData( IPN_InvSqrtVariance );
	CConstFloatHandle input = inputBlobs[0]->GetData();

	// The normalized input data
	CFloatHandle normalizedData = normalized->GetData();
	MathEngine().AddVectorToMatrixRows(1, input, normalizedData, fullBatchSize, objectSize, averageData);
	MathEngine().MultiplyMatrixByDiagMatrix(normalizedData, fullBatchSize, objectSize, invSqrtVarianceData,
		normalizedData, normalized->GetDataSize());
}

// Updates the final parameters
void CBatchNormalizationLayer::updateSlowParams(bool isInit)
{
	int fullBatchSize;
	int objectSize;
	getFullBatchAndObjectSize(fullBatchSize, objectSize);

	CConstFloatHandle average = internalParams->GetObjectData( IPN_Average );
	CConstFloatHandle variance = internalParams->GetObjectData( IPN_Variance );
	CFloatHandle slowAverage = internalParams->GetObjectData( IPN_SlowAverage );
	CFloatHandle slowVariance = internalParams->GetObjectData( IPN_SlowVariance );

	if(isInit) {
		// Set the initial values for the average and the batch variance
		MathEngine().VectorFill(slowAverage, 0.f, objectSize);
		MathEngine().VectorFill(slowVariance, 1.f, objectSize);
	}

	// Average the variance and average values over the batches
	MathEngine().VectorMultiply(slowAverage, slowAverage, objectSize, residual->GetData());
	MathEngine().VectorMultiplyAndSub(slowAverage, average, slowAverage, objectSize, slowConvergenceRate->GetData());
	MathEngine().VectorMultiply(slowVariance, slowVariance, objectSize, residual->GetData());
	MathEngine().VectorMultiplyAndAdd(slowVariance, variance, slowVariance, objectSize, varianceMult->GetData());

	isFinalParamDirty = true;
}

void CBatchNormalizationLayer::updateFinalParams()
{
	if(!isFinalParamDirty) {
		return;
	}

	isFinalParamDirty = false;

	int fullBatchSize;
	int objectSize;
	getFullBatchAndObjectSize(fullBatchSize, objectSize);

	CFloatHandle slowAverage = internalParams->GetObjectData( IPN_SlowAverage );
	CFloatHandle slowVariance = internalParams->GetObjectData( IPN_SlowVariance );

	// Calculate the final values
	CConstFloatHandle gamma = paramBlobs[0]->GetObjectData( PN_Gamma );
	CConstFloatHandle beta = paramBlobs[0]->GetObjectData( PN_Beta );
	CFloatHandle finalGamma = finalParams->GetObjectData( PN_Gamma );
	CFloatHandle finalBeta = finalParams->GetObjectData( PN_Beta );

	MathEngine().VectorAddValue(slowVariance, finalBeta, objectSize, varianceEpsilon->GetData());
	MathEngine().VectorSqrt(finalBeta, finalBeta, objectSize);
	MathEngine().VectorEltwiseDivide(gamma, finalBeta, finalGamma, objectSize);

	if(isZeroFreeTerm) {
		MathEngine().VectorFill(finalBeta, 0.0, objectSize);
	} else {
		MathEngine().VectorEltwiseMultiply(finalGamma, slowAverage, finalBeta, objectSize);
		MathEngine().VectorSub(beta, finalBeta, finalBeta, objectSize);
	}
}

// Performs a step in network run with learning
void CBatchNormalizationLayer::runWhenLearning()
{
	bool isInit = checkAndCreateParams();

	calculateAverage();
	calculateVariance();

	calculateNormalized();

	if(isInit) {
		// Set the initial gamma and beta values
		MathEngine().VectorFill( paramBlobs[0]->GetObjectData( PN_Gamma ), 1.f, paramBlobs[0]->GetObjectSize() );
		MathEngine().VectorFill( paramBlobs[0]->GetObjectData( PN_Beta ), 0.f, paramBlobs[0]->GetObjectSize() );
	}

	updateSlowParams(isInit);

	processInput(normalized, paramBlobs[0]);
}

// Performs a step in network run without learning
void CBatchNormalizationLayer::runWhenNoLearning()
{
	updateFinalParams();
	processInput(inputBlobs[0], finalParams);
}

void CBatchNormalizationLayer::BackwardOnce()
{
	if(IsLearningPerformed()) {
		backwardWhenLearning();
	} else {
		backwardWhenNoLearning();
	}
}

// Performs backward propagation when learning
void CBatchNormalizationLayer::backwardWhenLearning()
{
	int fullBatchSize;
	int objectSize;
	getFullBatchAndObjectSize(fullBatchSize, objectSize);

	CFloatHandleStackVar averageDiff(MathEngine(), paramBlobs[0]->GetObjectSize());
	CFloatHandleStackVar averageNormDiff(MathEngine(), paramBlobs[0]->GetObjectSize());
	CFloatHandleStackVar normGamma(MathEngine(), paramBlobs[0]->GetObjectSize());
	CFloatHandleStackVar temp(MathEngine(), outputBlobs[0]->GetDataSize());

	CConstFloatHandle gamma = paramBlobs[0]->GetObjectData( PN_Gamma );
	CConstFloatHandle invSqrtVariance = internalParams->GetObjectData( IPN_InvSqrtVariance );
	CConstFloatHandle normalizedData = normalized->GetData();

	MathEngine().VectorEltwiseMultiply(gamma, invSqrtVariance, normGamma, objectSize);

	CConstFloatHandle outputDiff = outputDiffBlobs[0]->GetData();

	MathEngine().SumMatrixRows(1, averageDiff, outputDiff, fullBatchSize, objectSize);
	MathEngine().VectorEltwiseMultiply(outputDiff, normalizedData, temp, temp.Size());
	MathEngine().SumMatrixRows(1, averageNormDiff, temp, fullBatchSize, objectSize);
	MathEngine().VectorNegMultiply(averageDiff, averageDiff, objectSize, fullBatchInv->GetData());
	MathEngine().VectorMultiply(averageNormDiff, averageNormDiff, objectSize, fullBatchInv->GetData());

	// Calculate inputDiff
	CFloatHandle inputDiff = inputDiffBlobs[0]->GetData();

	MathEngine().AddVectorToMatrixRows(1, outputDiff, inputDiff, fullBatchSize, objectSize, averageDiff);
	MathEngine().MultiplyMatrixByDiagMatrix(normalizedData, fullBatchSize, objectSize, averageNormDiff,
		temp, temp.Size());
	MathEngine().VectorSub(inputDiff, temp, inputDiff, temp.Size());
	MathEngine().MultiplyMatrixByDiagMatrix(inputDiff, fullBatchSize, objectSize, normGamma,
		inputDiff, inputDiffBlobs[0]->GetDataSize());
}

// Performs backward propagation when not learning
void CBatchNormalizationLayer::backwardWhenNoLearning()
{
	updateFinalParams();

	CConstFloatHandle outputDiff = outputDiffBlobs[0]->GetData();
	CFloatHandle inputDiff = inputDiffBlobs[0]->GetData();

	CConstFloatHandle gammas = finalParams->GetObjectData( PN_Gamma );

	int fullBatchSize;
	int objectSize;
	getFullBatchAndObjectSize(fullBatchSize, objectSize);

	MathEngine().MultiplyMatrixByDiagMatrix(outputDiff, fullBatchSize, objectSize, gammas,
		inputDiff, inputDiffBlobs[0]->GetDataSize());
}

void CBatchNormalizationLayer::LearnOnce()
{
	// No regularization
	int fullBatchSize;
	int objectSize;
	getFullBatchAndObjectSize(fullBatchSize, objectSize);

	CFloatHandle gammaDiff = paramDiffBlobs[0]->GetObjectData( 0 );
	CFloatHandle betaDiff = paramDiffBlobs[0]->GetObjectData( 1 );
	CFloatHandleStackVar temp(MathEngine(), outputDiffBlobs[0]->GetDataSize());

	CConstFloatHandle outputDiff = outputDiffBlobs[0]->GetData();
	CConstFloatHandle normalizedData = normalized->GetData();

	if(!isZeroFreeTerm) {
		MathEngine().SumMatrixRowsAdd(1, betaDiff, outputDiff, fullBatchSize, objectSize);
	}
	MathEngine().VectorEltwiseMultiply(outputDiff, normalizedData, temp, temp.Size());
	MathEngine().SumMatrixRowsAdd(1, gammaDiff, temp, fullBatchSize, objectSize);

	isFinalParamDirty = true;
}

void CBatchNormalizationLayer::SetFinalParams(const CPtr<CDnnBlob>& _params)
{
	if(finalParams != 0) {
		NeoAssert(finalParams->GetObjectCount() == _params->GetObjectCount());
		NeoAssert(finalParams->GetHeight() == _params->GetHeight());
		NeoAssert(finalParams->GetWidth() == _params->GetWidth());
		NeoAssert(finalParams->GetDepth() == _params->GetDepth());
		NeoAssert(finalParams->GetChannelsCount() == _params->GetChannelsCount());

		finalParams->CopyFrom(_params);
	} else {
		finalParams = _params->GetCopy();
	}
	isFinalParamDirty = false;
}

static const int BatchNormalizationLayerVersion = 2000;

void CBatchNormalizationLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( BatchNormalizationLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		updateFinalParams();
		archive << isChannelBased;
		archive << GetSlowConvergenceRate();
		SerializeBlob( MathEngine(), archive, finalParams );
		SerializeBlob( MathEngine(), archive, internalParams );
		archive << isZeroFreeTerm;
		archive << useFinalParamsForInitialization;
	} else if( archive.IsLoading() ) {
		archive >> isChannelBased;
		float tempFloat;
		archive >> tempFloat;
		SetSlowConvergenceRate(tempFloat);
		SerializeBlob( MathEngine(), archive, finalParams );
		SerializeBlob( MathEngine(), archive, internalParams );
		archive >> isZeroFreeTerm;
		archive >> useFinalParamsForInitialization;
		isFinalParamDirty = false;
	} else {
		NeoAssert( false );
	}
}

CLayerWrapper<CBatchNormalizationLayer> BatchNormalization(
	bool isChannelBased, bool isZeroFreeTerm, float slowConvergenceRate )
{
	return CLayerWrapper<CBatchNormalizationLayer>( "BatchNormalization", [=]( CBatchNormalizationLayer* result ) {
		result->SetChannelBased( isChannelBased );
		result->SetZeroFreeTerm( isZeroFreeTerm );
		result->SetSlowConvergenceRate( slowConvergenceRate );
	} );
}

} // namespace NeoML
