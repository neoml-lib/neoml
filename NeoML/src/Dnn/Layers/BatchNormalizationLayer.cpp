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

#include <NeoML/Dnn/Layers/BatchNormalizationLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// The minimum batch size for correct operation of the algorithm
constexpr int batchNormMinBatchSize = 8;
// A small value to be added to the variance to avoid zero
constexpr float batchNormVarianceEpsilon = 1e-12f;

CBatchNormalizationLayer::CBatchNormalizationLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnBatchNormalizationLayer", true ),
	slowConvergenceRate( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	varianceEpsilon( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	fullBatchInv( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	varianceNorm( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	residual( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	varianceMult( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) )
{
	SetSlowConvergenceRate(0.01f);
	varianceEpsilon->GetData().SetValue( batchNormVarianceEpsilon );
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
bool CBatchNormalizationLayer::checkAndCreateParams( const CFloatHandle& temp )
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
		initializeFromFinalParams( temp );
		useFinalParamsForInitialization = false;
		isInit = false; // All parameters have been set already in initializeFromFinalParams()
	}
	return isInit;
}

// Sets the parameters using the precalculated values from finalParams
void CBatchNormalizationLayer::initializeFromFinalParams( const CFloatHandle& ones )
{
	const int paramSize = finalParams->GetObjectSize();
	CConstFloatHandle finalBeta = finalParams->GetObjectData( PN_Beta );
	CConstFloatHandle finalGamma = finalParams->GetObjectData( PN_Gamma );

	CFloatHandle slowAverage = internalParams->GetObjectData( IPN_SlowAverage );
	CFloatHandle slowVariance = internalParams->GetObjectData( IPN_SlowVariance );
	CFloatHandle gamma = paramBlobs[0]->GetObjectData( PN_Gamma );
	CFloatHandle beta = paramBlobs[0]->GetObjectData(  PN_Beta );

	// Deduce the gamma, beta, slowVar, slowAvg values from the final data
	// We suppose the slow parameters to be equal to the current one because no history is available

	// gamma == slowVar == finalGamma ^ 2
	MathEngine().VectorEltwiseMultiply( finalGamma, finalGamma, gamma, paramSize );
	MathEngine().VectorCopy( slowVariance, gamma, paramSize );

	MathEngine().VectorFill( ones, 1.f, paramSize );

	// beta == slowAvg = finalBeta / ( 1 - finalGamma )
	// No need to consider the IsZeroFreeTerm case because we will anyway be getting beta == slowAvg == 0
	MathEngine().VectorSub( ones, finalGamma, slowAverage, paramSize );
	MathEngine().VectorInv( slowAverage, slowAverage, paramSize );
	MathEngine().VectorEltwiseMultiply( finalBeta, slowAverage, slowAverage, paramSize );
	MathEngine().VectorCopy( beta, slowAverage, paramSize );
}

void CBatchNormalizationLayer::getFullBatchAndObjectSize(int& fullBatchSize, int& objectSize)
{
	fullBatchSize = isChannelBased ?
		( inputDescs[0].BlobSize() / inputDescs[0].Channels() )
		: inputDescs[0].ObjectCount();
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
	CheckLayerArchitecture( inputDescs.Size() == 1, "batch normalization with more than 1 input" );

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
		MathEngine().VectorFill(finalParams->GetObjectData( PN_Gamma ), 1.0, finalParams->GetObjectSize());
		MathEngine().VectorFill(finalParams->GetObjectData( PN_Beta ), 0.0, finalParams->GetObjectSize());
	} else {
		CheckLayerArchitecture( finalParams->GetObjectCount() == PN_Count, "Parameters batch size must be 2" );
		CheckLayerArchitecture( finalParams->GetObjectSize() == objectSize, 
			"Object data size from params must be equal to actual object size" );
	}

	fullBatchInv->GetData().SetValue(1.f / fullBatchSize);
	float varianceNormValue  = (fullBatchSize > 1) ? (float)fullBatchSize / (fullBatchSize - 1) : 0;
	varianceNorm->GetData().SetValue(varianceNormValue);
	residual->GetData().SetValue(1);
	MathEngine().VectorSub(residual->GetData(), slowConvergenceRate->GetData(), residual->GetData(), 1);
	MathEngine().VectorEltwiseMultiply(slowConvergenceRate->GetData(), varianceNorm->GetData(), varianceMult->GetData(), 1);
	
	normalized = nullptr;
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
		CheckLayerArchitecture( fullBatchSize >= batchNormMinBatchSize,
			"in batch normalization fullBatchSize is more than MinBatchSize" );

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

void CBatchNormalizationLayer::calculateVariance( const CFloatHandle& temp )
{
	int fullBatchSize;
	int objectSize;
	getFullBatchAndObjectSize(fullBatchSize, objectSize);

	CConstFloatHandle averageData = internalParams->GetObjectData( IPN_Average );
	CFloatHandle varianceData = internalParams->GetObjectData( IPN_Variance );
	CFloatHandle invSqrtVarianceData = internalParams->GetObjectData( IPN_InvSqrtVariance );
	CConstFloatHandle input = inputBlobs[0]->GetData();

	const int tempSize = inputBlobs[0]->GetDataSize();
	MathEngine().AddVectorToMatrixRows(1, input, temp, fullBatchSize, objectSize, averageData);
	MathEngine().VectorEltwiseMultiply(temp, temp, temp, tempSize);
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
	CFloatHandleStackVar temp( MathEngine(), max( finalParams->GetObjectSize(), inputBlobs[0]->GetDataSize() ) );
	const bool isInit = checkAndCreateParams( temp );
	calculateAverage();
	calculateVariance( temp );
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

	const int tempSize = outputDiffBlobs[0]->GetDataSize();
	CFloatHandleStackVar temp( MathEngine(), tempSize + objectSize );
	CFloatHandle averageDiff = temp + tempSize;
	NeoAssert( paramBlobs[0]->GetObjectSize() == objectSize );

	CConstFloatHandle normalizedData = normalized->GetData();
	CConstFloatHandle outputDiff = outputDiffBlobs[0]->GetData();
	CFloatHandle inputDiff = inputDiffBlobs[0]->GetData();

	// Calculate averageDiff
	MathEngine().SumMatrixRows(1, averageDiff, outputDiff, fullBatchSize, objectSize);
	MathEngine().VectorNegMultiply(averageDiff, averageDiff, objectSize, fullBatchInv->GetData());

	// Calculate temp
	MathEngine().VectorEltwiseMultiply(outputDiff, normalizedData, temp, tempSize);

	// Calculate inputDiff
	MathEngine().AddVectorToMatrixRows(1, outputDiff, inputDiff, fullBatchSize, objectSize, averageDiff);

	// Calculate averageNormDiff
	CFloatHandle averageNormDiff = averageDiff;
	MathEngine().SumMatrixRows(1, averageNormDiff, temp, fullBatchSize, objectSize );
	MathEngine().VectorMultiply(averageNormDiff, averageNormDiff, objectSize, fullBatchInv->GetData());

	// Calculate inputDiff
	MathEngine().MultiplyMatrixByDiagMatrix(normalizedData, fullBatchSize, objectSize, averageNormDiff,
		temp, tempSize);
	MathEngine().VectorSub(inputDiff, temp, inputDiff, tempSize);

	CConstFloatHandle gamma = paramBlobs[0]->GetObjectData( PN_Gamma );
	CConstFloatHandle invSqrtVariance = internalParams->GetObjectData( IPN_InvSqrtVariance );
	CFloatHandle normGamma = averageDiff;

	// Calculate inputDiff
	MathEngine().VectorEltwiseMultiply( gamma, invSqrtVariance, normGamma, objectSize );
	MathEngine().MultiplyMatrixByDiagMatrix( inputDiff, fullBatchSize, objectSize, normGamma,
		inputDiff, inputDiffBlobs[0]->GetDataSize() );
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

constexpr int batchNormalizationLayerVersion = 2000;

void CBatchNormalizationLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( batchNormalizationLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		updateFinalParams();
	}

	archive.Serialize( isChannelBased );
	float temp = GetSlowConvergenceRate();
	archive.Serialize( temp );
	SetSlowConvergenceRate( temp );

	SerializeBlob( MathEngine(), archive, finalParams );
	SerializeBlob( MathEngine(), archive, internalParams );
	archive.Serialize( isZeroFreeTerm );
	archive.Serialize( useFinalParamsForInitialization );
	
	if( archive.IsLoading() ) {
		normalized = nullptr;
		varianceEpsilon->GetData().SetValue( batchNormVarianceEpsilon );
		fullBatchInv->Clear();
		varianceNorm->Clear();
		residual->Clear();
		varianceMult->Clear();
		isFinalParamDirty = false;
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
