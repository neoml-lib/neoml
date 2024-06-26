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

#include <NeoML/Dnn/Layers/FocalLossLayer.h>

namespace NeoML {

// The handle for acceptable minimum and maximum probability values (so that separation can be performed correctly)
static const float focalLossMinProbValue = 1e-6f;
static const float focalLossMaxProbValue = 1.0f;

const float CFocalLossLayer::DefaultFocalForceValue = 2.0f;

static const int focalLossLayerVersion = 2000;

void CFocalLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( focalLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );

	archive.Serialize( focalForce );
}

void CFocalLossLayer::SetFocalForce( float value )
{
	NeoAssert( value > 0.0f );
	focalForce = ( value );
}

CFocalLossLayer::CFocalLossLayer( IMathEngine& mathEngine ) :
	CLossLayer( mathEngine, "FmlCnnFocalLossLayer" ),
	focalForce( DefaultFocalForceValue )
{}

void CFocalLossLayer::Reshape()
{
	CLossLayer::Reshape();
	CheckLayerArchitecture( inputDescs[1].GetDataType() == CT_Float, "labels must be CT_Float" );
	CheckLayerArchitecture( inputDescs[0].ObjectSize() == inputDescs[1].ObjectSize(),
		"the labels dimensions should be equal to the first input dimensions" );
	CheckLayerArchitecture( inputDescs[0].ObjectSize() >= 2,
		"FocalLoss layer works only with multi-class classification" );
}

void CFocalLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize,
	CConstFloatHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient )
{
	const int dataSize = vectorSize * batchSize;
	CFloatHandleVar tempMatrixHandle( MathEngine(), dataSize );
	// tempMatrix: P_t * y_t
	MathEngine().VectorEltwiseMultiply( data, label, tempMatrixHandle.GetHandle(), dataSize );

	// correctClassProbabilityPerBatch: P_t * y_t
	CFloatHandleVar correctClassProbabilityPerBatch( MathEngine(), batchSize );
	MathEngine().SumMatrixColumns( correctClassProbabilityPerBatch.GetHandle(), tempMatrixHandle.GetHandle(),
		batchSize, labelSize );

	// tempMatrix: (1 - P_t) * y_t
	// remainderVector: (1 - P_t) * y_t
	MathEngine().VectorFill( tempMatrixHandle.GetHandle(), 1.0, batchSize * vectorSize );
	MathEngine().VectorSub( tempMatrixHandle.GetHandle(), data, tempMatrixHandle.GetHandle(), dataSize );
	MathEngine().VectorEltwiseMultiply( tempMatrixHandle.GetHandle(), label, tempMatrixHandle.GetHandle(), dataSize );
	CFloatHandleVar remainderVector( MathEngine(), batchSize );
	MathEngine().SumMatrixColumns( remainderVector.GetHandle(), tempMatrixHandle.GetHandle(), batchSize, labelSize );

	// batchEntropy: -log(P_t)
	CFloatHandleVar entropyPerBatch( MathEngine(), batchSize );
	MathEngine().VectorNegLog( correctClassProbabilityPerBatch.GetHandle(), entropyPerBatch.GetHandle(), batchSize );

	// Loss
	// tempMatrix: (1-P_t)^focalForce
	MathEngine().VectorPower( focalForce, remainderVector.GetHandle(), tempMatrixHandle.GetHandle(), batchSize );
	MathEngine().VectorEltwiseMultiply( tempMatrixHandle.GetHandle(), entropyPerBatch.GetHandle(), lossValue, batchSize );

	// Gradient
	if( !lossGradient.IsNull() ) {
		calculateGradient( correctClassProbabilityPerBatch.GetHandle(), batchSize, labelSize, remainderVector.GetHandle(),
			entropyPerBatch.GetHandle(), tempMatrixHandle.GetHandle(), label, lossGradient );
	}
}

void CFocalLossLayer::calculateGradient( CFloatHandle correctClassProbabilityPerBatchHandle, int batchSize, int labelSize,
	CFloatHandle remainderVectorHandle, CFloatHandle entropyPerBatchHandle, CFloatHandle tempMatrixHandle, CConstFloatHandle label,
	CFloatHandle lossGradient )
{
	int dataSize = labelSize * batchSize;
	// inversedProbailitiesHandle: 1 / P_t
	CFloatHandle inversedProbailitiesHandle = correctClassProbabilityPerBatchHandle;
	MathEngine().VectorMinMax( correctClassProbabilityPerBatchHandle, correctClassProbabilityPerBatchHandle, batchSize,
		focalLossMinProbValue, focalLossMaxProbValue );
	MathEngine().VectorInv( correctClassProbabilityPerBatchHandle, inversedProbailitiesHandle, batchSize );

	// diffPart: (1 - P_t )^focalForce / P_t
	CFloatHandle diffPart = inversedProbailitiesHandle;
	MathEngine().VectorEltwiseMultiply( tempMatrixHandle, inversedProbailitiesHandle,
		diffPart, batchSize );

	// tempMatrix: (1 - P_t )^(focalForce - 1)
	MathEngine().VectorPower( focalForce - 1, remainderVectorHandle,
		tempMatrixHandle, batchSize );
	// batchEntropy: - (1 - P_t )^(focalForce - 1) log(P_t)
	MathEngine().VectorEltwiseMultiply( tempMatrixHandle, entropyPerBatchHandle,
		entropyPerBatchHandle, batchSize );
	// diffPart: (1 - P_t )^focalForce / P_t - focalForce(1 - P_t )^(focalForce - 1) log(P_t)
	MathEngine().VectorMultiplyAndAdd( diffPart, entropyPerBatchHandle,
		diffPart, batchSize, focalForce );

	// diffPart: - (1 - P_t )^focalForce / P_t + focalForce(1 - P_t )^(focalForce - 1) log(P_t)
	MathEngine().VectorMultiply( diffPart, diffPart, batchSize, -1.f );
	MathEngine().MultiplyDiagMatrixByMatrix( diffPart, batchSize, label, labelSize, lossGradient, dataSize );
}

CLayerWrapper<CFocalLossLayer> FocalLoss( float focalForce, float lossWeight )
{
	return CLayerWrapper<CFocalLossLayer>( "FocalLoss", [=]( CFocalLossLayer* result )
		{
			result->SetFocalForce( focalForce );
			result->SetLossWeight( lossWeight );
		}
	);
}
} // namespace NeoML
