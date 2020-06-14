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

#include <NeoML/Dnn/Layers/FocalLossLayer.h>

//////////////////////////////////////////////////////////////////////////

static const float MinProbValue = 1e-6f;
static const float MaxProbValue = 1.0f;

//////////////////////////////////////////////////////////////////////////

namespace NeoML {

const float CFocalLossLayer::DefaultFocalForceValue = 2.0f;

static const int FocalLossLayerVersion = 2000;

void CFocalLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( FocalLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		float value = focalForce->GetData().GetValue();
		archive.Serialize( value );
	} else if( archive.IsLoading() ) {
		float value;
		archive.Serialize( value );
		focalForce->GetData().SetValue( value );
	} else {
		NeoAssert( false );
	}
}

void CFocalLossLayer::SetFocalForce( float value )
{
	NeoAssert( value > 0.0f );
	focalForce->GetData().SetValue( value );
}

CFocalLossLayer::CFocalLossLayer( IMathEngine& mathEngine ) :
	CLossLayer( mathEngine, "FmlCnnFocalLossLayer" ),
	focalForce( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	minusOne( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	minProbValue( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	maxProbValue( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) )
{
	SetFocalForce( DefaultFocalForceValue );
	minusOne->GetData().SetValue( -1 );
	minProbValue->GetData().SetValue( MinProbValue );
	maxProbValue->GetData().SetValue( MaxProbValue );
}

void CFocalLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize,
	CConstFloatHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient )
{
	NeoAssert( vectorSize == labelSize );
	NeoAssert( labelSize >= 2 );

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
	MathEngine().VectorPower( focalForce->GetData().GetValue(), remainderVector.GetHandle(), tempMatrixHandle.GetHandle(), batchSize );
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
		minProbValue->GetData(), maxProbValue->GetData() );
	MathEngine().VectorInv( correctClassProbabilityPerBatchHandle, inversedProbailitiesHandle, batchSize );

	// diffPart: (1 - P_t )^focalForce / P_t
	CFloatHandle diffPart = inversedProbailitiesHandle;
	MathEngine().VectorEltwiseMultiply( tempMatrixHandle, inversedProbailitiesHandle,
		diffPart, batchSize );

	// tempMatrix: (1 - P_t )^(focalForce - 1)
	MathEngine().VectorPower( focalForce->GetData().GetValue() - 1, remainderVectorHandle,
		tempMatrixHandle, batchSize );
	// batchEntropy: - (1 - P_t )^(focalForce - 1) log(P_t)
	MathEngine().VectorEltwiseMultiply( tempMatrixHandle, entropyPerBatchHandle,
		entropyPerBatchHandle, batchSize );
	// diffPart: (1 - P_t )^focalForce / P_t - focalForce(1 - P_t )^(focalForce - 1) log(P_t)
	MathEngine().VectorMultiplyAndAdd( diffPart, entropyPerBatchHandle,
		diffPart, batchSize, focalForce->GetData() );

	// diffPart: - (1 - P_t )^focalForce / P_t + focalForce(1 - P_t )^(focalForce - 1) log(P_t)
	MathEngine().VectorMultiply( diffPart, diffPart, batchSize, minusOne->GetData() );
	MathEngine().MultiplyDiagMatrixByMatrix( diffPart, batchSize, label, labelSize, lossGradient, dataSize );
}

} // namespace NeoML
