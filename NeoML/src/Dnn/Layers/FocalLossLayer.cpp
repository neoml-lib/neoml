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

constexpr float focalLossMinProbValue = 1e-6f;
constexpr float focalLossMaxProbValue = 1.0f;

const float CFocalLossLayer::DefaultFocalForceValue = 2.0f;

constexpr int focalLossLayerVersion = 2000;

void CFocalLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( focalLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
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
	SetFocalForce( CFocalLossLayer::DefaultFocalForceValue );
	minusOne->GetData().SetValue( -1 );
	minProbValue->GetData().SetValue( focalLossMinProbValue );
	maxProbValue->GetData().SetValue( focalLossMaxProbValue );
}

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
	CFloatHandleStackVar temp( MathEngine(), batchSize + max( dataSize, batchSize ) );

	CFloatHandle remainderVector = temp;
	CFloatHandle correctClassProbabilityPerBatch = ( !lossGradient.IsNull() ) ? lossGradient : lossValue;

	{
		CFloatHandle tempMatrix = remainderVector + batchSize;
		// tempMatrix: P_t * y_t
		MathEngine().VectorEltwiseMultiply( data, label, tempMatrix, dataSize );
		// correctClassProbabilityPerBatch: P_t * y_t
		MathEngine().SumMatrixColumns( correctClassProbabilityPerBatch, tempMatrix, batchSize, labelSize );

		// tempMatrix: (1 - P_t) * y_t
		MathEngine().VectorSub( 1.f, data, tempMatrix, dataSize );
		MathEngine().VectorEltwiseMultiply( tempMatrix, label, tempMatrix, dataSize );
		// remainderVector: (1 - P_t) * y_t
		MathEngine().SumMatrixColumns( remainderVector, tempMatrix, batchSize, labelSize );
	}

	CFloatHandle remainderPowered = remainderVector + batchSize;
	CFloatHandle entropyPerBatch = lossValue;

	// batchEntropy: -log(P_t)
	MathEngine().VectorNegLog( correctClassProbabilityPerBatch, entropyPerBatch, batchSize );
	// remainderPowered: (1-P_t)^focalForce
	MathEngine().VectorPower( focalForce->GetData().GetValue(), remainderVector, remainderPowered, batchSize );

	// Gradient
	if( !lossGradient.IsNull() ) {
		calculateGradient( correctClassProbabilityPerBatch, batchSize, labelSize, remainderVector,
			entropyPerBatch, remainderPowered, label, lossGradient );
	}
	// Loss
	MathEngine().VectorEltwiseMultiply( entropyPerBatch, remainderPowered, lossValue, batchSize );
}

void CFocalLossLayer::calculateGradient( CFloatHandle correctClassProbabilityPerBatch, int batchSize, int labelSize,
	CFloatHandle remainderVector, CConstFloatHandle entropyPerBatch, CConstFloatHandle remainderPowered,
	CConstFloatHandle label, CFloatHandle lossGradient )
{
	// inversedProbailities: 1 / P_t
	CFloatHandle inversedProbailities = correctClassProbabilityPerBatch;
	MathEngine().VectorMinMax( correctClassProbabilityPerBatch, correctClassProbabilityPerBatch, batchSize,
		minProbValue->GetData(), maxProbValue->GetData() );
	MathEngine().VectorInv( correctClassProbabilityPerBatch, inversedProbailities, batchSize );

	// diffPart: (1 - P_t )^focalForce / P_t
	CFloatHandle diffPart = inversedProbailities;
	MathEngine().VectorEltwiseMultiply( remainderPowered, inversedProbailities, diffPart, batchSize );

	// remainderVector: (1 - P_t )^(focalForce - 1)
	MathEngine().VectorPower( focalForce->GetData().GetValue() - 1, remainderVector, remainderVector, batchSize );
	// batchEntropy: - (1 - P_t )^(focalForce - 1) log(P_t)
	MathEngine().VectorEltwiseMultiply( remainderVector, entropyPerBatch, remainderVector, batchSize );
	// diffPart: (1 - P_t )^focalForce / P_t - focalForce(1 - P_t )^(focalForce - 1) log(P_t)
	MathEngine().VectorMultiplyAndAdd( diffPart, remainderVector, diffPart, batchSize, focalForce->GetData() );

	// diffPart: - (1 - P_t )^focalForce / P_t + focalForce(1 - P_t )^(focalForce - 1) log(P_t)
	CFloatHandle diffPartTmp = remainderVector;
	MathEngine().VectorMultiply( diffPart, diffPartTmp, batchSize, minusOne->GetData() );
	MathEngine().MultiplyDiagMatrixByMatrix( diffPartTmp, batchSize, label, labelSize, lossGradient, labelSize * batchSize );
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
