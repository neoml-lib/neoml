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
#include <float.h>

namespace NeoML {

void CCrossEntropyLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize,
	CConstFloatHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient )
{
	BatchCalculateLossAndGradient( batchSize, data, vectorSize, label, labelSize,
		lossValue, lossGradient, CFloatHandle{} );
}

void CCrossEntropyLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize,
	CConstFloatHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient, CFloatHandle labelLossGradient )
{
	CheckLayerArchitecture( labelSize == vectorSize,
		"for float labels the dimensions should be equal to the first input dimensions" );
	CheckLayerArchitecture( vectorSize >= 2, "CrossEntropyLoss layer works only with multi-class classification" );

	const int totalSize = batchSize * vectorSize;
	CFloatHandleStackVar activation( MathEngine(), totalSize * 2 );
	CFloatHandle activationEltwiseMul = activation.GetHandle() + totalSize;

	if( isSoftmaxApplied ) {
		MathEngine().MatrixSoftmaxByRows( data, batchSize, vectorSize, activation );
	} else {
		// For computational stability
		const float maxValue = ( 1.f - FLT_EPSILON );
		const float minValue = ( FLT_EPSILON );
		MathEngine().VectorMinMax( data, activation, totalSize, minValue, maxValue );
	}

	if( labelLossGradient.IsNull() ) {
		MathEngine().VectorNegLog( activation, activationEltwiseMul, totalSize );
		MathEngine().VectorEltwiseMultiply( activationEltwiseMul, label, activationEltwiseMul, totalSize );
		MathEngine().SumMatrixColumns( lossValue, activationEltwiseMul, batchSize, vectorSize );
	} else {
		MathEngine().VectorNegLog( activation, labelLossGradient, totalSize );
		MathEngine().VectorEltwiseMultiply( labelLossGradient, label, activationEltwiseMul, totalSize );
		MathEngine().SumMatrixColumns( lossValue, activationEltwiseMul, batchSize, vectorSize );
		MathEngine().SubVectorFromMatrixColumns( labelLossGradient, labelLossGradient, batchSize, vectorSize, lossValue );
	}
	if( lossGradient.IsNull() ) {
		return;
	}

	if( isSoftmaxApplied ) {
		MathEngine().VectorSub( activation, label, activationEltwiseMul, totalSize );
	} else {
		MathEngine().VectorInv( activation, activation, totalSize );
		MathEngine().VectorEltwiseMultiply( activation, label, activation, totalSize );
		MathEngine().VectorFill( activationEltwiseMul, 1.f, totalSize );
		MathEngine().VectorSub( activationEltwiseMul, activation, activationEltwiseMul, totalSize );
	}

	// Put 0 for those elements for which the label sum is 0
	CFloatHandle labelSum = activation;
	MathEngine().SumMatrixColumns( labelSum, label, batchSize, vectorSize );
	MathEngine().MultiplyDiagMatrixByMatrix( labelSum, batchSize, activationEltwiseMul, vectorSize,
		lossGradient, totalSize );
}

void CCrossEntropyLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize,
	CConstIntHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient )
{
	CheckLayerArchitecture( labelSize == 1,
		"for int labels each object in the blob should contain the number of the class" );
	CheckLayerArchitecture( vectorSize >= 2, "CrossEntropyLoss layer works only with multi-class classification" );

	const int totalSize = batchSize * vectorSize;
	CFloatHandleStackVar activationMul( MathEngine(), batchSize + totalSize );
	CFloatHandle activation = activationMul.GetHandle() + batchSize;

	if( isSoftmaxApplied ) {
		MathEngine().MatrixSoftmaxByRows( data, batchSize, vectorSize, activation );
	} else {
		// For computational stability
		const float maxValue = ( 1.f - FLT_EPSILON );
		const float minValue = ( FLT_EPSILON );
		MathEngine().VectorMinMax( data, activation, totalSize, minValue, maxValue );
	}

	MathEngine().VectorFill( activationMul, 0, batchSize );
	MathEngine().AddMatrixElementsToVector( activation, batchSize, vectorSize, label, activationMul, batchSize );

	MathEngine().VectorNegLog( activationMul, lossValue, batchSize );

	if( lossGradient.IsNull() ) {
		return;
	}

	if( isSoftmaxApplied ) {
		// lossGradient = softmax
		MathEngine().VectorFill( activationMul, -1, batchSize );
	} else {
		MathEngine().VectorInv( activation, activation, totalSize );
		MathEngine().VectorMultiply( activation, activation, totalSize, -1.f );
		MathEngine().VectorFill( activationMul, 0, batchSize );
		MathEngine().AddMatrixElementsToVector( activation, batchSize, vectorSize, label, activationMul, batchSize );
		MathEngine().VectorFill( activation, 1, totalSize );
	}
	
	MathEngine().AddVectorToMatrixElements( activation, batchSize, vectorSize, label, activationMul );

	MathEngine().VectorEltwiseNotNegative( label, activationMul, batchSize );
	MathEngine().MultiplyDiagMatrixByMatrix( activationMul, batchSize, activation, vectorSize, lossGradient, totalSize );
}

static constexpr int CrossEntropyLossLayerVersion = 2000;

void CCrossEntropyLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CrossEntropyLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );

	archive.Serialize( isSoftmaxApplied );
}

CLayerWrapper<CCrossEntropyLossLayer> CrossEntropyLoss( bool isSoftmaxApplied, float lossWeight )
{
	return CLayerWrapper<CCrossEntropyLossLayer>( "CrossEntropyLoss", [=]( CCrossEntropyLossLayer* result ) {
		result->SetApplySoftmax( isSoftmaxApplied );
		result->SetLossWeight( lossWeight );
	} );
}

} // namespace NeoML
