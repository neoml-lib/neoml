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

#include <NeoML/Dnn/Layers/MultiHingeLossLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

void CBaseMultiHingeLossLayer::BatchCalculateLossAndGradient(int batchSize,
	CConstFloatHandle data, int vectorSize, CConstFloatHandle label, int labelSize,
	CFloatHandle lossValue, CFloatHandle lossGradient)
{
	NeoAssert(labelSize == vectorSize);

	CFloatHandleStackVar ones( MathEngine(), batchSize * vectorSize );
	MathEngine().VectorFill(ones, 1.f, batchSize * vectorSize);

	// lossValue - positives
	MathEngine().RowMultiplyMatrixByMatrix(data, label, batchSize, vectorSize, lossValue);

	CFloatHandleStackVar temp( MathEngine(), batchSize * vectorSize );
	MathEngine().VectorSub(ones, label, temp, batchSize * vectorSize);
	MathEngine().VectorEltwiseMultiply(data, temp, temp, batchSize * vectorSize);

	CFloatHandleStackVar negatives( MathEngine(), batchSize );
	CIntHandleStackVar negIdx( MathEngine(), lossGradient.IsNull() ? 0 : batchSize );

	if(lossGradient.IsNull()) {
		MathEngine().FindMaxValueInRows(temp, batchSize, vectorSize, negatives, batchSize);
	} else {
		MathEngine().FindMaxValueInRows(temp, batchSize, vectorSize, negatives, negIdx, batchSize);
	}

	MathEngine().VectorSub(lossValue, negatives, lossValue, batchSize);

	if(!lossGradient.IsNull()) {
		CFloatHandle derivative = negatives;
		CalculateEltwiseLossDiff( lossValue, ones, derivative, batchSize );

		CFloatHandle temp2 = ones;
		MathEngine().VectorFill(temp2, 0.f, batchSize * vectorSize);
		MathEngine().AddMatrixElementsToMatrix(temp, batchSize, vectorSize, temp2, negIdx);
		MathEngine().VectorSub(label, temp2, temp2, batchSize * vectorSize);
		MathEngine().MultiplyDiagMatrixByMatrix(derivative, batchSize, temp2, vectorSize,
			lossGradient, batchSize * vectorSize);
	}

	CalculateEltwiseLoss(lossValue, lossValue, batchSize);
}

void CBaseMultiHingeLossLayer::BatchCalculateLossAndGradient(int batchSize,
	CConstFloatHandle data, int vectorSize, CConstIntHandle label, int labelSize,
	CFloatHandle lossValue, CFloatHandle lossGradient)
{
	NeoAssert(labelSize == 1);

	// lossValue - positives
	MathEngine().VectorFill(lossValue, 0, batchSize);
	MathEngine().AddMatrixElementsToVector(data, batchSize, vectorSize, label, lossValue, batchSize);

	CFloatHandleStackVar temp( MathEngine(), batchSize * vectorSize );
	MathEngine().VectorFill(temp, 0, batchSize * vectorSize);
	MathEngine().AddVectorToMatrixElements(temp, batchSize, vectorSize, label, lossValue);
	MathEngine().VectorSub(data, temp, temp, batchSize * vectorSize);

	CFloatHandleStackVar negatives( MathEngine(), batchSize );
	CIntHandleStackVar negIdx( MathEngine(), lossGradient.IsNull() ? 0 : batchSize );

	if(lossGradient.IsNull()) {
		MathEngine().FindMaxValueInRows(temp, batchSize, vectorSize, negatives, batchSize);
	} else {
		MathEngine().FindMaxValueInRows(temp, batchSize, vectorSize, negatives, negIdx, batchSize);
	}

	MathEngine().VectorSub(lossValue, negatives, lossValue, batchSize);

	if(!lossGradient.IsNull()) {
		CFloatHandleStackVar temp2( MathEngine(), batchSize * vectorSize );
		MathEngine().VectorFill(temp2, 1.f, batchSize);
		CFloatHandle derivative = negatives;
		CalculateEltwiseLossDiff( lossValue, temp2, derivative, batchSize );

		MathEngine().VectorFill(temp2, 0.f, batchSize * vectorSize);
		MathEngine().AddMatrixElementsToMatrix(temp, batchSize, vectorSize, temp2, negIdx);
		MathEngine().EnumBinarization(batchSize, label, vectorSize, temp);
		MathEngine().VectorSub(temp, temp2, temp2, batchSize * vectorSize);
		MathEngine().MultiplyDiagMatrixByMatrix(derivative, batchSize, temp2, vectorSize,
			lossGradient, batchSize * vectorSize);
	}

	CalculateEltwiseLoss(lossValue, lossValue, batchSize);
}

static const int BaseMultiHingeLossLayerVersion = 2000;

void CBaseMultiHingeLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( BaseMultiHingeLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );
}

// ====================================================================================================================

void CMultiHingeLossLayer::CalculateEltwiseLoss(const CFloatHandle& first, const CFloatHandle& result, int vectorSize)
{
	MathEngine().VectorHinge(first, result, vectorSize);
}

void CMultiHingeLossLayer::CalculateEltwiseLossDiff(const CFloatHandle& first, const CFloatHandle& second,
	const CFloatHandle& result, int vectorSize)
{
	MathEngine().VectorHingeDiff(first, second, result, vectorSize);
}

static const int MultiHingeLossLayerVersion = 2000;

void CMultiHingeLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MultiHingeLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseMultiHingeLossLayer::Serialize( archive );
}

CLayerWrapper<CMultiHingeLossLayer> MultiHingeLoss( float lossWeight )
{
	return CLayerWrapper<CMultiHingeLossLayer>( "MultiHingeLoss", [=]( CMultiHingeLossLayer* result ) {
		result->SetLossWeight( lossWeight );
	} );
}

// ====================================================================================================================

void CMultiSquaredHingeLossLayer::CalculateEltwiseLoss(const CFloatHandle& first, const CFloatHandle& result,
	int vectorSize)
{
	MathEngine().VectorSquaredHinge(first, result, vectorSize);
}

void CMultiSquaredHingeLossLayer::CalculateEltwiseLossDiff(const CFloatHandle& first, const CFloatHandle& second,
	const CFloatHandle& result, int vectorSize)
{
	MathEngine().VectorSquaredHingeDiff(first, second, result, vectorSize);
}

static const int MultiSquaredHingeLossLayerVersion = 2000;

void CMultiSquaredHingeLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MultiSquaredHingeLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseMultiHingeLossLayer::Serialize( archive );
}

CLayerWrapper<CMultiSquaredHingeLossLayer> MultiSquaredHingeLoss( float lossWeight )
{
	return CLayerWrapper<CMultiSquaredHingeLossLayer>( "MultiSquaredHingeLoss", [=]( CMultiSquaredHingeLossLayer* result ) {
		result->SetLossWeight( lossWeight );
	} );
}

} // namespace NeoML

