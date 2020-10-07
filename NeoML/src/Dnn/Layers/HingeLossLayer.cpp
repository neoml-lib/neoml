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

///////////////////////////////////////////////////////////////////////////////////
// CHingeLossLayer
void CHingeLossLayer::BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize,
	CConstFloatHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient)
{
	NeoAssert(vectorSize == labelSize);

	int totalSize = batchSize * vectorSize;

	CFloatHandleStackVar temp( MathEngine(), totalSize );

	MathEngine().VectorEltwiseMultiply(data, label, temp, totalSize);
	if(!lossGradient.IsNull()) {
		MathEngine().VectorHingeDiff(temp, label, lossGradient, totalSize);
	}
	MathEngine().VectorHinge(temp, temp, totalSize);
	MathEngine().SumMatrixColumns(lossValue, temp, batchSize, vectorSize);
}

static const int HingeLossLayerVersion = 2000;

void CHingeLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( HingeLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );
}

CLayerWrapper<CHingeLossLayer> HingeLoss( float lossWeight )
{
	return CLayerWrapper<CHingeLossLayer>( "HingeLoss", [=]( CHingeLossLayer* result ) {
		result->SetLossWeight( lossWeight );
	} );
}

///////////////////////////////////////////////////////////////////////////////////
// CSquaredHingeLossLayer
void CSquaredHingeLossLayer::BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize,
	CConstFloatHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient)
{
	NeoAssert(vectorSize == labelSize);

	int totalSize = batchSize * vectorSize;

	CFloatHandleStackVar temp( MathEngine(), totalSize );

	MathEngine().VectorEltwiseMultiply(data, label, temp, totalSize);
	if(!lossGradient.IsNull()) {
		MathEngine().VectorSquaredHingeDiff(temp, label, lossGradient, totalSize);
	}
	MathEngine().VectorSquaredHinge(temp, temp, totalSize);
	MathEngine().SumMatrixColumns(lossValue, temp, batchSize, vectorSize);
}

static const int SquaredHingeLossLayerVersion = 2000;

void CSquaredHingeLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( SquaredHingeLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );
}

CLayerWrapper<CSquaredHingeLossLayer> SquaredHingeLoss( float lossWeight )
{
	return CLayerWrapper<CSquaredHingeLossLayer>( "SquaredHingeLoss", [=]( CSquaredHingeLossLayer* result ) {
		result->SetLossWeight( lossWeight );
	} );
}

} // namespace NeoML
