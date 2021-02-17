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
// CEuclideanLossLayer

void CEuclideanLossLayer::Reshape()
{
	CLossLayer::Reshape();
	CheckArchitecture( inputDescs[1].GetDataType() == CT_Float, GetName(), "labels must be CT_Float" );
}

void CEuclideanLossLayer::BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize,
	CConstFloatHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient)
{
	CheckArchitecture( labelSize == vectorSize, GetName(), "the labels dimensions should be equal to the first input dimensions" );

	int totalSize = batchSize * vectorSize;

	CFloatHandleStackVar temp( MathEngine(), totalSize );

	MathEngine().VectorSub(data, label, temp, totalSize);

	if(!lossGradient.IsNull()) {
		MathEngine().VectorHuberDerivative(temp, lossGradient, totalSize);
	}
	MathEngine().VectorHuber(temp, temp, totalSize);
	MathEngine().SumMatrixColumns(lossValue, temp, batchSize, vectorSize);
}

static const int EuclideanLossLayerVersion = 2000;

void CEuclideanLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( EuclideanLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );
}

CLayerWrapper<CEuclideanLossLayer> EuclideanLoss( float lossWeight )
{
	return CLayerWrapper<CEuclideanLossLayer>( "EuclideanLoss", [=]( CEuclideanLossLayer* result ) {
		result->SetLossWeight( lossWeight );
	} );
}

}
