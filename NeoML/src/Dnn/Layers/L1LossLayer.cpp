/* Copyright © 2017-2021 ABBYY Production LLC

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
// CL1LossLayer

void CL1LossLayer::Reshape()
{
	CLossLayer::Reshape();
	CheckArchitecture( inputDescs[1].GetDataType() == CT_Float, GetName(), "labels must be CT_Float" );
	CheckArchitecture( inputDescs[0].ObjectSize() == inputDescs[1].ObjectSize(), GetName(),
		"the labels dimensions should be equal to the first input dimensions" );
}

void CL1LossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize,
	CConstFloatHandle label, int /* labelSize */, CFloatHandle lossValue, CFloatHandle lossGradient )
{
	int totalSize = batchSize * vectorSize;

	CFloatHandleStackVar temp( MathEngine(), totalSize );

	MathEngine().VectorSub( data, label, temp, totalSize );

	if( !lossGradient.IsNull() ) {
		CFloatHandleStackVar ones( MathEngine(), totalSize );
		MathEngine().VectorFill( ones.GetHandle(), 1.f, totalSize );

		MathEngine().VectorAbsDiff( temp, ones, lossGradient, totalSize );
	}

	MathEngine().VectorAbs( temp, temp, totalSize );
	MathEngine().SumMatrixColumns( lossValue, temp, batchSize, vectorSize );
}

static const int L1LossLayerVersion = 0;

void CL1LossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( L1LossLayerVersion );
	CLossLayer::Serialize( archive );
}

CLayerWrapper<CL1LossLayer> L1Loss( float lossWeight )
{
	return CLayerWrapper<CL1LossLayer>( "L1Loss", [=]( CL1LossLayer* result ) {
		result->SetLossWeight( lossWeight );
	} );
}

} // namespace NeoML
