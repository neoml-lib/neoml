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

#include <NeoML/Dnn/Layers/BinaryFocalLossLayer.h>

namespace NeoML {

const float CBinaryFocalLossLayer::DefaultFocalForceValue = 2.0f;

static const int BinaryFocalLossLayerVersion = 2000;

void CBinaryFocalLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( BinaryFocalLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		float focalForceValue = focalForce->GetData().GetValue();
		archive.Serialize( focalForceValue );
	} else if( archive.IsLoading() ) {
		float focalForceValue;
		archive.Serialize( focalForceValue );
		focalForce->GetData().SetValue( focalForceValue );
	} else {
		NeoAssert( false );
	}
}

void CBinaryFocalLossLayer::SetFocalForce( float value )
{
	NeoAssert( value > 0.0f );
	focalForce->GetData().SetValue( value );
}

CBinaryFocalLossLayer::CBinaryFocalLossLayer( IMathEngine& mathEngine ) :
	CLossLayer( mathEngine, "CCnnBinaryFocalLossLayer" ),
	focalForce( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) )
{
	SetFocalForce( DefaultFocalForceValue );
}

void CBinaryFocalLossLayer::Reshape()
{
	CLossLayer::Reshape();
	CheckArchitecture( inputDescs[1].GetDataType() == CT_Float, GetName(), "labels must be CT_Float" );
}

void CBinaryFocalLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data,
	int vectorSize, CConstFloatHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient )
{
	CheckArchitecture( labelSize == vectorSize, GetName(), "the labels dimensions should be equal to the first input dimensions" );
	CheckArchitecture( vectorSize == 1, GetName(), "BinaryFocalLoss layer works only with binary-class classification" );

	CFloatHandleStackVar tempVector(MathEngine(), batchSize);
	CFloatHandleStackVar sigmoidVector(MathEngine(), batchSize);
	CFloatHandleStackVar onesVector(MathEngine(), batchSize);
	CFloatHandleStackVar sigmoidPowerFocal(MathEngine(), batchSize);

	// tempVector = -y * r
	MathEngine().VectorEltwiseNegMultiply( label, data, tempVector, batchSize );
	// sigmoidVector = sigma(-y*r)
	MathEngine().VectorSigmoid( tempVector, sigmoidVector, batchSize );
	MathEngine().VectorFill( onesVector, 1.0f, batchSize );
	MathEngine().VectorPower( focalForce->GetData().GetValue(), sigmoidVector,
		sigmoidPowerFocal, batchSize ); 
	// tempVector = e^(-y*r)
	MathEngine().VectorExp( tempVector, tempVector, batchSize );
	// tempVector = 1 + e^(-y*r)
	MathEngine().VectorAdd( onesVector, tempVector, tempVector, batchSize );
	// entropyValues = log(1 + e^(-y*r))
	CFloatHandle entropyValues = tempVector;
	MathEngine().VectorLog( tempVector, entropyValues, batchSize );
	// loss = sigma(-y*r)^focalForce * log(1 + e^(-y*r))
	MathEngine().VectorEltwiseMultiply( sigmoidPowerFocal, entropyValues, lossValue, batchSize );
	if( !lossGradient.IsNull() ) {
		calculateGradient( onesVector, entropyValues, sigmoidVector, sigmoidPowerFocal, label,
			batchSize, lossGradient );
	}
}

void CBinaryFocalLossLayer::calculateGradient( CFloatHandle onesVector, CFloatHandle entropyValues,
	CFloatHandle sigmoidVector, CFloatHandle sigmoidPowerFocal, CConstFloatHandle labels,
	int batchSize, CFloatHandle lossGradient )
{
	NeoAssert( !lossGradient.IsNull() );
	CFloatHandle tempVector = onesVector;
	// tempVector = sigma(-y*r) - 1
	MathEngine().VectorSub( sigmoidVector, onesVector, tempVector, batchSize );
	// tempVector = (sigma(-y*r) - 1)*log(1+e^(-y*r))^M
	MathEngine().VectorEltwiseMultiply( tempVector, entropyValues, tempVector, batchSize );
	// tempVector = focalForce*(sigma(-y*r) - 1)*log(1+e^(-y*r))
	MathEngine().VectorMultiply( tempVector, tempVector, batchSize, focalForce->GetData() );
	// tempVector = focalForce*(sigma(-y*r) - 1)*log(1+e^(-y*r)) - sigma(-y*r)
	MathEngine().VectorSub( tempVector, sigmoidVector, tempVector, batchSize ); 
	// tempVector = sigma(-y*r)^focalForce*(focalForce*(sigma(-y*r) - 1)*log(1+e^(-y*r)) - sigma(-y*r))
	MathEngine().VectorEltwiseMultiply( tempVector, sigmoidPowerFocal, tempVector, batchSize ); 
	// grad = y*sigma(-y*r)^focalForce*(focalForce*(sigma(-y*r) - 1)*log(1+e^(-y*r)) - sigma(-y*r))
	MathEngine().VectorEltwiseMultiply( tempVector, labels, lossGradient, batchSize );
}

CLayerWrapper<CBinaryFocalLossLayer> BinaryFocalLoss( float focalForce, float lossWeight )
{
	return CLayerWrapper<CBinaryFocalLossLayer>( "BinaryFocalLoss", [=]( CBinaryFocalLossLayer* result ) {
		result->SetFocalForce( focalForce );
		result->SetLossWeight( lossWeight );
	} );
}

}
