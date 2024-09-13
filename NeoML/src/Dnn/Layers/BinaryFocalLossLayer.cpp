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

#include <NeoML/Dnn/Layers/BinaryFocalLossLayer.h>

namespace NeoML {

static void softplus( const CConstFloatHandle& first, const CFloatHandle& result, int size )
{
	IMathEngine& mathEngine = *first.GetMathEngine();
	// log( 1 + e^x ) = log( 1 + e^-|x| ) + max( 0, x )
	CFloatHandleStackVar temp( mathEngine, size );

	// temp = e^-|x|
	mathEngine.VectorAbs( first, temp, size );
	mathEngine.VectorNeg( temp, temp, size );
	mathEngine.VectorExp( temp, temp, size );

	// temp = log( 1 + e^-|x| )
	mathEngine.VectorAddValue( temp, temp, size, 1.f );
	mathEngine.VectorLog( temp, temp, size );

	// result = max( 0, x )
	mathEngine.VectorReLU( first, result, size, 0.f );

	// result = max( 0, x ) + log( 1 + e^-|x| )
	mathEngine.VectorAdd( result, temp, result, size );
}

// --------------------------------------------------------------------------------------------------------------------

const float CBinaryFocalLossLayer::DefaultFocalForceValue = 2.0f;

static constexpr int binaryFocalLossLayerVersion = 2000;

void CBinaryFocalLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( binaryFocalLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );

	archive.Serialize( focalForce );
}

void CBinaryFocalLossLayer::SetFocalForce( float value )
{
	NeoAssert( value > 0.0f );
	focalForce = value;
}

void CBinaryFocalLossLayer::Reshape()
{
	CLossLayer::Reshape();
	CheckLayerArchitecture( inputDescs[1].GetDataType() == CT_Float, "labels must be CT_Float" );
	CheckLayerArchitecture( inputDescs[0].ObjectSize() == inputDescs[1].ObjectSize(),
		"the labels dimensions should be equal to the first input dimensions" );
	CheckLayerArchitecture( inputDescs[1].ObjectSize() == 1,
		"BinaryFocalLoss layer works only with binary-class classification" );
}

void CBinaryFocalLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data,
	int /* vectorSize */, CConstFloatHandle label, int /* labelSize */, CFloatHandle lossValue, CFloatHandle lossGradient )
{
	CFloatHandleStackVar tempVector(MathEngine(), batchSize);
	CFloatHandleStackVar sigmoidVector(MathEngine(), batchSize);
	CFloatHandleStackVar sigmoidPowerFocal(MathEngine(), batchSize);

	// tempVector = -y * r
	MathEngine().VectorEltwiseNegMultiply( label, data, tempVector, batchSize );
	// sigmoidVector = sigma(-y*r)
	MathEngine().VectorSigmoid( tempVector, sigmoidVector, batchSize );
	MathEngine().VectorPower( focalForce, sigmoidVector,
		sigmoidPowerFocal, batchSize ); 
	// entropyValues = log(1 + e^(-y*r))
	CFloatHandle entropyValues = tempVector;
	softplus( tempVector, entropyValues, batchSize );
	// loss = sigma(-y*r)^focalForce * log(1 + e^(-y*r))
	MathEngine().VectorEltwiseMultiply( sigmoidPowerFocal, entropyValues, lossValue, batchSize );
	if( !lossGradient.IsNull() ) {
		calculateGradient( entropyValues, sigmoidVector, sigmoidPowerFocal, label,
			batchSize, lossGradient );
	}
}

void CBinaryFocalLossLayer::calculateGradient( CFloatHandle entropyValues,
	CFloatHandle sigmoidVector, CFloatHandle sigmoidPowerFocal, CConstFloatHandle labels,
	int batchSize, CFloatHandle lossGradient )
{
	NeoAssert( !lossGradient.IsNull() );
	CFloatHandleStackVar tempVector( MathEngine(), batchSize );
	// tempVector = sigma(-y*r) - 1
	MathEngine().VectorAddValue( sigmoidVector, tempVector, batchSize, -1.f );
	// tempVector = (sigma(-y*r) - 1)*log(1+e^(-y*r))^M
	MathEngine().VectorEltwiseMultiply( tempVector, entropyValues, tempVector, batchSize );
	// tempVector = focalForce*(sigma(-y*r) - 1)*log(1+e^(-y*r))
	MathEngine().VectorMultiply( tempVector, tempVector, batchSize, focalForce );
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

} // namespace NeoML
