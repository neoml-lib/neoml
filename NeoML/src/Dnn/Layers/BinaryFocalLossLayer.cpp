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
	CFloatHandleStackVar one( mathEngine );
	one.SetValue( 1 );
	mathEngine.VectorAddValue( temp, temp, size, one );
	mathEngine.VectorLog( temp, temp, size );

	// result = max( 0, x )
	CFloatHandleStackVar zero( mathEngine );
	zero.SetValue( 0 );
	mathEngine.VectorReLU( first, result, size, zero );

	// result = max( 0, x ) + log( 1 + e^-|x| )
	mathEngine.VectorAdd( result, temp, result, size );
}

// --------------------------------------------------------------------------------------------------------------------

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
	CheckArchitecture( inputDescs[0].ObjectSize() == inputDescs[1].ObjectSize(), GetName(), "the labels dimensions should be equal to the first input dimensions" );
	CheckArchitecture( inputDescs[1].ObjectSize() == 1, GetName(), "BinaryFocalLoss layer works only with binary-class classification" );
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
	MathEngine().VectorPower( focalForce->GetData().GetValue(), sigmoidVector,
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
	CFloatHandleStackVar minusOne( MathEngine() );
	minusOne.SetValue( -1.f );
	MathEngine().VectorAddValue( sigmoidVector, tempVector, batchSize, minusOne );
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
