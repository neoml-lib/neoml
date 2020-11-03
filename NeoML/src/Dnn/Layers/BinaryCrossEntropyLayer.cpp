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

namespace NeoML {

CBinaryCrossEntropyLossLayer::CBinaryCrossEntropyLossLayer( IMathEngine& mathEngine ) :
	CLossLayer( mathEngine, "CCnnBinaryCrossEntropyLossLayer" ),
	positiveWeightMinusOneValue( 0 )
{
}

void CBinaryCrossEntropyLossLayer::SetPositiveWeight( float value )
{
	positiveWeightMinusOneValue = value - 1;
}

float CBinaryCrossEntropyLossLayer::GetPositiveWeight() const
{
	return positiveWeightMinusOneValue + 1;
}

void CBinaryCrossEntropyLossLayer::Reshape()
{
	CLossLayer::Reshape();
	CheckArchitecture( inputDescs[1].GetDataType() == CT_Float, GetName(), "labels must be CT_Float" );
}

void CBinaryCrossEntropyLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize,
	CConstFloatHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient )
{
	// This layer can only work with a binary classificaion problem
	// Therefore the labels vector can only contain {-1, 1} values
	NeoAssert(vectorSize == 1 && labelSize == 1);

	CFloatHandleStackVar one( MathEngine() );
	one.SetValue( 1.f );
	CFloatHandleStackVar half( MathEngine() );
	half.SetValue( 0.5f );
	CFloatHandleStackVar minusOne( MathEngine() );
	minusOne.SetValue( -1.f );
	CFloatHandleStackVar zero( MathEngine() );
	zero.SetValue( 0.f );
	CFloatHandleStackVar positiveWeightMinusOne( MathEngine() );
	positiveWeightMinusOne.SetValue( positiveWeightMinusOneValue );

	// Convert the target values to [0, 1] range using the binaryLabel = 0.5 * ( label + 1 ) formula
	CFloatHandleStackVar binaryLabel( MathEngine(), batchSize );
	MathEngine().VectorAddValue( label, binaryLabel, batchSize, one );
	MathEngine().VectorMultiply( binaryLabel, binaryLabel, batchSize, half );

	// Notations:
	// x = logits, z = labels, q = pos_weight, l = 1 + (q - 1) * z

	// The original loss function formula:
	// loss =  (1 - z) * x + l * log(1 + exp(-x))

	// The formula to avoid overflow for large exponent power in exp(-x):
	// loss = (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

	// (1-z)*x
	CFloatHandleStackVar temp( MathEngine(), batchSize);
	MathEngine().VectorAddValue( binaryLabel, temp, batchSize, minusOne );
	MathEngine().VectorEltwiseNegMultiply( temp, data, temp, batchSize );

	// l = (1 + (q - 1) * z), 
	CFloatHandleStackVar temp2( MathEngine(), batchSize );
	MathEngine().VectorMultiply( binaryLabel, temp2, batchSize, positiveWeightMinusOne );
	MathEngine().VectorAddValue( temp2, temp2, batchSize, one );

	// max(-x, 0)
	CFloatHandleStackVar temp3( MathEngine(), batchSize );
	MathEngine().VectorNegMultiply( data, temp3, batchSize, one );
	MathEngine().VectorReLU( temp3, temp3, batchSize, zero );

	// log( 1 + e^-|x|)
	CFloatHandleStackVar temp4( MathEngine(), batchSize );
	MathEngine().VectorAbs( data, temp4, batchSize );
	MathEngine().VectorNegMultiply( temp4, temp4, batchSize, one );
	MathEngine().VectorExp( temp4, temp4, batchSize );
	MathEngine().VectorAddValue( temp4, temp4, batchSize, one );
	MathEngine().VectorLog( temp4, temp4, batchSize );

	// l * (log(1 + exp(-abs(x))) + max(-x, 0))
	MathEngine().VectorAdd( temp3, temp4, lossValue, batchSize );
	MathEngine().VectorEltwiseMultiply( lossValue, temp2, lossValue, batchSize );

	// The loss
	MathEngine().VectorAdd( lossValue, temp, lossValue, batchSize );

	if( !lossGradient.IsNull() ) {
		// loss' = (1-z) - l / ( 1+exp(x) ) = (1-z) - l * sigmoid(-x) 

		// (z-1)
		CFloatHandleStackVar temp5( MathEngine(), batchSize );
		MathEngine().VectorAddValue( binaryLabel, temp5, batchSize, minusOne );

		// -x
		CFloatHandleStackVar temp6( MathEngine(), batchSize );
		MathEngine().VectorNegMultiply( data, temp6, batchSize, one );

		// sigmoid(-x)
		calculateStableSigmoid( temp6, temp6, batchSize );
		//MathEngine().VectorSigmoid( temp6, temp6, batchSize );

		// l * sigmoid(-x)
		MathEngine().VectorEltwiseMultiply( temp6, temp2, temp6, batchSize );

		// (z-1) + l * sigmoid(-x)
		MathEngine().VectorAdd( temp5, temp6, lossGradient, batchSize );

		//(1-z) - l * sigmoid(-x)
		MathEngine().VectorNegMultiply( lossGradient, lossGradient, batchSize, one );
	}
}

// Overflow-safe sigmoid calculation
void CBinaryCrossEntropyLossLayer::calculateStableSigmoid( const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize ) const
{
	CFloatHandleStackVar one( MathEngine() );
	one.SetValue( 1.f );
	CFloatHandleStackVar zero( MathEngine() );
	zero.SetValue( 0.f );

	// The sigmoid formula:
	// Sigmoid(x) = 1 / (1 + e^-x )

	// The formula to avoid overflow for large exponent power in exp(-x):
	// Sigmoid(x) = e^(-max(-x, 0) ) / ( 1 + e^-|x| ) 

	// e^(-max(-x, 0) )
	CFloatHandleStackVar temp( MathEngine(), vectorSize );
	MathEngine().VectorNegMultiply( firstHandle, temp, vectorSize, one );
	MathEngine().VectorReLU( temp, temp, vectorSize, zero );
	MathEngine().VectorNegMultiply( temp, temp, vectorSize, one );
	MathEngine().VectorExp( temp, temp, vectorSize );

	// ( 1 + e^-|x| ) 
	CFloatHandleStackVar temp2( MathEngine(), vectorSize );
	MathEngine().VectorAbs( firstHandle, temp2, vectorSize );
	MathEngine().VectorNegMultiply( temp2, temp2, vectorSize, one );
	MathEngine().VectorExp( temp2, temp2, vectorSize );
	MathEngine().VectorAddValue( temp2, temp2, vectorSize, one );

	// The sigmoid
	MathEngine().VectorEltwiseDivide( temp, temp2, resultHandle, vectorSize );
}

static const int BinaryCrossEntropyLossLayerVersion = 2000;

void CBinaryCrossEntropyLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( BinaryCrossEntropyLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );
	
	archive.Serialize( positiveWeightMinusOneValue );
}

CLayerWrapper<CBinaryCrossEntropyLossLayer> BinaryCrossEntropyLoss(
	float positiveWeight, float lossWeight )
{
	return CLayerWrapper<CBinaryCrossEntropyLossLayer>( "BinaryCrossEntropyLoss", [=]( CBinaryCrossEntropyLossLayer* result ) {
		result->SetPositiveWeight( positiveWeight );
		result->SetLossWeight( lossWeight );
	} );
}

} // namespace NeoML
