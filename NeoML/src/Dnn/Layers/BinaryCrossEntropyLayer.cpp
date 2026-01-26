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

namespace NeoML {

void CBinaryCrossEntropyLossLayer::Reshape()
{
	CLossLayer::Reshape();
	CheckLayerArchitecture( inputDescs[1].GetDataType() == CT_Float, "labels must be CT_Float" );
	CheckLayerArchitecture( inputDescs[0].ObjectSize() == 1 && inputDescs[1].ObjectSize() == 1,
		"BinaryCrossEntropy layer can only work with a binary classificaion problem" );
}

void CBinaryCrossEntropyLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int /*vectorSize*/,
	CConstFloatHandle label, int /*labelSize*/, CFloatHandle lossValue, CFloatHandle lossGradient )
{
	// Therefore the labels vector can only contain {-1, 1} values
	CFloatHandleStackVar one( MathEngine() );
	one.SetValue( 1.f );
	CFloatHandleStackVar half( MathEngine() );
	half.SetValue( 0.5f );
	CFloatHandleStackVar minusOne( MathEngine() );
	minusOne.SetValue( -1.f );
	CFloatHandleStackVar zero( MathEngine() );
	zero.SetValue( 0.f );
	CFloatHandleStackVar positiveWeightMinusOneVar( MathEngine() );
	positiveWeightMinusOneVar.SetValue( positiveWeightMinusOne );

	CFloatHandleStackVar temp( MathEngine(), batchSize * 3 );
	// Convert the target values to [0, 1] range using the binaryLabel = 0.5 * ( label + 1 ) formula
	CFloatHandle binaryLabel = temp.GetHandle();
	MathEngine().VectorAddValue( label, binaryLabel, batchSize, one );
	MathEngine().VectorMultiply( binaryLabel, binaryLabel, batchSize, half );

	// Notations:
	// x = logits, z = labels, q = pos_weight, lCoef = 1 + (q - 1) * z

	// The original loss function formula:
	// loss = (1 - z) * x + lCoef * log(1 + exp(-x))

	// The formula to avoid overflow for large exponent power in exp(-x):
	// loss = (1 - z) * x + lCoef * (log(1 + exp(-abs(x))) + max(-x, 0))

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

		// (z-1)
		CFloatHandleStackVar temp5( MathEngine(), batchSize );
		MathEngine().VectorAddValue( binaryLabel, temp5, batchSize, minusOne );
		// loss' = (1 - z) - lCoef / ( 1 + exp(x) ) = (1 - z) - lCoef * sigmoid(-x) 

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
void CBinaryCrossEntropyLossLayer::calculateStableSigmoid( const CFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize ) const
{
	CFloatHandleStackVar one( MathEngine() );
	one.SetValue( 1.f );
	CFloatHandleStackVar zero( MathEngine() );
	zero.SetValue( 0.f );

	NeoPresume( !firstHandle.IsNull() );
	NeoPresume( !resultHandle.IsNull() );
	NeoPresume( firstHandle != resultHandle );
	// reduced memory usage for calculation
	CFloatHandle numerator = resultHandle;
	CFloatHandle denominator = firstHandle;

	// The sigmoid formula:
	// Sigmoid(x) = 1 / ( 1 + e^-x )

	// The formula to avoid overflow for large exponent power in exp(-x):
	// Sigmoid(x) = e^( -max(-x, 0) ) / ( 1 + e^-|x| ) 

	// e^( -max(-x, 0) )
	MathEngine().VectorNegMultiply( firstHandle, numerator, vectorSize, one );
	MathEngine().VectorReLU( numerator, numerator, vectorSize, zero );
	MathEngine().VectorNegMultiply( numerator, numerator, vectorSize, one );
	MathEngine().VectorExp( numerator, numerator, vectorSize );

	// ( 1 + e^-|x| ) 
	MathEngine().VectorAbs( firstHandle, denominator, vectorSize );
	MathEngine().VectorNegMultiply( denominator, denominator, vectorSize, one );
	MathEngine().VectorExp( denominator, denominator, vectorSize );
	MathEngine().VectorAddValue( denominator, denominator, vectorSize, one );

	// The sigmoid
	MathEngine().VectorEltwiseDivide( numerator, denominator, resultHandle, vectorSize );
}

constexpr int binaryCrossEntropyLossLayerVersion = 2000;

void CBinaryCrossEntropyLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( binaryCrossEntropyLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );

	archive.Serialize( positiveWeightMinusOne );
}

CLayerWrapper<CBinaryCrossEntropyLossLayer> BinaryCrossEntropyLoss( float positiveWeight, float lossWeight )
{
	return CLayerWrapper<CBinaryCrossEntropyLossLayer>( "BinaryCrossEntropyLoss",
		[=]( CBinaryCrossEntropyLossLayer* result ) {
			result->SetPositiveWeight( positiveWeight );
			result->SetLossWeight( lossWeight );
		} );
}

} // namespace NeoML
