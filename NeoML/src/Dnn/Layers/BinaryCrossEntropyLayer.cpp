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

	// reduced memory usage for calculation
	CFloatHandleStackVar temp( MathEngine(), batchSize * 3 );
	// Convert the target values to [0, 1] range using the binaryLabel = 0.5 * ( label + 1 ) formula
	CFloatHandle binaryLabel = temp.GetHandle();
	MathEngine().VectorAddValue( label, binaryLabel, batchSize, 1.f );
	MathEngine().VectorMultiply( binaryLabel, binaryLabel, batchSize, 0.5f );

	// Notations:
	// x = logits, z = labels, q = pos_weight, lCoef = 1 + (q - 1) * z

	// The original loss function formula:
	// loss = (1 - z) * x + lCoef * log(1 + exp(-x))

	// The formula to avoid overflow for large exponent power in exp(-x):
	// loss = (1 - z) * x + lCoef * (log(1 + exp(-abs(x))) + max(-x, 0))

	CFloatHandle lCoef = temp.GetHandle() + batchSize;
	CFloatHandle xValue = temp.GetHandle() + batchSize * 2;
	CFloatHandle logValue = lossValue; // reduced memory usage for calculation

	// log( 1 + exp(-abs(x)) )
	MathEngine().VectorAbs( data, logValue, batchSize );
	MathEngine().VectorNegMultiply( logValue, logValue, batchSize, 1.f );
	MathEngine().VectorExp( logValue, logValue, batchSize );
	MathEngine().VectorAddValue( logValue, logValue, batchSize, 1.f );
	MathEngine().VectorLog( logValue, logValue, batchSize );

	// max(-x, 0)
	MathEngine().VectorNegMultiply( data, xValue, batchSize, 1.f );
	MathEngine().VectorReLU( xValue, xValue, batchSize, 0.f );
	// lossValue = log( 1 + exp(-abs(x)) ) + max(-x, 0)
	MathEngine().VectorAdd( xValue, logValue, lossValue, batchSize );

	// lCoef = (1 + (q - 1) * z)
	MathEngine().VectorMultiply( binaryLabel, lCoef, batchSize, positiveWeightMinusOne );
	MathEngine().VectorAddValue( lCoef, lCoef, batchSize, 1.f );
	// lCoef * lossValue
	MathEngine().VectorEltwiseMultiply( lossValue, lCoef, lossValue, batchSize );

	// The total loss
	{
		// binaryLabel = (1 - z)
		MathEngine().VectorSub( 1.f, binaryLabel, binaryLabel, batchSize );
		// lossValue += (1 - z) * x
		MathEngine().VectorEltwiseMultiply( binaryLabel, data, xValue, batchSize );
		MathEngine().VectorAdd( lossValue, xValue, lossValue, batchSize );
	}

	if( !lossGradient.IsNull() ) {
		// loss' = (1 - z) - lCoef / ( 1 + exp(x) ) = (1 - z) - lCoef * sigmoid(-x) 

		// -x
		MathEngine().VectorNegMultiply( data, lossGradient, batchSize, 1.f );
		// sigmoid(-x)
		calculateStableSigmoid( lossGradient, xValue, batchSize );
		// lCoef * sigmoid(-x)
		MathEngine().VectorEltwiseMultiply( xValue, lCoef, lossGradient, batchSize );
		// (1 - z) - lCoef * sigmoid(-x)
		MathEngine().VectorSub( binaryLabel, lossGradient, lossGradient, batchSize );
	}
}

// Overflow-safe sigmoid calculation
void CBinaryCrossEntropyLossLayer::calculateStableSigmoid( const CFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize ) const
{
	NeoPresume( !firstHandle.IsNull() );
	NeoPresume( !resultHandle.IsNull() );
	NeoPresume( firstHandle != resultHandle );
	// reduced memory usage for calculation
	CFloatHandle numerator = firstHandle;
	CFloatHandle denominator = resultHandle;

	// The sigmoid formula:
	// Sigmoid(x) = 1 / ( 1 + e^-x )

	// The formula to avoid overflow for large exponent power in exp(-x):
	// Sigmoid(x) = e^( -max(-x, 0) ) / ( 1 + e^-|x| ) 

	// e^( -max(-x, 0) )
	MathEngine().VectorNegMultiply( firstHandle, numerator, vectorSize, 1.f );
	MathEngine().VectorReLU( numerator, numerator, vectorSize, 0.f );
	MathEngine().VectorNegMultiply( numerator, numerator, vectorSize, 1.f );
	MathEngine().VectorExp( numerator, numerator, vectorSize );

	// ( 1 + e^-|x| ) 
	MathEngine().VectorAbs( firstHandle, denominator, vectorSize );
	MathEngine().VectorNegMultiply( denominator, denominator, vectorSize, 1.f );
	MathEngine().VectorExp( denominator, denominator, vectorSize );
	MathEngine().VectorAddValue( denominator, denominator, vectorSize, 1.f );

	// The sigmoid
	MathEngine().VectorEltwiseDivide( numerator, denominator, resultHandle, vectorSize );
}

static const int BinaryCrossEntropyLossLayerVersion = 2000;

void CBinaryCrossEntropyLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( BinaryCrossEntropyLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );

	archive.Serialize( positiveWeightMinusOne );
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
