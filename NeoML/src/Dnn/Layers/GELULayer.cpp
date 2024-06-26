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

#include <NeoML/Dnn/Layers/ActivationLayers.h>

namespace NeoML {

// constants for the precise calculation and its backward
static constexpr float geluSqrt2Inv = 0.70710678f;
static constexpr float geluSqrt2PiInv = 0.39894229f;
// scale for the approximation
static constexpr float geluApproxScale = 1.702f;

static constexpr int geluLayerVersion = 1;

void CGELULayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( geluLayerVersion );
	CBaseLayer::Serialize( archive );

	if( version >= 1 ) {
		archive.SerializeEnum( mode );
	} else {
		mode = CM_SigmoidApproximate;
	}
}

void CGELULayer::SetCalculationMode( TCalculationMode _mode )
{
	mode = _mode;
	ForceReshape();
}

void CGELULayer::Reshape()
{
	CheckInputs();
	NeoAssert( inputDescs.Size() == 1 );

	const CBlobDesc& inputDesc = inputDescs[0];

	outputDescs.SetSize( 1 );
	outputDescs[0] = inputDesc;

	if( IsBackwardPerformed() ) {
		erfMemoization = CDnnBlob::CreateBlob( MathEngine(), CT_Float, inputDesc );
		RegisterRuntimeBlob( erfMemoization );
	}
}

void CGELULayer::RunOnce()
{
	CheckInput1();

	switch( mode ) {
		case TCalculationMode::CM_Precise:
			runPrecise();
			break;
		case TCalculationMode::CM_SigmoidApproximate:
			runFastApproximate();
			break;
		default: 
			NeoAssert( false );
	}
}

void CGELULayer::BackwardOnce()
{
	switch( mode ) {
		case TCalculationMode::CM_Precise:
			backwardPrecise();
			break;
		case TCalculationMode::CM_SigmoidApproximate:
			backwardFastApproximate();
			break;
		default: 
			NeoAssert( false );
	}
}

// x * 0.5( 1 + erf( x / sqrt(2) ) )
void CGELULayer::runPrecise()
{
	CFloatHandle input =  inputBlobs[0]->GetData();
	CFloatHandle output = outputBlobs[0]->GetData();
	const int dataSize = inputBlobs[0]->GetDataSize();

	// output = input / sqrt(2)
	MathEngine().VectorMultiply( input, output, dataSize, geluSqrt2Inv );

	// output = erf( input / sqrt(2) )
	MathEngine().VectorErf( output, output, dataSize );

	// output = 1 + erf( input / sqrt(2) )
	MathEngine().VectorAddValue( output, output, dataSize, 1.f );

	// output = 0.5( 1 + erf( input / sqrt(2) ) )
	MathEngine().VectorMultiply( output, output, dataSize, 0.5f );

	if( IsBackwardPerformed() ) {
		NeoAssert( erfMemoization != nullptr );
		erfMemoization->CopyFrom( outputBlobs[0] );
	}

	// output = input * 0.5( 1 + erf( input / sqrt(2) ) )
	MathEngine().VectorEltwiseMultiply( input, output, output, dataSize );
}

// x * sigmoid(1.702x)
void CGELULayer::runFastApproximate()
{
	CFloatHandle input =  inputBlobs[0]->GetData();
	CFloatHandle output = outputBlobs[0]->GetData();
	const int dataSize = inputBlobs[0]->GetDataSize();

	// output = 1.702 * input
	MathEngine().VectorMultiply( input, output, dataSize, geluApproxScale );

	// output = sigmoid(1.702 * input)
	MathEngine().VectorSigmoid( output, output, dataSize );

	// output = input * sigmoid(1.702 * input)
	MathEngine().VectorEltwiseMultiply( input, output, output, dataSize );
}

// (x * f(x))' = f(x) + xf'(x) = [memoized] + xerf'(x)
// erf'(x) = 2/sqrt(pi) * e^(x^2)
// 
// Adding some scales and shifts, we get
// [0.5( 1 + erf( x / sqrt(2) ) )] + x / sqrt( 2pi ) * e ^ ( -x^2 / 2 ),
// where [...] is saved from the forward pass
void CGELULayer::backwardPrecise()
{
	const int dataSize = inputBlobs[0]->GetDataSize();
	CFloatHandle input =  inputBlobs[0]->GetData();
	CFloatHandle inputDiff = inputDiffBlobs[0]->GetData();

	// inputDiff = input / sqrt(2)
	MathEngine().VectorMultiply( input, inputDiff, dataSize, geluSqrt2Inv );

	// inputDiff = -(input^2 / 2)
	MathEngine().VectorEltwiseNegMultiply( inputDiff, inputDiff, inputDiff, dataSize );

	// inputDiff = e^( -( input^2 / 2) )
	MathEngine().VectorExp( inputDiff, inputDiff, dataSize );

	// inputDiff = e^( -( input^2 / 2) ) / sqrt(2pi)
	MathEngine().VectorMultiply( inputDiff, inputDiff, dataSize, geluSqrt2PiInv );

	// inputDiff *= input
	MathEngine().VectorEltwiseMultiply( inputDiff, input, inputDiff, dataSize );

	// inputDiff = inputDiff + 0.5( 1 + erf( x / sqrt(2) ) )
	MathEngine().VectorAdd( inputDiff, erfMemoization->GetData(), inputDiff, dataSize );

	// inputDiff *= outputDiff
	MathEngine().VectorEltwiseMultiply( inputDiff, outputDiffBlobs[0]->GetData(), inputDiff, dataSize );
}

void CGELULayer::backwardFastApproximate()
{
	const int dataSize = inputBlobs[0]->GetDataSize();
	CFloatHandle input =  inputBlobs[0]->GetData();
	CFloatHandle inputDiff = inputDiffBlobs[0]->GetData();

	CFloatHandleStackVar multipliedInput( MathEngine(), dataSize );

	// multipliedInput = 1.702 * input
	MathEngine().VectorMultiply( input, multipliedInput, dataSize, geluApproxScale );

	// inputDiffs = input * sigmoid_diff(1.702 * input)
	MathEngine().VectorSigmoidDiff( multipliedInput, input, inputDiff, dataSize );

	// inputDiffs = input * sigmoid_diff(1.702 * input) * 1.702
	MathEngine().VectorMultiply( inputDiff, inputDiff, dataSize, geluApproxScale );

	// multipliedInput = sigmoid(1.702 * input)
	MathEngine().VectorSigmoid( multipliedInput, multipliedInput, dataSize );

	// inputDiff = input * sigmoid_diff(1.702 * input) * 1.702 + sigmoid(1.702 * input)
	MathEngine().VectorAdd( inputDiff, multipliedInput, inputDiff, dataSize );

	// inputDiff *= outputDiff
	 MathEngine().VectorEltwiseMultiply( inputDiff, outputDiffBlobs[0]->GetData(), inputDiff, dataSize );
}

CLayerWrapper<CGELULayer> Gelu()
{
	return CLayerWrapper<CGELULayer>( "Gelu" );
}

} // namespace NeoML
