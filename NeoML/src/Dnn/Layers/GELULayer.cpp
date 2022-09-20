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

#include <NeoML/Dnn/Layers/GELULayer.h>

namespace NeoML {

// constants for the precise calculation and its backward
static const float GELUOne = 1.0f;
static const float GELUHalf = 0.5f;
static const float GELUSqrt2Inv = 0.70710678f;
static const float GELUSqrt2PiInv = 0.39894229f;

// scale for the approximation
static const float GELUApproximationMultiplier = 1.702f;

static const int CGELULayerVersion = 1;

CGELULayer::CGELULayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CGELULayer", false ),
	oneVar( mathEngine ),
	halfVar( mathEngine ),
	sqrt2InvVar( mathEngine ),
	sqrt2PiInvVar( mathEngine ),
	approxScaleVar( mathEngine )
{
	oneVar.SetValue( GELUOne );
	halfVar.SetValue( GELUHalf );
	sqrt2InvVar.SetValue( GELUSqrt2Inv );
	sqrt2PiInvVar.SetValue( GELUSqrt2PiInv );
	approxScaleVar.SetValue( GELUApproximationMultiplier );
}

void CGELULayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( CGELULayerVersion );
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
	MathEngine().VectorMultiply( input, output, dataSize, sqrt2InvVar );

	// output = erf( input / sqrt(2) )
	MathEngine().VectorErf( output, output, dataSize );

	// output = 1 + erf( input / sqrt(2) )
	MathEngine().VectorAddValue( output, output, dataSize, oneVar );

	// output = 0.5( 1 + erf( input / sqrt(2) ) )
	MathEngine().VectorMultiply( output, output, dataSize, halfVar );

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
	MathEngine().VectorMultiply( input, output, dataSize, approxScaleVar );

	// output = sigmoid(1.702 * input)
	MathEngine().VectorSigmoid( output, output, dataSize );

	// output = input * sigmoid(1.702 * input)
	MathEngine().VectorEltwiseMultiply( input, output, output, dataSize );
}

// (x * f(x))' = f(x) + xf'(x) = [memoized] + xerf'(x)
// erf'(x) = 2/sqrt(pi) * e^(x^2)
// Adding some scales and shifts, we get [0.5( 1 + erf( x / sqrt(2) ) )] + x / sqrt( 2pi ) * e ^ ( -x^2 / 2 ), where [...] is saved from the forward pass
void CGELULayer::backwardPrecise()
{
	const int dataSize = inputBlobs[0]->GetDataSize();
	CFloatHandle input =  inputBlobs[0]->GetData();
	CFloatHandle inputDiff = inputDiffBlobs[0]->GetData();

	// inputDiff = input / sqrt(2)
	MathEngine().VectorMultiply( input, inputDiff, dataSize, sqrt2InvVar );

	// inputDiff = -(input^2 / 2)
	MathEngine().VectorNegMultiply( inputDiff, inputDiff, dataSize, inputDiff );

	// inputDiff = e^( -( input^2 / 2) )
	MathEngine().VectorExp( inputDiff, inputDiff, dataSize );

	// inputDiff = e^( -( input^2 / 2) ) / sqrt(2pi)
	MathEngine().VectorMultiply( inputDiff, inputDiff, dataSize, sqrt2PiInvVar );

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

	CFloatHandleStackVar buff( MathEngine(), 2 * static_cast<size_t>( dataSize ) );

	CFloatHandle multipliedInput = buff.GetHandle();
	CFloatHandle sigmoidMultipliedInput = buff.GetHandle() + dataSize;

	// multipliedInput = 1.702 * input
	MathEngine().VectorMultiply( input, multipliedInput, dataSize, approxScaleVar );

	// sigmoidMultipliedInput = sigmoid(1.702 * input)
	MathEngine().VectorSigmoid( multipliedInput, sigmoidMultipliedInput, dataSize );

	// inputDiffs = input * sigmoid_diff(1.702 * input)
	MathEngine().VectorSigmoidDiff( multipliedInput, input, inputDiff, dataSize );

	// inputDiffs = input * sigmoid_diff(1.702 * input) * 1.702
	MathEngine().VectorMultiply( inputDiff, inputDiff, dataSize, approxScaleVar );

	// inputDiff = sigmoid(1.702 * input) + input * sigmoid_diff(1.702 * input) * 1.702
	MathEngine().VectorAdd( inputDiff, sigmoidMultipliedInput, inputDiff, dataSize );

	// inputDiff *= outputDiff
	 MathEngine().VectorEltwiseMultiply( inputDiff, outputDiffBlobs[0]->GetData(), inputDiff, dataSize );
}

CLayerWrapper<CGELULayer> Gelu()
{
	return CLayerWrapper<CGELULayer>( "Gelu" );
}

} // namespace NeoML
