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

static const float GELUMultiplier = 1.702f;

CGELULayer::CGELULayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CGELULayer", false ),
	multiplierVar( mathEngine )
{
	multiplierVar.SetValue( GELUMultiplier );
}

static const int CGELULayerVersion = 0;

void CGELULayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CGELULayerVersion );
	CBaseLayer::Serialize( archive );
}

void CGELULayer::Reshape()
{
	CheckInputs();
	assert( inputDescs.Size() == 1 );

	const CBlobDesc& inputDesc = inputDescs[0];

	outputDescs.SetSize( 1 );
	outputDescs[0] = inputDesc;
}

void CGELULayer::RunOnce()
{
	CheckInput1();

	// output = 1.702 * input
	MathEngine().VectorMultiply( inputBlobs[0]->GetData(), outputBlobs[0]->GetData(),
		inputBlobs[0]->GetDataSize(), multiplierVar );

	// output = sigmoid(1.702 * intput)
	MathEngine().VectorSigmoid( outputBlobs[0]->GetData(), outputBlobs[0]->GetData(),
		outputBlobs[0]->GetDataSize() );

	// output = input * sigmoid(1.702 * input)
	MathEngine().VectorEltwiseMultiply( inputBlobs[0]->GetData(), outputBlobs[0]->GetData(),
		outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize() );
}

void CGELULayer::BackwardOnce()
{
	const int blobSize = inputBlobs[0]->GetDataSize();

	CFloatHandleStackVar buff( MathEngine(), 2 * static_cast<size_t>( blobSize ) );

	CFloatHandle multipliedInput = buff.GetHandle();
	CFloatHandle sigmoidMultipliedInput = buff.GetHandle() + blobSize;

	// multipliedInput = 1.702 * input
	MathEngine().VectorMultiply( inputBlobs[0]->GetData(), multipliedInput, blobSize, multiplierVar );

	// sigmoidMultipliedInput = sigmoid(1.702 * input)
	MathEngine().VectorSigmoid( multipliedInput, sigmoidMultipliedInput, blobSize );

	// inputDiffs = input * sigmoid_diff(1.702 * input)
	MathEngine().VectorSigmoidDiff( multipliedInput, inputBlobs[0]->GetData(), inputDiffBlobs[0]->GetData(), blobSize );

	// inputDiffs = input * sigmoid_diff(1.702 * input) * 1.702
	MathEngine().VectorMultiply( inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetData(), blobSize, multiplierVar );

	// inputDiff = sigmoid(1.702 * input) + input * sigmoid_diff(1.702 * input) * 1.702
	MathEngine().VectorAdd( inputDiffBlobs[0]->GetData(), sigmoidMultipliedInput, inputDiffBlobs[0]->GetData(), blobSize );

	// inputDiff *= outputDiff
	MathEngine().VectorEltwiseMultiply( inputDiffBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), blobSize );
}

} // namespace FML
