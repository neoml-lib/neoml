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

#include <NeoML/Dnn/Layers/3dTransposedConvLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

C3dTransposedConvLayer::C3dTransposedConvLayer( IMathEngine& mathEngine ) :
	CBase3dConvLayer( mathEngine, "CCnn3dTransposedConvLayer" ),
	convDesc(0)
{
}

static const int TransposedConv3dVersion = 2000;

void C3dTransposedConvLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( TransposedConv3dVersion );
	CBase3dConvLayer::Serialize( archive );
}

void C3dTransposedConvLayer::calcOutputBlobSize(int& outputHeight, int& outputWidth, int& outputDepth) const
{
	// The output blob height, which is the number of filter windows vertically
	outputHeight = strideHeight * (inputDescs[0].Height() - 1) + filterHeight - 2 * paddingHeight;
	// The output blob width, which is the number of filter windows horizontally
	outputWidth = strideWidth * (inputDescs[0].Width() - 1) + filterWidth - 2 * paddingWidth;
	outputDepth = strideDepth * (inputDescs[0].Depth() - 1) + filterDepth - 2 * paddingDepth;
}

void C3dTransposedConvLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( GetInputCount() == GetOutputCount(),
		GetName(), "different number of inputs and outputs in conv layer" );
	CheckArchitecture( paddingHeight < filterHeight && paddingWidth < filterWidth && paddingDepth < filterDepth,
		GetName(), "padding is more or equal to filter size" );
	CheckArchitecture( activation.Type == AF_None, GetName(), "activation is not supported in transposed conv" );

	int outputHeight, outputWidth, outputDepth;
	calcOutputBlobSize(outputHeight, outputWidth, outputDepth);
	for( int i = 0; i < GetInputCount(); i++ ) {
		if( Filter() == 0 ) {
			// Create a weights matrix
			Filter() = CDnnBlob::Create3DImageBlob(MathEngine(), CT_Float, 1, inputDescs[i].Channels(),
				filterHeight, filterWidth, filterDepth, filterCount);
			// Initialize
			InitializeParamBlob(i, *Filter(), Filter()->GetObjectSize());
		} else {
			NeoAssert(Filter()->GetObjectCount() == inputDescs[i].Channels());
			NeoAssert(Filter()->GetHeight() == filterHeight);
			NeoAssert(Filter()->GetWidth() == filterWidth);
			NeoAssert(Filter()->GetDepth() == filterDepth);
			NeoAssert(Filter()->GetChannelsCount() == filterCount);
		}

		if( FreeTerms() == 0 ) {
			FreeTerms() = CDnnBlob::CreateVector(MathEngine(), CT_Float, filterCount);
			// Initialize
			FreeTerms()->Fill(0);
		} else {
			CheckArchitecture( FreeTerms()->GetDataSize() == filterCount,
				GetName(), "number of free members in convolution is not equal to number of filters" );
		}

		// For each layer element there is a channel in the output blob
		outputDescs[i] = inputDescs[i];
		outputDescs[i].SetDimSize( BD_Height, outputHeight );
		outputDescs[i].SetDimSize( BD_Width, outputWidth );
		outputDescs[i].SetDimSize( BD_Depth, outputDepth );
		outputDescs[i].SetDimSize( BD_Channels, filterCount );
	}
	destroyConvDesc();
}

void C3dTransposedConvLayer::initConvDesc()
{
	if( convDesc == 0 ) {
		convDesc = MathEngine().InitBlob3dConvolution(outputBlobs[0]->GetDesc(), paddingHeight, paddingWidth, paddingDepth,
			strideHeight, strideWidth, strideDepth, Filter()->GetDesc(), inputBlobs[0]->GetDesc(), activation);
	}
}

void C3dTransposedConvLayer::destroyConvDesc()
{
	if( convDesc != 0 ) {
		delete convDesc;
		convDesc = 0;
	}
}

void C3dTransposedConvLayer::RunOnce()
{
	initConvDesc();

	CFloatHandle freeTerm = FreeTerms()->GetData();
	for( int i = 0; i < outputBlobs.Size(); ++i ) {
		MathEngine().Blob3dConvolutionBackward( *convDesc, CConstFloatHandle(), inputBlobs[i]->GetData(),
			Filter()->GetData(), IsZeroFreeTerm() ? 0 : &freeTerm, outputBlobs[i]->GetData() );
	}
}

void C3dTransposedConvLayer::BackwardOnce()
{
	initConvDesc();

	for( int i = 0; i < inputDiffBlobs.Size(); ++i ) {
		MathEngine().Blob3dConvolution( *convDesc, outputDiffBlobs[i]->GetData(),
			Filter()->GetData(), 0, inputDiffBlobs[i]->GetData() );
	}
}

void C3dTransposedConvLayer::LearnOnce()
{
	initConvDesc();

	CFloatHandle freeTermDiff = FreeTermsDiff()->GetData();
	for( int i = 0; i < outputDiffBlobs.Size(); ++i ) {
		MathEngine().Blob3dConvolutionLearnAdd( *convDesc, outputDiffBlobs[i]->GetData(), inputBlobs[i]->GetData(),
			FilterDiff()->GetData(), IsZeroFreeTerm() ? 0 : &freeTermDiff, true);
	}
}

} // namespace NeoML
