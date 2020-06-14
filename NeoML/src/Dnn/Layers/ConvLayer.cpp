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

#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CConvLayer::CConvLayer( IMathEngine& mathEngine ) :
	CBaseConvLayer( mathEngine, "CCnnConvLayer" ),
	convDesc( 0 )
{
}

CConvLayer::~CConvLayer()
{
	destroyConvDesc();
}

void CConvLayer::initConvDesc()
{
	if( convDesc == 0 ) {
		convDesc = MathEngine().InitBlobConvolution( inputBlobs[0]->GetDesc(),
			paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
			Filter()->GetDesc(), outputBlobs[0]->GetDesc() );
	}
}
void CConvLayer::destroyConvDesc()
{
	if( convDesc != 0 ) {
		delete convDesc;
		convDesc = 0;
	}
}

// Calculates the output blob size from the convolution parameters
void CConvLayer::calcOutputBlobSize(int& outputHeight, int& outputWidth) const
{
	// The blob height is the number of filter windows that fit into the input vertically
	outputHeight = 1 + ( inputDescs[0].Height() - ( filterHeight - 1 ) * dilationHeight + 2 * paddingHeight - 1 )
		/ strideHeight;
	// The blob width is the number of filter windows that fit into the window horizontally
	outputWidth = 1 + ( inputDescs[0].Width() - ( filterWidth - 1 ) * dilationWidth + 2 * paddingWidth - 1 )
		/ strideWidth;
}

void CConvLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( GetInputCount() == GetOutputCount(),
		GetName(), "different number of inputs and outputs in conv layer" );
	CheckArchitecture( paddingHeight < filterHeight * dilationHeight && paddingWidth < filterWidth * dilationWidth,
		GetName(), "padding is more or equal to receptive field size" );

	int outputHeight, outputWidth;
	calcOutputBlobSize(outputHeight, outputWidth);
	for(int i = 0; i < GetInputCount(); i++) {
		CheckArchitecture( filterHeight <= inputDescs[i].Height() + 2 * paddingHeight
			&& filterWidth <= inputDescs[i].Width() + 2 * paddingWidth,
			GetName(), "filter is bigger than input" );

		if(Filter() == 0) {
			// Create a weights matrix
			Filter() = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, filterCount, filterHeight, filterWidth,
				inputDescs[i].Depth(), inputDescs[i].Channels() );
			// Initialize
			InitializeParamBlob(i, *Filter(), Filter()->GetObjectSize());
		} else {
			NeoAssert(Filter()->GetObjectCount() == filterCount);
			NeoAssert(Filter()->GetHeight() == filterHeight);
			NeoAssert(Filter()->GetWidth() == filterWidth);
			NeoAssert(Filter()->GetDepth() == inputDescs[i].Depth());
			NeoAssert(Filter()->GetChannelsCount() == inputDescs[i].Channels());
		}

		if(FreeTerms() == 0) {
			FreeTerms() = CDnnBlob::CreateVector( MathEngine(), CT_Float, filterCount );
			// Initialize
			FreeTerms()->Fill(0);
		} else {
			CheckArchitecture( FreeTerms()->GetDataSize() == filterCount,
				GetName(), "number of free members in convolution is not equal to number of filters" );
		}

		// For each layer element there is one channel in the output blob
		outputDescs[i] = inputDescs[i];
		outputDescs[i].SetDimSize( BD_Height, outputHeight );
		outputDescs[i].SetDimSize( BD_Width, outputWidth );
		outputDescs[i].SetDimSize( BD_Depth, 1 );
		outputDescs[i].SetDimSize( BD_Channels, filterCount );
	}

	destroyConvDesc();
}

void CConvLayer::RunOnce()
{
	initConvDesc();

	for( int i = 0; i < outputBlobs.Size(); ++i ) {
		CFloatHandle freeTerm = FreeTerms()->GetData();
		MathEngine().BlobConvolution( *convDesc, inputBlobs[i]->GetData(),
			Filter()->GetData(), &freeTerm, outputBlobs[i]->GetData() );
	}
}

void CConvLayer::BackwardOnce()
{
	initConvDesc();

	for( int i = 0; i < inputDiffBlobs.Size(); ++i ) {
		MathEngine().BlobConvolutionBackward( *convDesc, outputDiffBlobs[i]->GetData(),
			Filter()->GetData(), 0, inputDiffBlobs[i]->GetData() );
	}
}

void CConvLayer::LearnOnce()
{
	initConvDesc();

	CFloatHandle freeTermDiff = FreeTermsDiff()->GetData();
	for(int i = 0; i < outputDiffBlobs.Size(); ++i) {
		MathEngine().BlobConvolutionLearnAdd( *convDesc, inputBlobs[i]->GetData(), outputDiffBlobs[i]->GetData(),
			FilterDiff()->GetData(), IsZeroFreeTerm() ? 0 : &freeTermDiff, false );
	}
}

static const int ConvLayerVersion = 2000;

void CConvLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ConvLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseConvLayer::Serialize( archive );
}

} // namespace NeoML
