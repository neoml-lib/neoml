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

#include <NeoML/Dnn/Layers/TransposedConvLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CTransposedConvLayer::CTransposedConvLayer( IMathEngine& mathEngine ) :
	CBaseConvLayer( mathEngine, "CCnnTransposedConvLayer" ),
	convDesc(0)
{
}

void CTransposedConvLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( GetInputCount() == GetOutputCount(),
		GetName(), "different number of inputs and outputs in conv layer" );
	CheckArchitecture( paddingHeight < filterHeight && paddingWidth < filterWidth,
		GetName(), "padding is more or equal to filter size" );
	CheckArchitecture( activation.Type == AF_None, GetName(), "activation is not supported in transposed conv" );

	int outputHeight, outputWidth;
	calcOutputBlobSize(outputHeight, outputWidth);
	for( int i = 0; i < GetInputCount(); i++ ) {
		if( Filter() == 0 ) {
			// Create the weights matrix
			Filter() = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1,
				inputDescs[i].Depth() * inputDescs[i].Channels(),
				filterHeight, filterWidth, filterCount );
			// Initialize
			InitializeParamBlob(i, *Filter(), Filter()->GetObjectSize());
		} else {
			NeoAssert(Filter()->GetBatchWidth() == inputDescs[i].Channels() * inputDescs[i].Depth());
			NeoAssert(Filter()->GetHeight() == filterHeight);
			NeoAssert(Filter()->GetWidth() == filterWidth);
			NeoAssert(Filter()->GetDepth() == 1);
			NeoAssert(Filter()->GetChannelsCount() == filterCount);
		}

		if( FreeTerms() == 0 ) {
			FreeTerms() = CDnnBlob::CreateVector( MathEngine(), CT_Float, filterCount );
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
		outputDescs[i].SetDimSize( BD_Depth, 1 );
		outputDescs[i].SetDimSize( BD_Channels, filterCount );
	}
	destroyConvDesc();
}

void CTransposedConvLayer::RunOnce()
{
	initConvDesc();

	CFloatHandle freeTerm = FreeTerms()->GetData();
	for(int i = 0; i < outputBlobs.Size(); ++i) {
		MathEngine().BlobConvolutionBackward( *convDesc, CFloatHandle(), inputBlobs[i]->GetData(),
			Filter()->GetData(), IsZeroFreeTerm() ? 0 : &freeTerm, outputBlobs[i]->GetData() );
	}
}

void CTransposedConvLayer::BackwardOnce()
{
	initConvDesc();

	for(int i = 0; i < inputDiffBlobs.Size(); ++i) {
		MathEngine().BlobConvolution( *convDesc, outputDiffBlobs[i]->GetData(),
			Filter()->GetData(), 0, inputDiffBlobs[i]->GetData() );
	}
}

void CTransposedConvLayer::LearnOnce()
{
	initConvDesc();

	CFloatHandle freeTermDiff = FreeTermsDiff()->GetData();
	for(int i = 0; i < outputDiffBlobs.Size(); ++i) {
		MathEngine().BlobConvolutionLearnAdd( *convDesc, outputDiffBlobs[i]->GetData(), CFloatHandle(),
			inputBlobs[i]->GetData(), FilterDiff()->GetData(), IsZeroFreeTerm() ? 0 : &freeTermDiff, true );
	}
}

void CTransposedConvLayer::destroyConvDesc()
{
	if( convDesc != 0 ) {
		delete convDesc;
		convDesc = 0;
	}
}

void CTransposedConvLayer::initConvDesc()
{
	if( convDesc == 0 ) {
		convDesc = MathEngine().InitBlobConvolution( outputBlobs[0]->GetDesc(), paddingHeight, paddingWidth,
			strideHeight, strideWidth, dilationHeight, dilationWidth, Filter()->GetDesc(), inputBlobs[0]->GetDesc(), activation );
	}
}

void CTransposedConvLayer::calcOutputBlobSize(int& outputHeight, int& outputWidth) const
{
	// The output blob height is the number of filter windows that fit in vertically
	outputHeight = strideHeight * (inputDescs[0].Height() - 1) + ( filterHeight - 1 ) * dilationHeight + 1 - 2 * paddingHeight;
	// The output blob width is the number of filter windows that fit in horizontally
	outputWidth = strideWidth * (inputDescs[0].Width() - 1) + ( filterWidth - 1 ) * dilationWidth + 1 - 2 * paddingWidth;
}

static const int TransposedConvLayerVersion = 2000;

void CTransposedConvLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( TransposedConvLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseConvLayer::Serialize( archive );
}

CLayerWrapper<CTransposedConvLayer> TransposedConv( int filterCount,
	const CConvAxisParams& heightParams, const CConvAxisParams& widthParams,
	bool isZeroFreeTerm )
{
	return CLayerWrapper<CTransposedConvLayer>( "TransposedConv", [=]( CTransposedConvLayer* result ) {
		result->SetFilterCount( filterCount );

		result->SetFilterHeight( heightParams.FilterSize );
		result->SetPaddingHeight( heightParams.Padding );
		result->SetStrideWidth( heightParams.Stride );
		result->SetDilationHeight( heightParams.Dilation );

		result->SetFilterWidth( widthParams.FilterSize );
		result->SetPaddingWidth( widthParams.Padding );
		result->SetStrideHeight( widthParams.Stride );
		result->SetDilationWidth( widthParams.Dilation );

		result->SetZeroFreeTerm( isZeroFreeTerm );
	} );
}

} // namespace NeoML
