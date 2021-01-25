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

#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>

namespace NeoML {

CChannelwiseConvLayer::CChannelwiseConvLayer( IMathEngine& mathEngine ) :
	CBaseConvLayer( mathEngine, "CCnnChannelwiseConvLayer" ),
	convDesc( 0 )
{
}

CPtr<CDnnBlob> CChannelwiseConvLayer::GetFilterData() const
{
	if(Filter() == 0) {
		return 0;
	}

	return Filter()->GetCopy();
}

void CChannelwiseConvLayer::SetFilterData(const CPtr<CDnnBlob>& newFilter)
{
	NeoAssert( newFilter == nullptr || newFilter->GetObjectCount() == 1 );
	NeoAssert( newFilter == nullptr || newFilter->GetDepth() == 1 );
	CBaseConvLayer::SetFilterData(newFilter);
	if( Filter() != 0 ) {
		filterCount = Filter()->GetChannelsCount();
	}
}

void CChannelwiseConvLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( GetInputCount() == GetOutputCount(),
		GetName(), "different number of inputs and outputs in conv layer" );
	CheckArchitecture( paddingHeight < filterHeight && paddingWidth < filterWidth,
		GetName(), "padding is more or equal to filter size" );
	CheckArchitecture( MathEngine().IsInPlaceActivation( activation.Type ), GetName(),
		"activation must be in-place" );

	int outputHeight = ( inputDescs[0].Height() - filterHeight + 2 * paddingHeight ) / strideHeight + 1;
	int outputWidth = ( inputDescs[0].Width() - filterWidth + 2 * paddingWidth ) / strideWidth + 1;
	for( int i = 0; i < GetInputCount(); i++ ) {
		CheckArchitecture( filterHeight <= inputDescs[i].Height() + 2 * paddingHeight
			&& filterWidth <= inputDescs[i].Width() + 2 * paddingWidth,
			GetName(), "filter is bigger than input" );
		CheckArchitecture( (Filter() == 0 || filterCount == inputDescs[i].Channels()),
			GetName(), "filter count is not equal to input channels count" );
		CheckArchitecture( inputDescs[i].Depth() == 1, GetName(), "input depth is not equal to one" );

		if( Filter() == 0 ) {
			filterCount = inputDescs[i].Channels();
			// Create the weights matrix
			Filter() = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, 1, filterHeight, filterWidth,
				filterCount );
			// Initialize
			InitializeParamBlob( i, *Filter(), Filter()->GetObjectSize() );
		} else {
			NeoAssert( Filter()->GetObjectCount() == 1 );
			NeoAssert( Filter()->GetHeight() == filterHeight );
			NeoAssert( Filter()->GetWidth() == filterWidth );
			NeoAssert( Filter()->GetDepth() == 1 );
			NeoAssert( Filter()->GetChannelsCount() == filterCount);
		}

		if( FreeTerms() == 0 ) {
			FreeTerms() = CDnnBlob::CreateVector( MathEngine(), CT_Float, filterCount);
			// Initialize
			FreeTerms()->Fill( 0 );
		} else {
			CheckArchitecture( FreeTerms()->GetDataSize() == filterCount,
				GetName(), "number of free members in convolution is not equal to number of filters" );
		}

		// For each layer element, there is one channel in the output blob
		outputDescs[i] = CBlobDesc( CT_Float );
		outputDescs[i].SetDimSize( BD_BatchLength, inputDescs[i].BatchLength() );
		outputDescs[i].SetDimSize( BD_BatchWidth, inputDescs[i].BatchWidth() );
		outputDescs[i].SetDimSize( BD_ListSize, inputDescs[i].ListSize() );
		outputDescs[i].SetDimSize( BD_Height, outputHeight );
		outputDescs[i].SetDimSize( BD_Width, outputWidth );
		outputDescs[i].SetDimSize( BD_Depth, 1 );
		outputDescs[i].SetDimSize( BD_Channels, filterCount );
	}

	destroyConvDesc();
}

void CChannelwiseConvLayer::RunOnce()
{
	initConvDesc();

	CConstFloatHandle freeTerm = FreeTerms()->GetData();
	for( int i = 0; i < outputBlobs.Size(); ++i ) {
		MathEngine().BlobChannelwiseConvolution( *convDesc,
			inputBlobs[i]->GetData(), Filter()->GetData(),
			IsZeroFreeTerm() ? 0 : &freeTerm, outputBlobs[i]->GetData() );
	}
}

void CChannelwiseConvLayer::BackwardOnce()
{
	initConvDesc();

	for( int i = 0; i < inputDiffBlobs.Size(); ++i ) {
		MathEngine().BlobChannelwiseConvolutionBackward( *convDesc, outputBlobs[i]->GetData(),
			outputDiffBlobs[i]->GetData(), Filter()->GetData(), inputDiffBlobs[i]->GetData() );
	}
}

void CChannelwiseConvLayer::LearnOnce()
{
	initConvDesc();

	CFloatHandle freeTermDiff = FreeTermsDiff()->GetData();
	for( int i = 0; i < outputDiffBlobs.Size(); ++i ) {
		MathEngine().BlobChannelwiseConvolutionLearnAdd( *convDesc,
			inputBlobs[i]->GetData(), IsBackwardPerformed() ? CFloatHandle() : outputBlobs[i]->GetData(),
			outputDiffBlobs[i]->GetData(), FilterDiff()->GetData(), IsZeroFreeTerm() ? 0 : &freeTermDiff );
	}
}

static const int ChannelwiseConvLayerVersion = 2000;

void CChannelwiseConvLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ChannelwiseConvLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseConvLayer::Serialize( archive );
}

void CChannelwiseConvLayer::initConvDesc()
{
	if( convDesc == 0 ) {
		convDesc = MathEngine().InitBlobChannelwiseConvolution( inputBlobs[0]->GetDesc(),
			paddingHeight, paddingWidth, strideHeight, strideWidth,
			Filter()->GetDesc(), &FreeTerms()->GetDesc(), outputBlobs[0]->GetDesc(), activation );
	}
}

void CChannelwiseConvLayer::destroyConvDesc()
{
	if( convDesc != 0 ) {
		delete convDesc;
		convDesc = 0;
	}
}

CLayerWrapper<CChannelwiseConvLayer> ChannelwiseConv( int filterCount,
	const CConvAxisParams& heightParams, const CConvAxisParams& widthParams,
	bool isZeroFreeTerm )
{
	return CLayerWrapper<CChannelwiseConvLayer>( "ChannelwiseConv", [=]( CChannelwiseConvLayer* result ) {
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
