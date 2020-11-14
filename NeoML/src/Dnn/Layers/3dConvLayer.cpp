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

#include <NeoML/Dnn/Layers/3dConvLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CBase3dConvLayer::CBase3dConvLayer( IMathEngine& mathEngine, const char* name ) :
	CBaseConvLayer( mathEngine, name )
{
	filterDepth = 1;
	strideDepth = 1;
	filterCount = 1;
	paddingDepth = 0;
}

void CBase3dConvLayer::SetFilterDepth(int _filterDepth)
{
	NeoAssert(GetDnn() == 0);
	filterDepth = _filterDepth;
	ForceReshape();
}

void CBase3dConvLayer::SetStrideDepth(int _strideDepth)
{
	NeoAssert(GetDnn() == 0);
	strideDepth = _strideDepth;
	ForceReshape();
}

void CBase3dConvLayer::SetPaddingDepth(int _paddingDepth)
{
	NeoAssert(GetDnn() == 0);
	paddingDepth = _paddingDepth;
	ForceReshape();
}

static const int Base3dConvLayerVersion = 2000;

void CBase3dConvLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( Base3dConvLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseConvLayer::Serialize( archive );
	archive.Serialize( filterDepth );
	archive.Serialize( strideDepth );
	archive.Serialize( paddingDepth );

	if( archive.IsLoading() ) {
		// Converts the free terms blob into a new tensor configuration, with the length in the first dimension not in Channels
		CDnnBlob* freeTerms = FreeTerms();
		if( freeTerms != 0 && freeTerms->DimSize(0) != freeTerms->GetDataSize() ) {
			NeoAssert( freeTerms->GetChannelsCount() == freeTerms->GetDataSize() );
			CBlobDesc desc( CT_Float );
			desc.SetDimSize( 0, freeTerms->GetDataSize() );
			freeTerms->ReinterpretDimensions( desc );
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////

C3dConvLayer::C3dConvLayer( IMathEngine& mathEngine ) :
	CBase3dConvLayer( mathEngine, "CCnn3dConvLayer" ),
	convDesc( 0 )
{
}

C3dConvLayer::~C3dConvLayer()
{
	destroyConvDesc();
}

void C3dConvLayer::calcOutputBlobSize(int& outputHeight, int& outputWidth, int& outputDepth) const
{
	// The output blob height, which is the number of filter windows vertically
	outputHeight = (inputDescs[0].Height() - filterHeight + 2 * paddingHeight) / strideHeight + 1;
	// The output blob width, which is the number of filter windows horizontally
	outputWidth = (inputDescs[0].Width() - filterWidth + 2 * paddingWidth) / strideWidth + 1;
	// The output blob depth, which is the number of filter windows by depth
	outputDepth = (inputDescs[0].Depth() - filterDepth + 2 * paddingDepth) / strideDepth + 1;
}

void C3dConvLayer::destroyConvDesc()
{
	if(convDesc != 0) {
		delete convDesc;
		convDesc = 0;
	}
}

void C3dConvLayer::initConvDesc()
{
	if(convDesc == 0) {
		convDesc = MathEngine().InitBlob3dConvolution(inputBlobs[0]->GetDesc(), paddingHeight, paddingWidth, paddingDepth,
			strideHeight, strideWidth, strideDepth, Filter()->GetDesc(), outputBlobs[0]->GetDesc(), activation);
	}
}

void C3dConvLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( GetInputCount() == GetOutputCount(),
		GetName(), "different number of inputs and outputs in conv layer" );
	CheckArchitecture( paddingHeight < filterHeight && paddingWidth < filterWidth && paddingDepth < filterDepth,
		GetName(), "padding is more or equal to filter size" );
	CheckArchitecture( MathEngine().IsInPlaceActivation( activation.Type ), GetName(),
		"activation must be in-place" );

	int outputHeight, outputWidth, outputDepth;
	calcOutputBlobSize(outputHeight, outputWidth, outputDepth);
	for(int i = 0; i < GetInputCount(); i++) {
		CheckArchitecture( filterHeight <= inputDescs[i].Height() + 2 * paddingHeight
			&& filterWidth <= inputDescs[i].Width() + 2 * paddingWidth
			&& filterDepth <= inputDescs[i].Depth() + 2 * paddingDepth,
			GetName(), "filter is bigger than input" );

		if(Filter() == 0) {
			// Create a weights matrix
			Filter() = CDnnBlob::Create3DImageBlob(MathEngine(), CT_Float, 1, filterCount, filterHeight, filterWidth,
				filterDepth, inputDescs[i].Channels());
			// Initialize
			InitializeParamBlob(i, *Filter(), Filter()->GetObjectSize());
		} else {
			NeoAssert(Filter()->GetObjectCount() == filterCount);
			NeoAssert(Filter()->GetHeight() == filterHeight);
			NeoAssert(Filter()->GetWidth() == filterWidth);
			NeoAssert(Filter()->GetDepth() == filterDepth);
			NeoAssert(Filter()->GetChannelsCount() == inputDescs[i].Channels());
		}

		if(FreeTerms() == 0) {
			FreeTerms() = CDnnBlob::CreateVector(MathEngine(), CT_Float, filterCount);
			// Initialize
			FreeTerms()->Fill(0);
		} else {
			CheckArchitecture( FreeTerms()->GetDataSize() == filterCount,
				GetName(), "number of free members in convolution is not equal to number of filters" );
		}
		// For each layer element, there is a channel in the output blob
		outputDescs[i] = inputDescs[i];
		outputDescs[i].SetDimSize( BD_Height, outputHeight );
		outputDescs[i].SetDimSize( BD_Width, outputWidth );
		outputDescs[i].SetDimSize( BD_Depth, outputDepth );
		outputDescs[i].SetDimSize( BD_Channels, filterCount );
	}
	destroyConvDesc();
}

void C3dConvLayer::RunOnce()
{
	initConvDesc();

	for(int i = 0; i < outputBlobs.Size(); ++i) {
		CFloatHandle freeTerm = FreeTerms()->GetData();
		MathEngine().Blob3dConvolution( *convDesc, inputBlobs[i]->GetData(), Filter()->GetData(),
			IsZeroFreeTerm() ? 0 : &freeTerm, outputBlobs[i]->GetData() );
	}
}

void C3dConvLayer::BackwardOnce()
{
	initConvDesc();

	for(int i = 0; i < inputDiffBlobs.Size(); ++i) {
		MathEngine().Blob3dConvolutionBackward( *convDesc, outputBlobs[i]->GetData(), outputDiffBlobs[i]->GetData(),
			Filter()->GetData(), 0, inputDiffBlobs[i]->GetData() );
	}
}

void C3dConvLayer::LearnOnce()
{
	initConvDesc();

	CFloatHandle freeTermDiff = FreeTermsDiff()->GetData();
	for(int i = 0; i < outputDiffBlobs.Size(); ++i) {
		MathEngine().Blob3dConvolutionLearnAdd( *convDesc, inputBlobs[i]->GetData(), outputDiffBlobs[i]->GetData(),
			FilterDiff()->GetData(), IsZeroFreeTerm() ? 0 : &freeTermDiff, false );
	}
}

static const int Conv3dLayerVersion = 2000;

void C3dConvLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( Conv3dLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBase3dConvLayer::Serialize( archive );

	if( archive.IsLoading() ) {
		destroyConvDesc();
	}
}

CLayerWrapper<C3dConvLayer> Conv3d( int filterCount,
	const CConvAxisParams& heightParams, const CConvAxisParams& widthParams,
	const CConvAxisParams& depthParams, bool isZeroFreeTerm )
{
	return CLayerWrapper<C3dConvLayer>( "Conv3d", [=]( C3dConvLayer* result ) {
		result->SetFilterCount( filterCount );

		result->SetFilterHeight( heightParams.FilterSize );
		result->SetPaddingHeight( heightParams.Padding );
		result->SetStrideWidth( heightParams.Stride );

		result->SetFilterWidth( widthParams.FilterSize );
		result->SetPaddingWidth( widthParams.Padding );
		result->SetStrideHeight( widthParams.Stride );

		result->SetFilterDepth( depthParams.FilterSize );
		result->SetPaddingDepth( depthParams.Padding );
		result->SetStrideDepth( depthParams.Stride );
		// layer has no dilation

		result->SetZeroFreeTerm( isZeroFreeTerm );
	} );
}

} // namespace NeoML
