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

CRleConvLayer::CRleConvLayer( IMathEngine& mathEngine ) :
	CBaseConvLayer( mathEngine, "CCnnRleConvLayer" ),
	strokeValue( 1 ),
	nonStrokeValue( -1 ),
	convDesc( 0 )
{
}

CRleConvLayer::~CRleConvLayer()
{
	destroyConvDesc();
}

void CRleConvLayer::SetFilterData( const CPtr<CDnnBlob>& newFilter )
{
	CBaseConvLayer::SetFilterData( newFilter );
	ForceReshape();
}

void CRleConvLayer::SetFreeTermData( const CPtr<CDnnBlob>& newFreeTerm )
{
	CBaseConvLayer::SetFreeTermData( newFreeTerm );
	ForceReshape();
}

void CRleConvLayer::Reshape()
{
	CheckInputs();
	NeoAssert( GetInputCount() > 0 && GetInputCount() == GetOutputCount() );
	NeoAssert( filterWidth <= MaxRleConvFilterWidth );
	NeoAssert( inputDescs[0].Width() <= MaxRleConvImageWidth );
	NeoAssert( inputDescs[0].Depth() == 1 );
	NeoAssert( inputDescs[0].Channels() == 1 );
	NeoAssert( paddingHeight == 0 );
	NeoAssert( paddingWidth == 0 );
	NeoAssert( dilationHeight == 1 );
	NeoAssert( dilationWidth == 1 );
	NeoAssert( ( filterCount % 4 ) == 0 );

	int outputHeight, outputWidth;
	calcOutputBlobSize( outputHeight, outputWidth );
	for( int i = 0; i < GetInputCount(); i++ ) {
		NeoAssert( filterHeight <= inputDescs[i].Height() && filterWidth <= inputDescs[i].Width() );

		if( Filter() == 0 ) {
			// Create the weights matrix
			Filter() = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1, filterCount, filterHeight, filterWidth, 1 );
			// Initialize
			InitializeParamBlob( i, *Filter(), Filter()->GetObjectSize() );
		} else {
			NeoAssert( Filter()->GetObjectCount() == filterCount );
			NeoAssert( Filter()->GetHeight() == filterHeight );
			NeoAssert( Filter()->GetWidth() == filterWidth );
			NeoAssert( Filter()->GetDepth() == 1 );
			NeoAssert( Filter()->GetChannelsCount() == 1 );
		}

		if( FreeTerms() == 0 ) {
			FreeTerms() = CDnnBlob::CreateVector( MathEngine(), CT_Float, filterCount );
			// Initialize
			FreeTerms()->Fill( 0 );
		} else {
			NeoAssert( FreeTerms()->GetDataSize() == filterCount );
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

void CRleConvLayer::RunOnce()
{
	static_assert( sizeof( CRleStroke ) == sizeof( float ), "sizeof( CRleStroke ) != sizeof( float )" );
	static_assert( sizeof( int ) == sizeof( float ), "sizeof( int ) != sizeof( float )" );

	initConvDesc();

	CFloatHandle freeTerms = FreeTerms()->GetData();
	for( int i = 0; i < inputBlobs.Size(); ++i ) {
		MathEngine().BlobRleConvolution( *convDesc, inputBlobs[i]->GetData(), Filter()->GetData(),
			IsZeroFreeTerm() ? 0 : &freeTerms, outputBlobs[i]->GetData() );
	}
}

void CRleConvLayer::BackwardOnce()
{
	NeoAssert( false ); // the previous layers may not be trained
}

void CRleConvLayer::LearnOnce()
{
	CFloatHandle freeTermDiff = FreeTermsDiff()->GetData();
	for( int i = 0; i < outputDiffBlobs.Size(); ++i ) {
		MathEngine().BlobRleConvolutionLearnAdd( *convDesc, inputBlobs[i]->GetData(), outputDiffBlobs[i]->GetData(),
			FilterDiff()->GetData(), IsZeroFreeTerm() ? 0 : &freeTermDiff );
	}
}

static const int RleConvLayerVersion = 2000;

void CRleConvLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( RleConvLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseConvLayer::Serialize( archive );

	archive.Serialize( strokeValue );
	archive.Serialize( nonStrokeValue );

	if( archive.IsLoading() ) {
		if( version < 2000 ) {
			// Starting with version 1001, the filter is stored in the same format for all platforms
			CPtr<CDnnBlob> convertedFilter = CDnnBlob::Create2DImageBlob( Filter()->GetMathEngine(), CT_Float, 1,
				Filter()->GetWidth(), Filter()->GetObjectCount(), Filter()->GetHeight(), 1 );
			Filter()->GetMathEngine().TransposeMatrix( 1, Filter()->GetData(),
				Filter()->GetHeight() * Filter()->GetObjectCount(), 1, Filter()->GetWidth(), 1,
				convertedFilter->GetData(), convertedFilter->GetDataSize() );
			Filter() = convertedFilter;
		}
		destroyConvDesc();
	}
}

void CRleConvLayer::calcOutputBlobSize( int& outputHeight, int& outputWidth ) const
{
	// The output blob height is the number of filter windows that fit in vertically
	outputHeight = (inputDescs[0].Height() - filterHeight + 2 * paddingHeight) / strideHeight + 1;
	// The output blob width is the number of filter windows that fit in horizontally
	outputWidth = (inputDescs[0].Width() - filterWidth + 2 * paddingWidth) / strideWidth + 1;
}

void CRleConvLayer::initConvDesc()
{
	if( convDesc == 0 ) {
		convDesc = MathEngine().InitBlobRleConvolution( inputDescs[0], strokeValue, nonStrokeValue,
			strideHeight, strideWidth, Filter()->GetDesc(), outputDescs[0] );
	}
}

void CRleConvLayer::destroyConvDesc()
{
	if( convDesc != 0 ) {
		delete convDesc;
		convDesc = 0;
	}
}

} // namespace NeoML
