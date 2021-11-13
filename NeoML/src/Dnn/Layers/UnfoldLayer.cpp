/* Copyright © 2017-2021 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/UnfoldLayer.h>

namespace NeoML {

CUnfoldLayer::CUnfoldLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CUnfoldLayer", false ),
	filterHeight( 1 ),
	filterWidth( 1 ),
	strideHeight( 1 ),
	strideWidth( 1 ),
	paddingHeight( 0 ),
	paddingWidth( 0 ),
	dilationHeight( 1 ),
	dilationWidth( 1 )
{	
}

void CUnfoldLayer::SetFilterHeight( int value )
{
	NeoAssert( value > 0 );
	filterHeight = value;
	ForceReshape();
}

void CUnfoldLayer::SetFilterWidth( int value )
{
	NeoAssert( value > 0 );
	filterWidth = value;
	ForceReshape();
}

void CUnfoldLayer::SetStrideHeight( int value )
{
	NeoAssert( value > 0 );
	strideHeight = value;
	ForceReshape();
}

void CUnfoldLayer::SetStrideWidth( int value )
{
	NeoAssert( value > 0 );
	strideWidth = value;
	ForceReshape();
}

void CUnfoldLayer::SetPaddingHeight( int value )
{
	NeoAssert( value >= 0 );
	paddingHeight = value;
	ForceReshape();
}

void CUnfoldLayer::SetPaddingWidth( int value )
{
	NeoAssert( value >= 0 );
	paddingWidth = value;
	ForceReshape();
}

void CUnfoldLayer::SetDilationHeight( int value )
{
	NeoAssert( value > 0 );
	dilationHeight = value;
	ForceReshape();
}

void CUnfoldLayer::SetDilationWidth( int value )
{
	NeoAssert( value > 0 );
	dilationWidth = value;
	ForceReshape();
}

static const int UnfoldLayerVersion = 0;

void CUnfoldLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( UnfoldLayerVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( filterHeight );
	archive.Serialize( filterWidth );
	archive.Serialize( strideHeight );
	archive.Serialize( strideWidth );
	archive.Serialize( paddingHeight );
	archive.Serialize( paddingWidth );
	archive.Serialize( dilationHeight );
	archive.Serialize( dilationWidth );
}

void CUnfoldLayer::Reshape()
{
	CheckInput1();
	const int filterRegionHeight = 1 + ( filterHeight - 1 ) * dilationHeight;
	const int fullImageHeight = inputDescs[0].Height() + 2 * paddingHeight;
	const int filterRegionWidth = 1 + ( filterWidth - 1 ) * dilationWidth;
	const int fullImageWidth = inputDescs[0].Width() + 2 * paddingWidth;
	CheckArchitecture( fullImageHeight >= filterRegionHeight && fullImageWidth >= filterRegionWidth,
		GetName(), "Wrong convolution parameters: filter is larger than image" );
	outputDescs[0] = inputDescs[0];
	const int convOutputHeight = 1 + ( fullImageHeight - filterRegionHeight ) / strideHeight;
	const int convOutputWidth = 1 + ( fullImageWidth - filterRegionWidth ) / strideWidth;
	outputDescs[0].SetDimSize( BD_Height, convOutputHeight * convOutputWidth );
	outputDescs[0].SetDimSize( BD_Width, 1 );
	outputDescs[0].SetDimSize( BD_Depth, 1 );
	outputDescs[0].SetDimSize( BD_Channels, filterHeight * filterWidth * inputDescs[0].Depth() * inputDescs[0].Channels() );
}

void CUnfoldLayer::RunOnce()
{
	MathEngine().Unfold( inputBlobs[0]->GetObjectCount(), inputBlobs[0]->GetData(), inputBlobs[0]->GetHeight(),
		inputBlobs[0]->GetWidth(), inputBlobs[0]->GetDepth() * inputBlobs[0]->GetChannelsCount(),
		outputBlobs[0]->GetData(), filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth,
		dilationHeight, dilationWidth );
}

void CUnfoldLayer::BackwardOnce()
{
	MathEngine().Fold( outputDiffBlobs[0]->GetObjectCount(), outputDiffBlobs[0]->GetData(), filterHeight, filterWidth,
		strideHeight, strideWidth, paddingHeight, paddingWidth, dilationHeight, dilationWidth,
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetHeight(), inputDiffBlobs[0]->GetWidth(),
		inputDiffBlobs[0]->GetDepth() * inputDiffBlobs[0]->GetChannelsCount() );
}

// --------------------------------------------------------------------------------------------------------------------

CLayerWrapper<CUnfoldLayer> Unfold( int filterHeight, int filterWidth, int strideHeight, int strideWidth,
	int paddingHeight, int paddingWidth, int dilationHeight, int dilationWidth )
{
	return CLayerWrapper<CUnfoldLayer>( "Unfold", [=]( CUnfoldLayer* result ) {
		result->SetFilterHeight( filterHeight );
		result->SetFilterWidth( filterWidth );
		result->SetStrideHeight( strideHeight );
		result->SetStrideWidth( strideWidth );
		result->SetPaddingHeight( paddingHeight );
		result->SetPaddingWidth( paddingWidth );
		result->SetDilationHeight( dilationHeight );
		result->SetDilationWidth( dilationWidth );
	} );
}

} // namespace NeoML
