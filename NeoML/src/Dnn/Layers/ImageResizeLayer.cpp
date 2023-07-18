/* Copyright © 2017-2023 ABBYY

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

#include <NeoML/Dnn/Layers/ImageResizeLayer.h>

namespace NeoML {

CImageResizeLayer::CImageResizeLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnImageResizeLayer", false ),
	deltaLeft( 0 ),
	deltaRight( 0 ),
	deltaTop( 0 ),
	deltaBottom( 0 ),
	defaultValue( 0 ),
	padding( TBlobResizePadding::Constant )
{
}

int CImageResizeLayer::GetDelta( TImageSide side ) const
{
	switch( side ) {
		case IS_Left:
			return deltaLeft;
		case IS_Right:
			return deltaRight;
		case IS_Top:
			return deltaTop;
		case IS_Bottom:
			return deltaBottom;
		default:
			NeoAssert( false );
	}
	return 0;
}

void CImageResizeLayer::SetDelta( TImageSide side, int delta )
{
	switch( side ) {
		case IS_Left:
			deltaLeft = delta;
			break;
		case IS_Right:
			deltaRight = delta;
			break;
		case IS_Top:
			deltaTop = delta;
			break;
		case IS_Bottom:
			deltaBottom = delta;
			break;
		default:
			NeoAssert( false );
	}
}

static const int ImageResizeLayerVersion = 2001;

void CImageResizeLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( ImageResizeLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( deltaLeft );
	archive.Serialize( deltaRight );
	archive.Serialize( deltaTop );
	archive.Serialize( deltaBottom );
	archive.Serialize( defaultValue );

	if( version > 2000 ) {
		archive.SerializeEnum( padding );
	} else if( archive.IsLoading() ) {
		padding = TBlobResizePadding::Constant;
	}
}

void CImageResizeLayer::Reshape()
{
	// Check the inputs
	CheckInputs();

	// Check that we are not trying to remove more pixels from any side than there are altogether
	CheckLayerArchitecture( deltaTop > -inputDescs[0].Height(), "deltaTop removes whole image" );
	CheckLayerArchitecture( deltaBottom > -inputDescs[0].Height(), "deltaBottom removes whole image" );
	CheckLayerArchitecture( deltaLeft > -inputDescs[0].Width(), "deltaLeft removes whole image" );
	CheckLayerArchitecture( deltaRight > -inputDescs[0].Width(), "deltaRight removes whole image" );

	// Check that we are not trying to remove more pixels from both sides than there are altogether
	CheckLayerArchitecture( inputDescs[0].Height() + deltaTop + deltaBottom > 0,
		"deltaTop + deltaBottom remove whole image" );
	CheckLayerArchitecture( inputDescs[0].Width() + deltaLeft + deltaRight > 0,
		"deltaLeft + deltaRight remove whole image" );

	if( IsBackwardPerformed() ) {
		CheckLayerArchitecture( padding == TBlobResizePadding::Constant,
			"backward with non-Constant padding" );
	}

	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize( BD_Height, outputDescs[0].Height() + deltaTop + deltaBottom );
	outputDescs[0].SetDimSize( BD_Width, outputDescs[0].Width() + deltaLeft + deltaRight );
}

void CImageResizeLayer::RunOnce()
{
	MathEngine().BlobResizeImage( inputBlobs[0]->GetDesc(), inputBlobs[0]->GetData(), deltaLeft, deltaRight,
		deltaTop, deltaBottom, padding, defaultValue, outputBlobs[0]->GetDesc(), outputBlobs[0]->GetData() );
}

void CImageResizeLayer::BackwardOnce()
{
	MathEngine().BlobResizeImage( outputDiffBlobs[0]->GetDesc(), outputDiffBlobs[0]->GetData(),
		-deltaLeft, -deltaRight, -deltaTop, -deltaBottom, padding, 0.f,
		inputDiffBlobs[0]->GetDesc(), inputDiffBlobs[0]->GetData() );
}

CLayerWrapper<CImageResizeLayer> ImageResize( int deltaLeft, int deltaRight, int deltaTop,
	int deltaBottom, float defaultValue, TBlobResizePadding padding )
{
	return CLayerWrapper<CImageResizeLayer>( "ImageResize", [=]( CImageResizeLayer* result ) {
		result->SetDelta( CImageResizeLayer::IS_Left, deltaLeft );
		result->SetDelta( CImageResizeLayer::IS_Right, deltaRight );
		result->SetDelta( CImageResizeLayer::IS_Bottom, deltaBottom );
		result->SetDelta( CImageResizeLayer::IS_Top, deltaTop );
		result->SetDefaultValue( defaultValue );
		result->SetPadding( padding );
	} );
}

} // namespace NeoML
