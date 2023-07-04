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

#include <NeoML/Dnn/Rowwise/ImageResize.h>
#include <NeoML/Dnn/Layers/ImageResizeLayer.h>

namespace NeoML {

CRowwiseImageResize::CRowwiseImageResize( const CImageResizeLayer& layer ) :
	mathEngine( layer.MathEngine() ),
	padding( layer.GetPadding() ),
	defaultValue( layer.GetDefaultValue() ),
	deltaLeft( layer.GetDelta( CImageResizeLayer::IS_Left ) ),
	deltaRight( layer.GetDelta( CImageResizeLayer::IS_Right ) ),
	deltaTop( layer.GetDelta( CImageResizeLayer::IS_Top ) ),
	deltaBottom( layer.GetDelta( CImageResizeLayer::IS_Bottom ) )
{
}

CRowwiseImageResize::CRowwiseImageResize( IMathEngine& mathEngine ) :
	mathEngine( mathEngine ),
	padding( TBlobResizePadding::Constant ),
	defaultValue( 0.f ),
	deltaLeft( 0 ),
	deltaRight( 0 ),
	deltaTop( 0 ),
	deltaBottom( 0 )
{
}

CRowwiseOperationDesc* CRowwiseImageResize::GetDesc( const CBlobDesc& inputDesc )
{
	return mathEngine.InitRowwiseResizeImage( padding, defaultValue, deltaLeft, deltaRight, deltaTop, deltaBottom );
}

void CRowwiseImageResize::Serialize( CArchive& archive )
{
	( void ) archive.SerializeVersion( 0 );
	archive.SerializeEnum( padding );
	archive.Serialize( defaultValue );
	archive.Serialize( deltaLeft );
	archive.Serialize( deltaRight );
	archive.Serialize( deltaTop );
	archive.Serialize( deltaBottom );
}

// Registration
REGISTER_NEOML_ROWWISE_OPERATION( CRowwiseImageResize, "RowwiseImageResizeOperation" )

} // namespace NeoML
