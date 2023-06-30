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

#include <NeoML/Dnn/Rowwise/ChannelwiseConv.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>

namespace NeoML {

CRowwiseChConv::CRowwiseChConv( const CChannelwiseConvLayer& chConvLayer ) :
	mathEngine( chConvLayer.MathEngine() ),
	paddingHeight( chConvLayer.GetPaddingHeight() ),
	paddingWidth( chConvLayer.GetPaddingWidth() ),
	strideHeight( chConvLayer.GetStrideHeight() ),
	strideWidth( chConvLayer.GetStrideWidth() ),
	filter( chConvLayer.GetFilterData() ),
	freeTerm( chConvLayer.GetFreeTermData() )
{
}

CRowwiseChConv::CRowwiseChConv( IMathEngine& mathEngine ) :
	mathEngine( mathEngine ),
	paddingHeight( 0 ),
	paddingWidth( 0 ),
	strideHeight( 0 ),
	strideWidth( 0 )
{
}

CRowwiseOperationDesc* CRowwiseChConv::GetDesc( const CBlobDesc& inputDesc )
{
	return mathEngine.InitRowwiseChConv( paddingHeight, paddingWidth, strideHeight, strideWidth,
        filter->GetDesc(), filter->GetData(),
		freeTerm == nullptr ? nullptr : &freeTerm->GetData<const float>() );
}

void CRowwiseChConv::Serialize( CArchive& archive )
{
	(void) archive.SerializeVersion( 0 ); // version
	archive.Serialize( paddingHeight );
	archive.Serialize( paddingWidth );
	archive.Serialize( strideHeight );
	archive.Serialize( strideWidth );
	SerializeBlob( mathEngine, archive, filter );
	SerializeBlob( mathEngine, archive, freeTerm );
}

REGISTER_NEOML_ROWWISE_OPERATION( CRowwiseChConv, "RowwiseChConvOperation" )

} // namespace NeoML
