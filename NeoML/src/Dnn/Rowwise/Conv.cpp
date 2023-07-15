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

#include <NeoML/Dnn/Rowwise/Conv.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>

namespace NeoML {

CRowwiseConv::CRowwiseConv( const CConvLayer& convLayer ) :
	mathEngine( convLayer.MathEngine() ),
	paddingHeight( convLayer.GetPaddingHeight() ),
	paddingWidth( convLayer.GetPaddingWidth() ),
	strideHeight( convLayer.GetStrideHeight() ),
	strideWidth( convLayer.GetStrideWidth() ),
	dilationHeight( convLayer.GetDilationHeight() ),
	dilationWidth( convLayer.GetDilationWidth() ),
	filter( convLayer.GetFilterData() ),
	freeTerm( convLayer.GetFreeTermData() )
{
}

CRowwiseConv::CRowwiseConv( IMathEngine& mathEngine ) :
	mathEngine( mathEngine ),
	paddingHeight( 0 ),
	paddingWidth( 0 ),
	strideHeight( 0 ),
	strideWidth( 0 ),
	dilationHeight( 0 ),
	dilationWidth( 0 )
{
}

CRowwiseOperationDesc* CRowwiseConv::GetDesc()
{
	CConstFloatHandle freeTermData = freeTerm == nullptr ? CConstFloatHandle() : freeTerm->GetData<const float>();
	return mathEngine.InitRowwiseConv( paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight,
		dilationWidth, filter->GetDesc(), filter->GetData(),
		freeTerm == nullptr ? nullptr : &freeTermData );
}

void CRowwiseConv::Serialize( CArchive& archive )
{
	(void) archive.SerializeVersion( 0 ); // version
	archive.Serialize( paddingHeight );
	archive.Serialize( paddingWidth );
	archive.Serialize( strideHeight );
	archive.Serialize( strideWidth );
	archive.Serialize( dilationHeight );
	archive.Serialize( dilationWidth );
	SerializeBlob( mathEngine, archive, filter );
	SerializeBlob( mathEngine, archive, freeTerm );
}

REGISTER_NEOML_ROWWISE_OPERATION( CRowwiseConv, "RowwiseConvOperation" )

} // namespace NeoML
