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

#include <NeoML/Dnn/Rowwise/Pooling.h>
#include <NeoML/Dnn/Layers/PoolingLayer.h>

namespace NeoML {

CRowwise2DPooling::CRowwise2DPooling( const CMaxPoolingLayer& layer ) :
	mathEngine( layer.MathEngine() ),
	isMax( true ),
	filterHeight( layer.GetFilterHeight() ),
	filterWidth( layer.GetFilterWidth() ),
	strideHeight( layer.GetStrideHeight() ),
	strideWidth( layer.GetStrideWidth() )
{
}

CRowwise2DPooling::CRowwise2DPooling( const CMeanPoolingLayer& layer ) :
	mathEngine( layer.MathEngine() ),
	isMax( false ),
	filterHeight( layer.GetFilterHeight() ),
	filterWidth( layer.GetFilterWidth() ),
	strideHeight( layer.GetStrideHeight() ),
	strideWidth( layer.GetStrideWidth() )
{
}

CRowwise2DPooling::CRowwise2DPooling( IMathEngine& mathEngine ) :
	mathEngine( mathEngine ),
	isMax( false ),
	filterHeight( 0 ),
	filterWidth( 0 ),
	strideHeight( 0 ),
	strideWidth( 0 )
{
}

CRowwiseOperationDesc* CRowwise2DPooling::GetDesc()
{
	return mathEngine.InitRowwise2DPooling( isMax, filterHeight, filterWidth, strideHeight, strideWidth );
}

void CRowwise2DPooling::Serialize( CArchive& archive )
{
	( void ) archive.SerializeVersion( 0 );
	archive.Serialize( isMax );
	archive.Serialize( filterHeight );
	archive.Serialize( filterWidth );
	archive.Serialize( strideHeight );
	archive.Serialize( strideWidth );
}

REGISTER_NEOML_ROWWISE_OPERATION( CRowwise2DPooling, "Rowwise2DPoolingOperation" )

} // namespace NeoML
