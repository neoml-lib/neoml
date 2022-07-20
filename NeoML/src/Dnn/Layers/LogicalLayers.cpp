/* Copyright © 2017-2022 ABBYY Production LLC

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

#pragma once

#include <NeoML/Dnn/Layers/LogicalLayers.h>

namespace NeoML {

CNotLayer::CNotLayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CNotLayer" )
{
}

void CNotLayer::OnReshaped()
{
	CheckArchitecture( inputDescs[0].GetDataType() == CT_Int, GetName(), "layer works only with integer data" );
}

void CNotLayer::RunOnce()
{
	MathEngine().VectorEltwiseNot( inputBlobs[0]->GetData<int>(), outputBlobs[0]->GetData<int>(),
		outputBlobs[0]->GetDataSize() );
}

void CNotLayer::BackwardOnce()
{
	NeoAssert( false );
}

static const int NotLayerVersion = 0;

void CNotLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( NotLayerVersion );
	CBaseInPlaceLayer::Serialize( archive );
}

CLayerWrapper<CNotLayer> Not()
{
	return CLayerWrapper<CNotLayer>( "Not" );
}

// --------------------------------------------------------------------------------------------------------------------

CLessLayer::CLessLayer( IMathEngine& mathEngine ) :
	CEltwiseBaseLayer( mathEngine, "CLessLayer" )
{
}

void CLessLayer::Reshape()
{
	CheckArchitecture( inputDescs.Size() == 2, GetName(), "Less operation expects 2 inputs" );
	CheckArchitecture( inputDescs[0].GetDataType() == inputDescs[1].GetDataType(), GetName(),
		"Inputs must be of the same data type" );

	CEltwiseBaseLayer::Reshape();

	outputDescs[0].SetDataType( CT_Int );
}

void CLessLayer::RunOnce()
{
	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		MathEngine().VectorEltwiseLess( inputBlobs[0]->GetData(), inputBlobs[1]->GetData(),
			outputBlobs[0]->GetData<int>(), inputBlobs[0]->GetDataSize() );
	} else {
		MathEngine().VectorEltwiseLess( inputBlobs[0]->GetData<int>(), inputBlobs[1]->GetData<int>(),
			outputBlobs[0]->GetData<int>(), inputBlobs[0]->GetDataSize() );
	}
}

void CLessLayer::BackwardOnce()
{
	NeoAssert( false );
}

static const int LessLayerVersion = 0;

void CLessLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( LessLayerVersion );
	CEltwiseBaseLayer::Serialize( archive );
}

} // namespace NeoML
