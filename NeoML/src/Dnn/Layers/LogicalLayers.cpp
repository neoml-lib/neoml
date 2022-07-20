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

} // namespace NeoML
