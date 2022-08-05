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

// --------------------------------------------------------------------------------------------------------------------

CEqualLayer::CEqualLayer( IMathEngine& mathEngine ) :
	CEltwiseBaseLayer( mathEngine, "CEqualLayer" )
{
}

void CEqualLayer::Reshape()
{
	CheckArchitecture( inputDescs.Size() == 2, GetName(), "Equal operation expects 2 inputs" );
	CheckArchitecture( inputDescs[0].GetDataType() == inputDescs[1].GetDataType(), GetName(),
		"Inputs must be of the same data type" );

	CEltwiseBaseLayer::Reshape();
	outputDescs[0].SetDataType( CT_Int );
}

void CEqualLayer::RunOnce()
{
	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		MathEngine().VectorEltwiseEqual( inputBlobs[0]->GetData(), inputBlobs[1]->GetData(),
			outputBlobs[0]->GetData<int>(), inputBlobs[0]->GetDataSize() );
	} else {
		MathEngine().VectorEltwiseEqual( inputBlobs[0]->GetData<int>(), inputBlobs[1]->GetData<int>(),
			outputBlobs[0]->GetData<int>(), inputBlobs[0]->GetDataSize() );
	}
}

void CEqualLayer::BackwardOnce()
{
	NeoAssert( false );
}

static const int EqualLayerVersion = 0;

void CEqualLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( EqualLayerVersion );
	CEltwiseBaseLayer::Serialize( archive );
}

// --------------------------------------------------------------------------------------------------------------------

CWhereLayer::CWhereLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CWhereLayer", false )
{
}

static const int WhereLayerVersion = 0;

void CWhereLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( WhereLayerVersion );
	CBaseLayer::Serialize( archive );
}

void CWhereLayer::Reshape()
{
	CheckArchitecture( inputDescs.Size() == 3, GetName(), "Layer expects 3 inputs" );
	CheckArchitecture( inputDescs[0].GetDataType() == CT_Int, GetName(), "First input must be integer" );
	CheckArchitecture( inputDescs[1].HasEqualDimensions( inputDescs[0] ), GetName(),
		"Second input size must match with the first" );
	CheckArchitecture( inputDescs[2].HasEqualDimensions( inputDescs[0] ), GetName(),
		"Third input size must match with the first" );
	CheckArchitecture( inputDescs[1].GetDataType() == inputDescs[2].GetDataType(), GetName(),
		"Data type mismatch between the second and the third inputs" );
	CheckArchitecture( outputDescs.Size() == 1, GetName(), "Layer expects 1 output" );
	outputDescs[0] = inputDescs[1];
}

void CWhereLayer::RunOnce()
{
	if( inputBlobs[1]->GetDataType() == CT_Float ) {
		MathEngine().VectorEltwiseWhere( inputBlobs[0]->GetData<int>(), inputBlobs[1]->GetData(),
			inputBlobs[2]->GetData(), outputBlobs[0]->GetData(), inputBlobs[0]->GetDataSize() );
	} else {
		MathEngine().VectorEltwiseWhere( inputBlobs[0]->GetData<int>(), inputBlobs[1]->GetData<int>(),
			inputBlobs[2]->GetData<int>(), outputBlobs[0]->GetData<int>(), inputBlobs[0]->GetDataSize() );
	}
}

void CWhereLayer::BackwardOnce()
{
	NeoAssert( false );
}

} // namespace NeoML
