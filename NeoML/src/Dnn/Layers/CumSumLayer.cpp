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

#include <NeoML/Dnn/Layers/CumSumLayer.h>

namespace NeoML {

static void getDimSizes( const CBlobDesc& desc, TBlobDim dim,
	int& preceding, int& dimension, int& following )
{
	static_assert( BD_Count == 7, "BD_Count != 7" );

	dimension = desc.DimSize( dim );

	following = 1;
	for( TBlobDim i = BD_BatchLength; i != dim; ++i ) {
		following *= desc.DimSize( i );
	}

	preceding = 1;
	for( TBlobDim i = dim + 1; i != BD_Count; ++i ) {
		preceding *= desc.DimSize( i );
	}
}

CCumSumLayer::CCumSumLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCumSumLayer", false ),
	dim( BD_Channels ),
	reverse( false )
{
}

static const int CumSumLayerVersion = 0;

void CCumSumLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CumSumLayerVersion );
	CBaseLayer::Serialize( archive );
	archive.SerializeEnum( dim );
	archive.Serialize( reverse );
}

void CCumSumLayer::Reshape()
{
	NeoAssert( dim >= 0 && dim < BD_Count );
	inputDescs.CopyTo( outputDescs );
	CheckArchitecture( inputDescs[0].GetDataType() == CT_Float || !IsBackwardPerformed(),
		GetName(), "Backward over integer data" );
}

void CCumSumLayer::RunOnce()
{
	int preceding = 1;
	int dimension = 1;
	int following = 1;
	getDimSizes( inputBlobs[0]->GetDesc(), dim, preceding, dimension, following );

	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		MathEngine().VectorCumSumAlongDimension( inputBlobs[0]->GetData(),
			preceding, dimension, following, outputBlobs[0]->GetData(), reverse );
	} else {
		MathEngine().VectorCumSumAlongDimension( inputBlobs[0]->GetData<int>(),
			preceding, dimension, following, outputBlobs[0]->GetData<int>(), reverse );
	}
}

void CCumSumLayer::BackwardOnce()
{
	int preceding = 1;
	int dimension = 1;
	int following = 1;
	getDimSizes( inputBlobs[0]->GetDesc(), dim, preceding, dimension, following );

	MathEngine().VectorCumSumAlongDimension( outputDiffBlobs[0]->GetData(),
		preceding, dimension, following, inputDiffBlobs[0]->GetData(), !reverse );
}

CLayerWrapper<CCumSumLayer> CumSum( TBlobDim dim, bool reverse )
{
	return CLayerWrapper<CCumSumLayer>( "CumSum", [=]( CCumSumLayer* result ) {
		result->SetDimension( dim );
		result->SetReverse( reverse );
	} );
}

} // namespace NeoML