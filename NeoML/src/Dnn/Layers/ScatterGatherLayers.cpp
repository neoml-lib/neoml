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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/ScatterGatherLayers.h>

namespace NeoML {

CScatterNDLayer::CScatterNDLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CScatterNDLayer", false )
{
}

static const int ScatterNDLayerVersion = 0;

void CScatterNDLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ScatterNDLayerVersion );
	CBaseLayer::Serialize( archive );
}

void CScatterNDLayer::Reshape()
{
	CheckArchitecture( GetInputCount() == 3, GetName(), "Layer must have 3 inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetName(), "Layer must have 1 output" );
	CheckArchitecture( inputDescs[I_Data].GetDataType() == inputDescs[I_Updates].GetDataType(), GetName(),
		"Data and updates must have similar data types" );
	CheckArchitecture( inputDescs[I_Indices].GetDataType() == CT_Int, GetName(), "Indices must be integer" );

	const int indexDims = inputDescs[I_Indices].Channels();
	const int updateCount = inputDescs[I_Indices].BlobSize() / indexDims;
	CheckArchitecture( inputDescs[I_Updates].BlobSize() % updateCount == 0, GetName(),
		"Updates must contain UpdateCount x ObjectSize elemnts" );
	const int objectSize = inputDescs[I_Updates].BlobSize() / updateCount;
	CheckArchitecture( inputDescs[I_Data].BlobSize() % objectSize == 0, GetName(),
		"Data must containt ObjectCount x ObjectSize elements" );
	int actualObjectSize = 1;
	for( int dim = indexDims; dim < static_cast<int>( BD_Count ); ++dim ) {
		actualObjectSize *= inputDescs[I_Data].DimSize( dim );
	}
	CheckArchitecture( actualObjectSize == objectSize, GetName(),
		"Last (BD_Count - N) dimensions of Data blob must have product of ObjectSize" );
	outputDescs[0] = inputDescs[I_Data];
}

void CScatterNDLayer::RunOnce()
{
	const int indexDims = inputDescs[I_Indices].Channels();
	const int updateCount = inputDescs[I_Indices].BlobSize() / indexDims;

	outputBlobs[0]->CopyFrom( inputBlobs[I_Data] );
	if( outputBlobs[0]->GetDataType() == CT_Float ) {
		MathEngine().ScatterND( inputBlobs[I_Indices]->GetData<int>(), inputBlobs[I_Updates]->GetData(),
			outputBlobs[0]->GetData(), outputBlobs[0]->GetDesc(), updateCount, indexDims );
	} else {
		MathEngine().ScatterND( inputBlobs[I_Indices]->GetData<int>(), inputBlobs[I_Updates]->GetData<int>(),
			outputBlobs[0]->GetData<int>(), outputBlobs[0]->GetDesc(), updateCount, indexDims );
	}
}

void CScatterNDLayer::BackwardOnce()
{
	NeoAssert( false );
}

} // namespace NeoML
