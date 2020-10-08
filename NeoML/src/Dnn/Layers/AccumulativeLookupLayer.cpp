/* Copyright © 2017-2020 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/AccumulativeLookupLayer.h>

namespace NeoML {

CAccumulativeLookupLayer::CAccumulativeLookupLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnAccumulativeLookupLayer", true )
{
	paramBlobs.SetSize( 1 );
}

void CAccumulativeLookupLayer::SetDimension( const CLookupDimension& newDimension )
{
	NeoAssert( newDimension.VectorCount > 0 );
	NeoAssert( newDimension.VectorSize > 0 );
	lookupDimension = newDimension;
}

void CAccumulativeLookupLayer::SetEmbeddings( const CPtr<CDnnBlob>& newEmbeddings )
{
	NeoAssert( newEmbeddings != 0 );
	
	NeoAssert( newEmbeddings->DimSize(0) == lookupDimension.VectorCount );
	NeoAssert( newEmbeddings->DimSize(1) == lookupDimension.VectorSize );

	paramBlobs[0] = newEmbeddings->GetCopy();
}

void CAccumulativeLookupLayer::Reshape()
{
	CheckInput1();

	CheckArchitecture( inputDescs[0].GetDataType() == CT_Int,
		GetName(), "CCnnAccumulativeLookupLayer must have integer input" );

	if( paramBlobs[0] == 0
		|| paramBlobs[0]->DimSize(0) != lookupDimension.VectorCount
		|| paramBlobs[0]->DimSize(1) != lookupDimension.VectorSize )
	{
		paramBlobs[0] = CDnnBlob::CreateMatrix( MathEngine(), CT_Float, lookupDimension.VectorCount, 
			lookupDimension.VectorSize );
		InitializeParamBlob( 0, *paramBlobs[0] );
	}

	outputDescs[0] = CBlobDesc( CT_Float );
	outputDescs[0].SetDimSize( BD_BatchLength, inputDescs[0].BatchLength() );
	outputDescs[0].SetDimSize( BD_BatchWidth, inputDescs[0].BatchWidth() );
	outputDescs[0].SetDimSize( BD_ListSize, inputDescs[0].ListSize() );
	outputDescs[0].SetDimSize( BD_Channels, lookupDimension.VectorSize );
}

void CAccumulativeLookupLayer::RunOnce()
{
	MathEngine().LookupAndSum( inputBlobs[0]->GetData<int>(), inputBlobs[0]->GetObjectCount(),
		inputBlobs[0]->GetObjectSize(), paramBlobs[0]->GetData(), lookupDimension.VectorSize,
		outputBlobs[0]->GetData() );
}

void CAccumulativeLookupLayer::BackwardOnce()
{
	NeoAssert( false );
}

void CAccumulativeLookupLayer::LearnOnce()
{
	MathEngine().LookupAndAddToTable( inputBlobs[0]->GetData<int>(), inputBlobs[0]->GetObjectCount(),
		inputBlobs[0]->GetObjectSize(), outputDiffBlobs[0]->GetData(), lookupDimension.VectorSize,
		paramDiffBlobs[0]->GetData(), lookupDimension.VectorCount );
}

static const int AccumulativeLookupLayerVersion = 2000;

void CAccumulativeLookupLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( AccumulativeLookupLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( lookupDimension.VectorCount );
	archive.Serialize( lookupDimension.VectorSize );
}

NEOML_API CLayerWrapper<NeoML::CAccumulativeLookupLayer> AccumulativeLookup(
	int count, int size )
{
	return CLayerWrapper<CAccumulativeLookupLayer>( "AccumulativeLookup", [=]( CAccumulativeLookupLayer* result ) {
		return result->SetDimension( CLookupDimension( count, size ) );
	} );
}

} // namespace NeoML
