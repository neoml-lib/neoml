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

#include <NeoML/Dnn/Layers/EnumBinarizationLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CEnumBinarizationLayer::CEnumBinarizationLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnEnumBinarizationLayer", false ),
	enumSize( 1 )
{
}

void CEnumBinarizationLayer::SetEnumSize(int _enumSize)
{
	if(enumSize == _enumSize) {
		return;
	}
	enumSize = _enumSize;
	ForceReshape();
}

static const int EnumBinarizationLayerVersion = 2000;

void CEnumBinarizationLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( EnumBinarizationLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize(enumSize);
}

void CEnumBinarizationLayer::Reshape()
{
	CheckInput1();
	CheckArchitecture(inputDescs[0].Channels() == 1,
		GetName(), "Enum binarization lookup layer must have input with size BATCHxHxWxDx1");

	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDataType( CT_Float );
	outputDescs[0].SetDimSize(BD_Channels, enumSize);
}

void CEnumBinarizationLayer::RunOnce()
{
	if(inputBlobs[0]->GetDataType() == CT_Float) {
		MathEngine().EnumBinarization(inputBlobs[0]->GetObjectCount() * inputBlobs[0]->GetHeight() * inputBlobs[0]->GetWidth(),
			inputBlobs[0]->GetData(), enumSize, outputBlobs[0]->GetData());
	} else {
		MathEngine().EnumBinarization( inputBlobs[0]->GetObjectCount() * inputBlobs[0]->GetHeight() * inputBlobs[0]->GetWidth(),
			inputBlobs[0]->GetData<int>(), enumSize, outputBlobs[0]->GetData() );
	}
}

void CEnumBinarizationLayer::BackwardOnce()
{
}

CLayerWrapper<CEnumBinarizationLayer> EnumBinarization( int enumSize )
{
	return CLayerWrapper<CEnumBinarizationLayer>( "EnumBinarization", [=]( CEnumBinarizationLayer* result ) {
		result->SetEnumSize( enumSize );
	} );
}

//-------------------------------------------------------------------------------------------------

CBitSetVectorizationLayer::CBitSetVectorizationLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CnnBitSetVectorizationLayerClassName", false ),
	bitSetSize( 1 )
{
}

void CBitSetVectorizationLayer::SetBitSetSize( int _bitSetSize )
{
	if( bitSetSize == _bitSetSize) {
		return;
	}
	bitSetSize = _bitSetSize;
	ForceReshape();
}

static const int BitSetVectorizationLayerVersion = 2000;

void CBitSetVectorizationLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( BitSetVectorizationLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( bitSetSize );
}

void CBitSetVectorizationLayer::Reshape()
{
	CheckInput1();
	CheckArchitecture( inputDescs[0].GetDataType() == CT_Int, GetName(),
		"Bitset vectorization layer must have integer input" );

	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize(BD_Channels, bitSetSize);
	outputDescs[0].SetDataType( CT_Float );
}

void CBitSetVectorizationLayer::RunOnce()
{
	int size = inputBlobs[0]->GetObjectCount() * inputBlobs[0]->GetGeometricalSize();
	MathEngine().BitSetBinarization( size, inputBlobs[0]->GetChannelsCount(), inputBlobs[0]->GetData<int>(),
		outputBlobs[0]->GetChannelsCount(), outputBlobs[0]->GetData<float>() );
}

void CBitSetVectorizationLayer::BackwardOnce()
{
	NeoAssert( false );
}

} // namespace NeoML
