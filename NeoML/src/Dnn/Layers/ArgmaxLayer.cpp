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

#include <NeoML/Dnn/Layers/ArgmaxLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CArgmaxLayer::CArgmaxLayer( IMathEngine& mathEngine ) :
	CBaseLayer(mathEngine, "CCnnArgmaxLayer", false),
	dimension(BD_Channels)
{
}

static const int ArgmaxLayerVersion = 2000;

void CArgmaxLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ArgmaxLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.SerializeEnum(dimension);
}

void CArgmaxLayer::SetDimension(TBlobDim d) 
{
	if(dimension == d) {
		return;
	}
	dimension = d; 
	ForceReshape();
}

void CArgmaxLayer::Reshape()
{
	CheckInput1();
	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDataType( CT_Int );
	outputDescs[0].SetDimSize(dimension, 1);
}

void CArgmaxLayer::RunOnce()
{
	CBlobDesc blobDesc = inputBlobs[0]->GetDesc();
	int dimIndex = dimension;
	int batchSize = 1;
	for(int i = 0; i < dimIndex; i++) {
		batchSize *= blobDesc.DimSize(i);
	}
	int rowSize = 1;
	for(int i = dimIndex + 1; i < CBlobDesc::MaxDimensions; i++) {
		rowSize *= blobDesc.DimSize(i);
	}
	CFloatHandleStackVar maxValues( MathEngine(), outputBlobs[0]->GetDataSize());
	MathEngine().FindMaxValueInColumns(batchSize, 
		inputBlobs[0]->GetData(), blobDesc.DimSize(dimension), rowSize, 
		maxValues, outputBlobs[0]->GetData<int>(), outputBlobs[0]->GetDataSize());
}

void CArgmaxLayer::BackwardOnce()
{
}

CLayerWrapper<CArgmaxLayer> Argmax( TBlobDim dim )
{
	return CLayerWrapper<CArgmaxLayer>( "Argmax", [=]( CArgmaxLayer* result ) {
		result->SetDimension( dim );
	} );
}

} // namespace NeoML
