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

#include <NeoML/Dnn/Layers/GlobalMaxPoolingLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CGlobalMaxPoolingLayer::CGlobalMaxPoolingLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnGlobalMaxPoolingLayer", false ),
	desc( 0 ),
	maxCount( 1 )
{
}

void CGlobalMaxPoolingLayer::SetMaxCount(int _maxCount)
{
	if( maxCount == _maxCount ) {
		return;
	}
	maxCount = _maxCount;
	ForceReshape();
}

static const int GlobalMaxPoolingLayerVersion = 2000;

void CGlobalMaxPoolingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( GlobalMaxPoolingLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( maxCount );
}

void CGlobalMaxPoolingLayer::Reshape()
{
	CheckInputs();

	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize( BD_Height, 1 );
	outputDescs[0].SetDimSize( BD_Width, maxCount );
	outputDescs[0].SetDimSize( BD_Depth, 1 );

	if(GetOutputCount() > 1) {
		// Write the index of the maximum into the second output
		outputDescs[1] = outputDescs[0];
		outputDescs[1].SetDataType( CT_Int );
		indexBlob = CDnnBlob::CreateBlob( MathEngine(), outputDescs[1] );
	} else {
		indexBlob = CDnnBlob::CreateBlob( MathEngine(), CT_Int, outputDescs[0] );
	}

	RegisterRuntimeBlob(indexBlob);
	destroyDesc();
}

void CGlobalMaxPoolingLayer::RunOnce()
{
	initDesc();

	MathEngine().BlobGlobalMaxPooling( *desc, inputBlobs[0]->GetData(), indexBlob->GetData<int>(),
		outputBlobs[0]->GetData() );
}

void CGlobalMaxPoolingLayer::BackwardOnce()
{
	initDesc();

	MathEngine().BlobGlobalMaxPoolingBackward( *desc, outputDiffBlobs[0]->GetData(), indexBlob->GetData<int>(),
		inputDiffBlobs[0]->GetData() );
}

void CGlobalMaxPoolingLayer::initDesc()
{
	if( desc == 0 ) {
		desc = MathEngine().InitGlobalMaxPooling( inputBlobs[0]->GetDesc(), indexBlob->GetDesc(), outputBlobs[0]->GetDesc() );
	}
}

void CGlobalMaxPoolingLayer::destroyDesc()
{
	if( desc != 0 ) {
		delete desc;
		desc = 0;
	}
}

CLayerWrapper<CGlobalMaxPoolingLayer> GlobalMaxPooling( int maxCount )
{
	return CLayerWrapper<CGlobalMaxPoolingLayer>( "", [=]( CGlobalMaxPoolingLayer* result ) {
		result->SetMaxCount( maxCount );
	} );
}

} // namespace NeoML
