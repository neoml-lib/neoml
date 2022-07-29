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

#include <NeoML/Dnn/Layers/GlobalSumPoolingLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CGlobalSumPoolingLayer::CGlobalSumPoolingLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CGlobalSumPoolingLayer", false )
{
}

static const int GlobalSumPoolingLayerVersion = 0;

void CGlobalSumPoolingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( GlobalSumPoolingLayerVersion );
	CBaseLayer::Serialize( archive );
}

void CGlobalSumPoolingLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( GetInputCount() == 1, GetName(), "multiple inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetName(), "multiple outputs" );

	NeoAssert( inputDescs.Size() == 1 );
	const CBlobDesc& inputDesc = inputDescs[0];

	outputDescs[0] = inputDesc;
	outputDescs[0].SetDimSize( BD_Height, 1 );
	outputDescs[0].SetDimSize( BD_Width, 1 );
	outputDescs[0].SetDimSize( BD_Depth, 1 );
}

void CGlobalSumPoolingLayer::RunOnce()
{
	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		MathEngine().SumMatrixRows( inputBlobs[0]->GetObjectCount(), outputBlobs[0]->GetData(),
			inputBlobs[0]->GetData(), inputBlobs[0]->GetGeometricalSize(), inputBlobs[0]->GetChannelsCount() );
	} else {
		MathEngine().SumMatrixRows( inputBlobs[0]->GetObjectCount(), outputBlobs[0]->GetData<int>(),
			inputBlobs[0]->GetData<int>(), inputBlobs[0]->GetGeometricalSize(), inputBlobs[0]->GetChannelsCount() );
	}
}

void CGlobalSumPoolingLayer::BackwardOnce()
{
	MathEngine().VectorFill( inputDiffBlobs[0]->GetData(), 0.0f, inputDiffBlobs[0]->GetDataSize() );
	MathEngine().AddVectorToMatrixRows( inputDiffBlobs[0]->GetObjectCount(), inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetGeometricalSize(), inputDiffBlobs[0]->GetChannelsCount(), outputDiffBlobs[0]->GetData() );
}

CLayerWrapper<CGlobalSumPoolingLayer> GlobalSumPooling()
{
	return CLayerWrapper<CGlobalSumPoolingLayer>( "GlobalSumPooling" );
}

} // namespace NeoML
