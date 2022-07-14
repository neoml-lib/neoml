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

#include <NeoML/Dnn/Layers/DotProductLayer.h>

namespace NeoML {

CDotProductLayer::CDotProductLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "DotProductLayer", false )
{}

void CDotProductLayer::Reshape()
{
	CheckInputs();

	CheckArchitecture( GetInputCount() == 2, GetName(), "layer must have 2 inputs" );
	CheckArchitecture( inputDescs[0].HasEqualDimensions( inputDescs[1] ), GetName(), "input blobs size mismatch" );
	CheckArchitecture( inputDescs[0].GetDataType() == CT_Float && inputDescs[1].GetDataType() == CT_Float,
		GetName(), "layer supports only float blobs" );

	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize( BD_Channels, 1 );
	outputDescs[0].SetDimSize( BD_Depth, 1 );
	outputDescs[0].SetDimSize( BD_Height, 1 );
	outputDescs[0].SetDimSize( BD_Width, 1 );
}

void CDotProductLayer::RunOnce()
{
	MathEngine().RowMultiplyMatrixByMatrix( inputBlobs[0]->GetData(), inputBlobs[1]->GetData(),
		inputBlobs[0]->GetObjectCount(), inputBlobs[0]->GetObjectSize(), outputBlobs[0]->GetData() );
}

void CDotProductLayer::BackwardOnce()
{
	MathEngine().MultiplyDiagMatrixByMatrix( outputDiffBlobs[0]->GetData(), outputDiffBlobs[0]->GetDataSize(),
		inputBlobs[0]->GetData(), inputBlobs[0]->GetObjectSize(), inputDiffBlobs[1]->GetData(),
		inputDiffBlobs[1]->GetDataSize() );
	MathEngine().MultiplyDiagMatrixByMatrix( outputDiffBlobs[0]->GetData(), outputDiffBlobs[0]->GetDataSize(),
		inputBlobs[1]->GetData(), inputBlobs[1]->GetObjectSize(), inputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetDataSize() );
}

static const int DotProductLayerVersion = 2000;

void CDotProductLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( DotProductLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

CLayerWrapper<CDotProductLayer> DotProduct()
{
	return CLayerWrapper<CDotProductLayer>( "DotProduct" );
}

} // namespace NeoML
