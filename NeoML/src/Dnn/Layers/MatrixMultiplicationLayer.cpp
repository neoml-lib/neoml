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

#include <NeoML/Dnn/Layers/MatrixMultiplicationLayer.h>

namespace NeoML {

CMatrixMultiplicationLayer::CMatrixMultiplicationLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CMatrixMultiplicationLayer", false )
{
}

static const int MatrixMultiplicationLayerVersion = 0;

void CMatrixMultiplicationLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MatrixMultiplicationLayerVersion );
	CBaseLayer::Serialize( archive );
}


void CMatrixMultiplicationLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( inputDescs.Size() == 2, GetName(), "layer must have 2 inputs" );

	CheckArchitecture( inputDescs[0].Channels() == inputDescs[1].GeometricalSize(), GetName(),
		"input[0].Channels must be equal to input[1].GeometricalSize" );
	CheckArchitecture( inputDescs[0].ObjectCount() == inputDescs[1].ObjectCount(), GetName(), "object count mismatch between inputs" );

	outputDescs.SetSize( 1 );
	CBlobDesc outputDesc = inputDescs[0];
	outputDesc.SetDimSize( BD_Channels, inputDescs[1].Channels() );
	outputDescs[0] = outputDesc;
}

void CMatrixMultiplicationLayer::RunOnce()
{
	MathEngine().MultiplyMatrixByMatrix( inputBlobs[0]->GetObjectCount(), inputBlobs[0]->GetData(),
		inputBlobs[0]->GetGeometricalSize(), inputBlobs[0]->GetChannelsCount(), inputBlobs[1]->GetData(),
		inputBlobs[1]->GetChannelsCount(), outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize() );
}

void CMatrixMultiplicationLayer::BackwardOnce()
{
	NeoAssert( outputDiffBlobs[0]->GetChannelsCount() == inputBlobs[1]->GetChannelsCount() );
	NeoAssert( outputDiffBlobs[0]->GetGeometricalSize() == inputBlobs[0]->GetGeometricalSize() );

	MathEngine().MultiplyMatrixByTransposedMatrix( inputBlobs[0]->GetObjectCount(),
		outputDiffBlobs[0]->GetData(), outputDiffBlobs[0]->GetGeometricalSize(),
		outputDiffBlobs[0]->GetChannelsCount(), inputBlobs[1]->GetData(),
		inputBlobs[1]->GetGeometricalSize(), inputDiffBlobs[0]->GetData(), 
		inputDiffBlobs[0]->GetDataSize() );

	MathEngine().MultiplyTransposedMatrixByMatrix( inputBlobs[0]->GetObjectCount(),
		inputBlobs[0]->GetData(), inputBlobs[0]->GetGeometricalSize(), inputBlobs[0]->GetChannelsCount(),
		outputDiffBlobs[0]->GetData(), outputDiffBlobs[0]->GetChannelsCount(),
		inputDiffBlobs[1]->GetData(), inputDiffBlobs[1]->GetDataSize() );
}

} // namespace NeoML
