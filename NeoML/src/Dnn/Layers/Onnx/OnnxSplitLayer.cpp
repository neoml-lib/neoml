/* Copyright © 2017-2023 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/Onnx/OnnxSplitLayer.h>

namespace NeoML {

static const int OnnxSplitLayerVersion = 0;

void COnnxSplitLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxSplitLayerVersion );
	COnnxLayerBase::Serialize( archive );
	archive.SerializeEnum( splitDim );
}

void COnnxSplitLayer::CalculateShapes()
{
	CheckArchitecture( GetInputCount() >= 1, GetPath(), "Layer must have at least 1 input" );
	CheckArchitecture( GetInputCount() <= 2, GetPath(), "Layer must have at most 2 inputs" );
	CheckArchitecture( GetOutputCount() >= 1, GetPath(), "Layer must have at least 1 output" );
	CheckArchitecture( inputShapeBlobs.Size() == 1 || inputShapeBlobs[1] != nullptr,
		GetPath(), "splits input is missing" );

	std::unique_ptr<CDnnBlobBuffer<int>> buff( inputShapeBlobs.Size() == 1 ? nullptr
		: new CDnnBlobBuffer<int>( *inputShapeBlobs[1], TDnnBlobBufferAccess::Read ) );

	if( inputShapeBlobs[0] == nullptr ) {
		CheckArchitecture( buff != nullptr || inputDescs[0].DimSize( splitDim ) % GetOutputCount() == 0,
			GetPath(), "Can't split dimension evenly" );
		for( int i = 0; i < GetOutputCount(); ++i ) {
			outputDescs[i] = inputDescs[0];
			outputDescs[i].SetDimSize( splitDim, buff == nullptr ?
				inputDescs[0].DimSize( splitDim ) / outputDescs.Size() : ( *buff )[i] );
		}
		return;
	}

	CheckArchitecture( buff != nullptr || inputShapeBlobs[0]->DimSize( splitDim ) % GetOutputCount() == 0,
		GetPath(), "Can't split dimension evenly" );

	for( int i = 0; i < GetOutputCount(); ++i ) {
		CBlobDesc outputDesc = inputShapeBlobs[0]->GetDesc();
		outputDesc.SetDimSize( splitDim, buff == nullptr ?
			outputDesc.DimSize( splitDim ) / outputDescs.Size() : ( *buff )[i] );
		outputShapeBlobs[i] = CDnnBlob::CreateBlob( inputShapeBlobs[0]->GetMathEngine(),
			outputDesc.GetDataType(), outputDesc );
	}

	CDnnBlob::SplitByDim( inputShapeBlobs[0]->GetMathEngine(), splitDim,
		inputShapeBlobs[0], outputShapeBlobs );
}

void COnnxSplitLayer::RunOnce()
{
	if( inputShapeBlobs[0] != nullptr ) {
		return;
	}

	CDnnBlob::SplitByDim( MathEngine(), splitDim, inputBlobs[0], outputBlobs );
}

} // namespace NeoML
