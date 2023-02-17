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

#include <NeoML/Dnn/Layers/Onnx/OnnxShapeToBlobLayer.h>

namespace NeoML {

static const int OnnxShapeToBlobLayerVersion = 0;

void COnnxShapeToBlobLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxShapeToBlobLayerVersion );
	COnnxLayerBase::Serialize( archive );
}

void COnnxShapeToBlobLayer::CalculateShapes()
{
	CheckLayerArchitecture( GetInputCount() == 1, "Layer must have 1 input" );
	CheckLayerArchitecture( GetOutputCount() == 1, "Layer must have 1 output" );
	CheckLayerArchitecture( inputShapeBlobs[0] != nullptr, "Input must contain shape" );
	outputDescs[0] = inputShapeBlobs[0]->GetDesc();
}

void COnnxShapeToBlobLayer::RunOnce()
{
	outputBlobs[0]->CopyFrom( inputShapeBlobs[0] );
}

} // namespace NeoML
