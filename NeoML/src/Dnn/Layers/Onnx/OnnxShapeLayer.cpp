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

#include <NeoML/Dnn/Layers/Onnx/OnnxShapeLayer.h>

namespace NeoML {

static const int OnnxShapeLayerVersion = 0;

void COnnxShapeLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxShapeLayerVersion );
	COnnxLayerBase::Serialize( archive );
	tensorLayout.Serialize( archive );
}

void COnnxShapeLayer::CalculateShapes()
{
	CheckInput1();
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "layer must have 1 output" );

	outputShapeBlobs[0] = CDnnBlob::CreateVector( GetSingleThreadCpuMathEngine(), CT_Int, tensorLayout.Size() );
	CDnnBlobBuffer<int> outputBuff( *outputShapeBlobs[0], TDnnBlobBufferAccess::Write );
	for( int dimIndex = 0; dimIndex < tensorLayout.Size(); ++dimIndex ) {
		outputBuff[dimIndex] = inputDescs[0].DimSize( tensorLayout[dimIndex] );
	}
}

} // namespace NeoML
