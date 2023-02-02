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

#include <NeoML/Dnn/Layers/Onnx/OnnxExpandLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxLayerBase.h>

namespace NeoML {

static const int OnnxExpandLayerVersion = 0;

void COnnxExpandLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxExpandLayerVersion );
	COnnxLayerBase::Serialize( archive );
	tensorLayout.Serialize( archive );
}

void COnnxExpandLayer::CalculateShapes()
{
	CheckLayerArchitecture( GetInputCount() == 2, "Layer must have 2 inputs" );
	CheckLayerArchitecture( GetOutputCount() == 1, "Layer must have 1 output" );
	CheckLayerArchitecture( inputShapeBlobs[0] == nullptr, "First input must be a blob" );
	CheckLayerArchitecture( inputShapeBlobs[1] != nullptr, "Shape input missing" );
	CheckLayerArchitecture( inputShapeBlobs[1]->GetDataSize() <= tensorLayout.Size(), "Dimension number mismatch" );
	CheckLayerArchitecture( inputShapeBlobs[1]->GetDataType() == CT_Int, "Non-integer shape" );

	// Expand operator expands tensor from the first input to the size from the second
	// If the second input contains less elements than rank of the expanded tensor
	// then last |input[1]| dimensions are expanded

	// Number of dimensions not affected by expand
	const int preservedDims = tensorLayout.Size() - inputShapeBlobs[1]->GetDataSize();

	outputDescs[0] = inputDescs[0];

	CDnnBlobBuffer<int> newShape( *inputShapeBlobs[1], TDnnBlobBufferAccess::Read );
	for( int dimIndex = 0; dimIndex < newShape.Size(); ++dimIndex ) {
		CheckLayerArchitecture( newShape[dimIndex] > 0, "Negative axis size" );
		const TBlobDim& dim = tensorLayout[preservedDims + dimIndex];
		// Corner-case: if expanded size is 1 but the original size is more than one
		// then the dimension remains as-is
		outputDescs[0].SetDimSize( dim, newShape[dimIndex] == 1 ? inputDescs[0].DimSize( dim ) : newShape[dimIndex] );
	}
}

void COnnxExpandLayer::RunOnce()
{
	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		MathEngine().BroadcastCopy( outputBlobs[0]->GetData(), inputBlobs[0]->GetData(), outputBlobs[0]->GetDesc(),
			inputBlobs[0]->GetDesc(), 1 );
	} else {
		MathEngine().BroadcastCopy( outputBlobs[0]->GetData<int>(), inputBlobs[0]->GetData<int>(), outputBlobs[0]->GetDesc(),
			inputBlobs[0]->GetDesc(), 1 );
	}
}

} // namespace NeoML
