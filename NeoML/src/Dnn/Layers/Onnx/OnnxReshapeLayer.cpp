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

#include <NeoML/Dnn/Layers/Onnx/OnnxReshapeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxLayerBase.h>

namespace NeoML {

// Copies the data into reshaped blob
static void onnxReshapeImpl( const CDnnBlob& input, CDnnBlob& output )
{
	if( input.GetDataType() == CT_Float ) {
		input.GetMathEngine().VectorCopy( output.GetData(), input.GetData(), input.GetDataSize() );
	} else {
		input.GetMathEngine().VectorCopy( output.GetData<int>(), input.GetData<int>(), input.GetDataSize() );
	}
}

//---------------------------------------------------------------------------------------------------------------------

static const int OnnxReshapeLayerVersion = 0;

void COnnxReshapeLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxReshapeLayerVersion );
	COnnxLayerBase::Serialize( archive );
	inputLayout.Serialize( archive );
	outputLayout.Serialize( archive );
}

void COnnxReshapeLayer::CalculateShapes()
{
	CheckLayerArchitecture( GetInputCount() == 2, "Layer must have 2 inputs" );
	CheckLayerArchitecture( GetOutputCount() == 1, "Layer must have 1 output" );
	CheckLayerArchitecture( inputShapeBlobs[1] != nullptr, "New shape is missing" );
	CheckLayerArchitecture( inputShapeBlobs[1]->GetDataSize() == outputLayout.Size(), "Dimension number mismatch" );
	CheckLayerArchitecture( inputShapeBlobs[1]->GetDataType() == CT_Int, "Non-integer shape" );

	const CBlobDesc& inputDesc = inputShapeBlobs[0] == nullptr ? inputDescs[0] : inputShapeBlobs[0]->GetDesc();
	CBlobDesc outputDesc = CBlobDesc( inputDesc.GetDataType() );

	// rem (remainder) - special dimension used for the rest of the blob size (not covered by other dims)
	int remIndex = NotFound;
	int remSize = inputDesc.BlobSize();
	CDnnBlobBuffer<int> newShape( *inputShapeBlobs[1], TDnnBlobBufferAccess::Read );
	for( int dimIndex = 0; dimIndex < outputLayout.Size(); ++dimIndex ) {
		if( newShape[dimIndex] == -1 ) {
			CheckLayerArchitecture( remIndex == NotFound, "Two remainders" );
			remIndex = dimIndex;
		} else if( newShape[dimIndex] == 0 ) {
			// 0 at index i in new shape means that i'th dimension of output
			// is equal to i'th dimension of input
			CheckLayerArchitecture( dimIndex < inputLayout.Size(),
				"Attempt to save the dimension of missing input axis" );
			outputDesc.SetDimSize( outputLayout[dimIndex], inputDesc.DimSize( inputLayout[dimIndex] ) );
		} else {
			CheckLayerArchitecture( newShape[dimIndex] > 0, "Negative axis size");
			outputDesc.SetDimSize( outputLayout[dimIndex], newShape[dimIndex] );
		}
		remSize /= outputDesc.DimSize( outputLayout[dimIndex] );
	}

	if( remIndex != NotFound ) {
		CheckLayerArchitecture( remSize > 0, "Output remainder isn't positive" );
		outputDesc.SetDimSize( outputLayout[remIndex], remSize );
		remSize = 1;
	}

	CheckLayerArchitecture( remSize == 1, "Reshape didn't cover all of the data" );

	if( inputShapeBlobs[0] == nullptr ) {
		// The layer will reshape inputBlobs[0] to outputBlobs[0] during RunOnce()
		outputDescs[0] = outputDesc;
	} else {
		outputShapeBlobs[0] = CDnnBlob::CreateBlob( inputShapeBlobs[0]->GetMathEngine(), outputDesc.GetDataType(),
			outputDesc );
		onnxReshapeImpl( *inputShapeBlobs[0], *outputShapeBlobs[0] );
	}
}

void COnnxReshapeLayer::RunOnce()
{
	if( inputShapeBlobs[0] == nullptr ) {
		onnxReshapeImpl( *inputBlobs[0], *outputBlobs[0] );
	}
}

} // namespace NeoML
