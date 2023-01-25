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

#include <NeoML/Dnn/Layers/Onnx/OnnxNonZeroLayer.h>

namespace NeoML {

// Calculates output size based on the input blob of data type T
template<class T>
static CBlobDesc onnxNonZeroOutputSize( const CFastArray<TBlobDim, 8>& inputLayout, CDnnBlob& input )
{
	CDnnBlobBuffer<T> inputBuffer( input, TDnnBlobBufferAccess::Read );
	int nonZeroElements = 0;
	for( int i = 0; i < inputBuffer.Size(); ++i ) {
		if( inputBuffer[i] != 0 ) {
			++nonZeroElements;
		}
	}

	CBlobDesc result = CBlobDesc( CT_Int );
	result.SetDimSize( 0, nonZeroElements );
	result.SetDimSize( 1, inputLayout.Size() );
	return result;
}

// Calculates output data based on the input blob of data type T
template<class T>
static void onnxNonZeroImpl( const CFastArray<TBlobDim, 8>& inputLayout, CDnnBlob& input, CDnnBlob& output )
{
	const int nonZeroElements = output.DimSize( BD_BatchLength );
	const int inputDims = output.DimSize( BD_BatchWidth );

	CDnnBlobBuffer<T> inputBuffer( input, TDnnBlobBufferAccess::Read );
	CDnnBlobBuffer<int> outputBuffer( output, TDnnBlobBufferAccess::Write );
	int outIndex = 0;
	for( int i = 0; i < inputBuffer.Size(); ++i ) {
		if( inputBuffer[i] != 0 ) {
			int flatIndex = i;
			for( int dim = inputDims - 1; dim >= 0; --dim ) {
				outputBuffer[outIndex + dim * nonZeroElements] = flatIndex % input.DimSize( inputLayout[dim] );
				flatIndex /= input.DimSize( inputLayout[dim] );
			}
			outIndex++;
		}
	}
}

//--------------------------------------------------------------------------------------------------------------

static const int OnnxNonZeroLayerVersion = 0;

void COnnxNonZeroLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxNonZeroLayerVersion );
	COnnxLayerBase::Serialize( archive );
	inputLayout.Serialize( archive );
}

void COnnxNonZeroLayer::CalculateShapes()
{
	CheckArchitecture( GetInputCount() == 1, GetPath(), "Layer must have 1 input" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );
	CheckArchitecture( inputShapeBlobs[0] != nullptr, GetPath(), "Input data missing" );

	if( inputShapeBlobs[0]->GetDataType() == CT_Float ) {
		outputDescs[0] = onnxNonZeroOutputSize<float>( inputLayout, *inputShapeBlobs[0] );
	} else {
		outputDescs[0] = onnxNonZeroOutputSize<int>( inputLayout, *inputShapeBlobs[0] );
	}
}

void COnnxNonZeroLayer::RunOnce()
{
	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		onnxNonZeroImpl<float>( inputLayout, *inputShapeBlobs[0], *outputBlobs[0] );
	} else {
		onnxNonZeroImpl<int>( inputLayout, *inputShapeBlobs[0], *outputBlobs[0] );
	}
}

} // namespace NeoML
