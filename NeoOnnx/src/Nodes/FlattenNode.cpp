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

#include "common.h"
#pragma hdrstop

#include "FlattenNode.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CFlattenNode::CFlattenNode( const onnx::NodeProto& flatten, int opsetVersion, IMathEngine& /*mathEngine*/ ) :
	COpNode( flatten, opsetVersion ),
	axis( attributes.GetOptionalInt( "axis", 1 ) )
{
	// The differences between versions are in supported data types and negative axis index
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", flatten );
	
	CheckOnnxProtocol( input.Size() == 1, "node must have 1 input", flatten );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", flatten );
}

void CFlattenNode::CalcOutputShape()
{
	const CTensorShape& inputShape = InputTensor( 0 ).Shape;
	CTensorShape& outputShape = output[0].Shape;
	outputShape = { 1, 1 };

	for( int dimIndex = 0; dimIndex < inputShape.Size(); ++dimIndex ) {
		outputShape[dimIndex < axis ? 0 : 1] *= inputShape[dimIndex];
	}
}

void CFlattenNode::CalcOutputData()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).Data == nullptr, "output pre-calculation", onnxNode );
	// The output[0].Data was already set to nullptr in default constructor.
}

void CFlattenNode::MarkTensorDims()
{
	const CTensorDim& inputDims = InputTensor( 0 ).Dim;

	if( !inputDims.IsEmpty() ) {
		CheckNeoOnnxInternal( output[0].SetTensorDim( { inputDims[axis - 1], inputDims[axis] } ),
			"marking output dimensions failed", onnxNode );
	}
}

void CFlattenNode::AddLayers( CDnn& )
{
	neoMLInputInfo.Add( InputInfo( 0 ) );
}

} // namespace NeoOnnx
