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

#include "../common.h"
#pragma hdrstop

#include "GatherNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGatherNode::CGatherNode( int nodeIndex, const onnx::NodeProto& gather, int opsetVersion ) :
	COpNode( nodeIndex, gather, opsetVersion )
{
	// Newer versions support negative indices
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= 10, "opset version", gather );

	CheckOnnxProtocol( InputCount() == 2, "node must have 2 inputs", gather );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", gather );
}

void CGatherNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	tensors[Input[1]].Shape.CopyTo( tensors[Output[0]].Shape );

	// TODO: add non-constant tensor support
	CheckNeoOnnxSupport( tensors[Input[0]].Data != nullptr, "non-constant input", OnnxNode );
	// TODO: add float tensor support
	CheckNeoOnnxSupport( tensors[Input[0]].Data->GetDataType() == CT_Int, "non-integer input", OnnxNode );

	CArray<int> data;
	data.SetSize( tensors[Input[0]].Data->GetDataSize() );
	tensors[Input[0]].Data->CopyTo( data.GetPtr() );

	CheckNeoOnnxSupport( tensors[Input[1]].Data != nullptr, "non-constant indices", OnnxNode );
	CheckOnnxProtocol( tensors[Input[1]].Data->GetDataType() == CT_Int, "indices must be integer", OnnxNode );

	CArray<int> indices;
	indices.SetSize( tensors[Input[1]].Data->GetDataSize() );
	tensors[Input[1]].Data->CopyTo( indices.GetPtr() );

	tensors[Output[0]].Data = tensors[Input[1]].Data->GetClone();
	int* outputBuffer = tensors[Output[0]].Data->GetBuffer<int>( 0, tensors[Output[0]].Data->GetDataSize() );
	for( int i = 0; i < indices.Size(); ++i ) {
		outputBuffer[i] = data[indices[i]];
	}
	tensors[Output[0]].Data->ReleaseBuffer( outputBuffer, true );
}

} // namespace NeoOnnx
