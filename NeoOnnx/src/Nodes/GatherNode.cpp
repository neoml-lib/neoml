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

#include "proto/onnx.pb.h"

namespace NeoOnnx {

CGatherNode::CGatherNode( const onnx::NodeProto& gather ) :
	CNode( gather )
{
	CheckOnnxProtocol( input.Size() == 2, "node must have 2 inputs", gather );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", gather );
}

void CGatherNode::CalcOutputShape()
{
	InputTensor( 1 ).Shape.CopyTo( output[0].Shape );
}

void CGatherNode::CalcOutputData()
{
	// TODO: add non-constant tensor support
	CheckNeoOnnxSupport( InputTensor( 0 ).Data != nullptr, "non-constant input", onnxNode );
	// TODO: add float tensor support
	CheckNeoOnnxSupport( InputTensor( 0 ).Data->GetDataType() == CT_Int, "non-integer input", onnxNode );

	CArray<int> data;
	data.SetSize( InputTensor( 0 ).Data->GetDataSize() );
	InputTensor( 0 ).Data->CopyTo( data.GetPtr() );

	CheckNeoOnnxSupport( InputTensor( 1 ).Data != nullptr, "non-constant indices", onnxNode );
	CheckOnnxProtocol( InputTensor( 1 ).Data->GetDataType() == CT_Int, "indices must be integer", onnxNode );

	CArray<int> indices;
	indices.SetSize( InputTensor( 1 ).Data->GetDataSize() );
	InputTensor( 1 ).Data->CopyTo( indices.GetPtr() );

	output[0].Data = InputTensor( 1 ).Data->GetClone();
	int* outputBuffer = output[0].Data->GetBuffer<int>( 0, output[0].Data->GetDataSize() );
	for( int i = 0; i < indices.Size(); ++i ) {
		outputBuffer[i] = data[indices[i]];
	}
	output[0].Data->ReleaseBuffer( outputBuffer, true );
}

} // namespace NeoOnnx
