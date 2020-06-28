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

CGatherNode::CGatherNode( const onnx::NodeProto& gather, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( gather, nodeOutputs )
{
	CheckOnnxProtocol( input.Size() == 2, "node must have 2 inputs", gather );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", gather );
}

void CGatherNode::OnnxReshape()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).GetType() == TT_ConstantTensor, "non-constant input", onnxNode );
	CheckNeoOnnxSupport( InputTensor( 0 ).GetData()->GetDataType() == CT_Int, "non-integer input", onnxNode );

	CArray<int> data;
	data.SetSize( InputTensor( 0 ).GetData()->GetDataSize() );
	InputTensor( 0 ).GetData()->CopyTo( data.GetPtr() );
	
	CheckNeoOnnxSupport( InputTensor( 1 ).GetType() == TT_ConstantTensor, "non-constant indices", onnxNode );
	CheckOnnxProtocol( InputTensor( 1 ).GetData()->GetDataType() == CT_Int, "indices must be integer", onnxNode );

	CArray<int> indices;
	indices.SetSize( InputTensor( 1 ).GetData()->GetDataSize() );
	InputTensor( 1 ).GetData()->CopyTo( indices.GetPtr() );

	CPtr<CDnnBlob> outputBlob = InputTensor( 1 ).GetData()->GetClone();
	int* outputBuffer = outputBlob->GetBuffer<int>( 0, outputBlob->GetDataSize() );
	for( int i = 0; i < indices.Size(); ++i ) {
		outputBuffer[i] = data[indices[i]];
	}
	outputBlob->ReleaseBuffer( outputBuffer, true );
	
	outputData.Add( CTensor( TT_ConstantTensor, InputTensor( 1 ).GetShape(), outputBlob ) );
}

} // namespace NeoOnnx
