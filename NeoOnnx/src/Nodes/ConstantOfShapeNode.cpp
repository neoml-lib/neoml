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

#include "ConstantOfShapeNode.h"
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CConstantOfShapeNode::CConstantOfShapeNode( int nodeIndex, const onnx::NodeProto& constantOfShape, int opsetVersion ) :
	COpNode( nodeIndex, constantOfShape, opsetVersion )
{
	// This op was introduced in version 9
	CheckOnnxProtocol( OpsetVersion >= 9, "wrong opset version", constantOfShape );
	CheckNeoOnnxSupport( OpsetVersion <= MaxOpsetVersion, "opset version", constantOfShape );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", constantOfShape );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", constantOfShape );
}

void CConstantOfShapeNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data != nullptr, "non-constant input tensor", OnnxNode );
	CheckNeoOnnxSupport( tensors[Input[0]].Data->GetDataType() == CT_Int, "non-integer input tensor", OnnxNode );

	tensors[Output[0]].Shape.SetSize( tensors[Input[0]].Data->GetDataSize() );
	tensors[Input[0]].Data->CopyTo( tensors[Output[0]].Shape.GetPtr() );

	CTensor value;
	value.Data = CDnnBlob::CreateVector( mathEngine, CT_Float, 1 );
	value.Data->Clear();
	value = Attributes.GetOptionalTensor( "value", value, mathEngine );

	CBlobDesc outputBlobDesc( value.Data->GetDataType() );
	for( int dimIndex = 0; dimIndex < tensors[Output[0]].Shape.Size(); ++dimIndex ) {
		outputBlobDesc.SetDimSize( dimIndex, tensors[Output[0]].Shape[dimIndex] );
	}

	tensors[Output[0]].Data = CDnnBlob::CreateBlob( mathEngine, value.Data->GetDataType(), outputBlobDesc );
	if( tensors[Output[0]].Data->GetDataType() == CT_Float ) {
		tensors[Output[0]].Data->Fill( value.Data->GetData().GetValue() );
	} else {
		tensors[Output[0]].Data->Fill<int>( value.Data->GetData<int>().GetValue() );
	}
}

} // namespace NeoOnnx
