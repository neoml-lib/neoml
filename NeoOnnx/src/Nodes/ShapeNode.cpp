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

#include "ShapeNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CShapeNode::CShapeNode( const onnx::NodeProto& shape, CMap<CString, CInputInfo>& nodeOutputs, IMathEngine& _mathEngine ) :
	CNode( shape, nodeOutputs ),
	mathEngine( _mathEngine )
{
	CheckOnnxProtocol( input.Size() == 1, "node must have 1 input", shape );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", shape );
}

void CShapeNode::OnnxReshape()
{
	const CTensorShape& inputShape = InputTensor( 0 ).GetShape();

	CPtr<CDnnBlob> outputBlob = CDnnBlob::CreateVector( mathEngine, CT_Int, inputShape.Size() );
	outputBlob->CopyFrom( inputShape.GetPtr() );

	outputData.Add( CTensor( TT_ConstantTensor, { inputShape.Size() }, outputBlob ) );
}

} // namespace NeoOnnx
