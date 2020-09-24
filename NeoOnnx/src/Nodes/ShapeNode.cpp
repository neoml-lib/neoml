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
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CShapeNode::CShapeNode( int nodeIndex, const onnx::NodeProto& shape, int opsetVersion ) :
	COpNode( nodeIndex, shape, opsetVersion )
{
	// This operator doesn't have multiple versions
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", shape );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", shape );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", shape );
}

void CShapeNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	const CTensorShape& inputShape = tensors[Input[0]].Shape;
	tensors[Output[0]].Shape = { inputShape.Size() };
	tensors[Output[0]].Data = CDnnBlob::CreateVector( mathEngine, CT_Int, inputShape.Size() );
	tensors[Output[0]].Data->CopyFrom( inputShape.GetPtr() );
}

} // namespace NeoOnnx
