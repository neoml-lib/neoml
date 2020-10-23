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

#include "IdentityNode.h"
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CIdentityNode::CIdentityNode( int nodeIndex, const onnx::NodeProto& identity, int opsetVersion ) :
	COpNode( nodeIndex, identity, opsetVersion )
{
	// The differences between versions are in legacy optimization flags
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", identity );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", identity );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", identity );
}

void CIdentityNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
	tensors[Input[0]].Shape.CopyTo( tensors[Output[0]].Shape );
}

void CIdentityNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( !dims[Input[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, dims[Input[0]], dims[Output[0]] ),
			"labeling output dimensions failed", OnnxNode );
	}

	if( !dims[Output[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, dims[Output[0]], dims[Input[0]] ),
			"labeling input dimensions failed", OnnxNode );
	}
}

void CIdentityNode::AddLayers( const CGraph& /* graph */, const CTensorCache& /* tensors */, const CDimCache& /* dims */,
	CNeoMLLinkCache& neoMLLinks, CDnn& /* dnn */ )
{
	neoMLLinks[Output[0]] = neoMLLinks[Input[0]];
}

} // namespace NeoOnnx
