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

#include "EluNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CEluNode::CEluNode( int nodeIndex, const onnx::NodeProto& elu, int opsetVersion ) :
	COpNode( nodeIndex, elu, opsetVersion )
{
	// The differences between versions are in legacy optimization flags
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", elu );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", elu );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", elu );
}

void CEluNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
	tensors[Input[0]].Shape.CopyTo( tensors[Output[0]].Shape );
}

void CEluNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
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

void CEluNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CELULayer> elu = new CELULayer( dnn.GetMathEngine() );
	elu->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	elu->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	
	dnn.AddLayer( *elu );

	neoMLLinks[Output[0]] = CNeoMLLink( elu, 0 );
}

} // namespace NeoOnnx
