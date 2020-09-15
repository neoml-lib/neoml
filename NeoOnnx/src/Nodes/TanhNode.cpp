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

#include "TanhNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CTanhNode::CTanhNode( int nodeIndex, const onnx::NodeProto& tanh, int opsetVersion ) :
	COpNode( nodeIndex, tanh, opsetVersion )
{
	// The differences between versions are in supported data types and legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", tanh );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", tanh );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", tanh );
}

void CTanhNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	tensors[Input[0]].Shape.CopyTo( tensors[Output[0]].Shape );

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
	// The tensors[Output[0]].Data was already set to nullptr in default constructor.
}

void CTanhNode::MarkTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( !dims[Input[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, dims[Input[0]], dims[Output[0]] ),
			"marking output dimensions failed", OnnxNode );
	}

	if( !dims[Output[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, dims[Output[0]], dims[Input[0]] ),
			"marking input dimensions failed", OnnxNode );
	}
}

void CTanhNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims, CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CTanhLayer> tanh = new CTanhLayer( dnn.GetMathEngine() );
	tanh->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	tanh->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	
	dnn.AddLayer( *tanh );

	neoMLLinks[Output[0]] = CNeoMLLink( tanh, 0 );
}

} // namespace NeoOnnx
