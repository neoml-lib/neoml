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

#include "LeakyReluNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CLeakyReluNode::CLeakyReluNode( int nodeIndex, const onnx::NodeProto& leakyRelu, int opsetVersion ) :
	COpNode( nodeIndex, leakyRelu, opsetVersion ),
	alpha( Attributes.GetOptionalFloat( "alpha", 0.01f ) )
{
	// v1 - original ver.
	// v6 - removed legacy optimization attribute
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", leakyRelu );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", leakyRelu );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", leakyRelu );
}

void CLeakyReluNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	tensors[Input[0]].Shape.CopyTo( tensors[Output[0]].Shape );

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
	// The tensors[Output[0]].Data was already set to nullptr in default constructor.
}

void CLeakyReluNode::MarkTensorDims( const CTensorCache& tensors, CDimCache& dims )
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

void CLeakyReluNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims, CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CLeakyReLULayer> leakyRelu = new CLeakyReLULayer( dnn.GetMathEngine() );
	leakyRelu->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	leakyRelu->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	leakyRelu->SetAlpha( alpha );
	
	dnn.AddLayer( *leakyRelu );

	neoMLLinks[Output[0]] = CNeoMLLink( leakyRelu, 0 );
}

} // namespace NeoOnnx
