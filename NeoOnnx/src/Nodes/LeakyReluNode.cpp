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
	alpha( attributes.GetOptionalFloat( "alpha", 0.01f ) )
{
	// v1 - original ver.
	// v6 - removed legacy optimization attribute
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", leakyRelu );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", leakyRelu );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", leakyRelu );
}

void CLeakyReluNode::CalcOutputTensors( CGraphTensors& tensors, IMathEngine& mathEngine )
{
	InputTensor( tensors, 0 ).Shape.CopyTo( OutputTensor( tensors, 0 ).Shape );

	CheckNeoOnnxSupport( InputTensor( tensors, 0 ).Data == nullptr, "output pre-calculation", onnxNode );
	// The OutputTensor( tensors, 0 ).Data was already set to nullptr in default constructor.
}

void CLeakyReluNode::MarkTensorDims( const CGraphTensors& tensors, CGraphDims& dims )
{
	if( !InputDim( dims, 0 ).IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( OutputTensor( tensors, 0 ).Shape, InputDim( dims, 0 ), OutputDim( dims, 0 ) ),
			"marking output dimensions failed", onnxNode );
	}

	if( !OutputDim( dims, 0 ).IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( InputTensor( tensors, 0 ).Shape, OutputDim( dims, 0 ), InputDim( dims, 0 ) ),
			"marking input dimensions failed", onnxNode );
	}
}

void CLeakyReluNode::AddLayers( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn )
{
	CPtr<CLeakyReLULayer> leakyRelu = new CLeakyReLULayer( dnn.GetMathEngine() );
	leakyRelu->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	leakyRelu->Connect( 0, *InputMapping( mappings, 0 ).Layer, InputMapping( mappings, 0 ).OutputIndex );
	leakyRelu->SetAlpha( alpha );
	
	dnn.AddLayer( *leakyRelu );

	OutputMapping( mappings, 0 ) = CNeoMLMapping( leakyRelu, 0 );
}

} // namespace NeoOnnx
