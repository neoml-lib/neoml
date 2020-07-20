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

#include "ReluNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CReluNode::CReluNode( int nodeIndex, const onnx::NodeProto& relu, int opsetVersion ) :
	COpNode( nodeIndex, relu, opsetVersion )
{
	// The differences between versions are in legacy optimization flags
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", relu );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", relu );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", relu );
}

void CReluNode::CalcOutputTensors( CGraphTensors& tensors, IMathEngine& mathEngine )
{
	InputTensor( tensors, 0 ).Shape.CopyTo( OutputTensor( tensors, 0 ).Shape );

	CheckNeoOnnxSupport( InputTensor( tensors, 0 ).Data == nullptr, "output pre-calculation", onnxNode );
	// The OutputTensor( tensors, 0 ).Data was already set to nullptr in default constructor.
}

void CReluNode::MarkTensorDims( const CGraphTensors& tensors, CGraphDims& dims )
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

void CReluNode::AddLayers( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn )
{
	CPtr<CReLULayer> relu = new CReLULayer( dnn.GetMathEngine() );
	relu->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	relu->Connect( 0, *InputMapping( mappings, 0 ).Layer, InputMapping( mappings, 0 ).OutputIndex );
	
	dnn.AddLayer( *relu );

	OutputMapping( mappings, 0 ) = CNeoMLMapping( relu, 0 );
}

} // namespace NeoOnnx
