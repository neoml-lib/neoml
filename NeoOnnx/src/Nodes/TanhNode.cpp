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
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", tanh );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", tanh );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", tanh );
}

void CTanhNode::CalcOutputTensors( CGraphTensors& tensors, IMathEngine& mathEngine )
{
	InputTensor( tensors, 0 ).Shape.CopyTo( OutputTensor( tensors, 0 ).Shape );

	CheckNeoOnnxSupport( InputTensor( tensors, 0 ).Data == nullptr, "output pre-calculation", onnxNode );
	// The OutputTensor( tensors, 0 ).Data was already set to nullptr in default constructor.
}

void CTanhNode::MarkTensorDims( const CGraphTensors& tensors, CGraphDims& dims )
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

void CTanhNode::AddLayers( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn )
{
	CPtr<CTanhLayer> tanh = new CTanhLayer( dnn.GetMathEngine() );
	tanh->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	tanh->Connect( 0, *InputMapping( mappings, 0 ).Layer, InputMapping( mappings, 0 ).OutputIndex );
	
	dnn.AddLayer( *tanh );

	OutputMapping( mappings, 0 ) = CNeoMLMapping( tanh, 0 );
}

} // namespace NeoOnnx
