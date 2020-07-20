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

#include "AbsNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CAbsNode::CAbsNode( int nodeIndex, const onnx::NodeProto& abs, int opsetVersion ) :
	COpNode( nodeIndex, abs, opsetVersion )
{
	// v1 - original
	// v6 - removed legacy optimization attributes and added new data types support
	// v13 - added new data types support
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", abs );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", abs );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", abs );
}

void CAbsNode::CalcOutputTensors( CGraphTensors& tensors, IMathEngine& mathEngine )
{
	CheckNeoOnnxSupport( InputTensor( tensors, 0 ).Data == nullptr, "output pre-calculation", onnxNode );
	// tensor data is already nullptr after default constructor
	InputTensor( tensors, 0 ).Shape.CopyTo( OutputTensor( tensors, 0 ).Shape );
}

void CAbsNode::MarkTensorDims( const CGraphTensors& tensors, CGraphDims& dims )
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

void CAbsNode::AddLayers( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn )
{
	CPtr<CAbsLayer> abs = new CAbsLayer( dnn.GetMathEngine() );
	abs->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	abs->Connect( 0, *InputMapping( mappings, 0 ).Layer, InputMapping( mappings, 0 ).OutputIndex );
	
	dnn.AddLayer( *abs );

	OutputMapping( mappings, 0 ) = CNeoMLMapping( abs, 0 );
}

} // namespace NeoOnnx
