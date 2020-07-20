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

#include "ClipNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CClipNode::CClipNode( int nodeIndex, const onnx::NodeProto& clip, int opsetVersion ) :
	COpNode( nodeIndex, clip, opsetVersion ),
	minValue( attributes.GetOptionalFloat( "min", -FLT_MAX ) ),
	maxValue( attributes.GetOptionalFloat( "max", FLT_MAX ) )
{
	// Newer versions getting min and max values as inputs, not as attributes
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= 10, "opset version", clip );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", clip );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", clip );
}

void CClipNode::CalcOutputTensors( CGraphTensors& tensors, IMathEngine& mathEngine )
{
	InputTensor( tensors, 0 ).Shape.CopyTo( OutputTensor( tensors, 0 ).Shape );

	CheckNeoOnnxSupport( InputTensor( tensors, 0 ).Data == nullptr, "output pre-calculation", onnxNode );
	// The OutputTensor( tensors, 0 ).Data was already set to nullptr in default constructor.
}

void CClipNode::MarkTensorDims( const CGraphTensors& tensors, CGraphDims& dims )
{
	if( !InputDim( dims, 0 ).IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( OutputTensor( tensors, 0 ).Shape, InputDim( dims, 0 ), OutputDim( dims, 0 ) ), "marking output dimensions failed", onnxNode );
	}

	if( !OutputDim( dims, 0 ).IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( InputTensor( tensors, 0 ).Shape, OutputDim( dims, 0 ), InputDim( dims, 0 ) ), "marking input dimensions failed", onnxNode );
	}
}

void CClipNode::AddLayers( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn )
{
	CheckNeoOnnxSupport( minValue == 0.f, "'min' value must be equal to 0", onnxNode );

	CPtr<CReLULayer> relu = new CReLULayer( dnn.GetMathEngine() );
	relu->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	if( maxValue < FLT_MAX ) {
		relu->SetUpperThreshold( maxValue );
	}

	relu->Connect( 0, *InputMapping( mappings, 0 ).Layer, InputMapping( mappings, 0 ).OutputIndex );
	dnn.AddLayer( *relu );

	OutputMapping( mappings, 0 ) = CNeoMLMapping( relu, 0 );
}

} // namespace NeoOnnx
