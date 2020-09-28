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
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CClipNode::CClipNode( int nodeIndex, const onnx::NodeProto& clip, int opsetVersion ) :
	COpNode( nodeIndex, clip, opsetVersion ),
	minValue( Attributes.GetOptionalFloat( "min", -FLT_MAX ) ),
	maxValue( Attributes.GetOptionalFloat( "max", FLT_MAX ) )
{
	// Newer versions are getting min and max values from node inputs instead of node attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= 10, "opset version", clip );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", clip );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", clip );
}

void CClipNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	tensors[Input[0]].Shape.CopyTo( tensors[Output[0]].Shape );

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
}

void CClipNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
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

void CClipNode::AddLayers( const CGraph& /* graph */, const CTensorCache& /* tensors */, const CDimCache& /* dims */,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CheckNeoOnnxSupport( minValue == 0.f, "'min' value must be equal to 0", OnnxNode );

	CPtr<CReLULayer> relu = new CReLULayer( dnn.GetMathEngine() );
	relu->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	if( maxValue < FLT_MAX ) {
		relu->SetUpperThreshold( maxValue );
	}

	relu->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *relu );

	neoMLLinks[Output[0]] = CNeoMLLink( relu, 0 );
}

} // namespace NeoOnnx
