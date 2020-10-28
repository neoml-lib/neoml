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
	minValue( -FLT_MAX ),
	maxValue( FLT_MAX )
{
	// v1 and v6 get min/max values from node attributes
	// v11 and older get min/max value from additional inputs
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", clip );

	if( OpsetVersion < 11 ) {
		CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", clip );
	} else {
		CheckOnnxProtocol( InputCount() >= 1 || InputCount() <= 3, "node must have from 1 up to 3 inputs", clip );
	}

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

void CClipNode::AddLayers( const CGraph& /* graph */, const CTensorCache& tensors, const CDimCache& /* dims */,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	getClipValues( tensors );
	CheckNeoOnnxSupport( minValue == 0.f, "clipping with min != 0", OnnxNode );

	CPtr<CReLULayer> relu = new CReLULayer( dnn.GetMathEngine() );
	relu->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	if( maxValue < FLT_MAX ) {
		relu->SetUpperThreshold( maxValue );
	}

	relu->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *relu );

	neoMLLinks[Output[0]] = CNeoMLLink( relu, 0 );
}

// Gets clip values based on opset version
void CClipNode::getClipValues( const CTensorCache& tensors )
{
	if( OpsetVersion < 11 ) {
		minValue = Attributes.GetOptionalFloat( "min", -FLT_MAX );
		maxValue = Attributes.GetOptionalFloat( "max", FLT_MAX );
		return;
	}

	if( InputCount() > 1 ) {
		const CDnnBlob* minValueBlob = tensors[Input[1]].Data;
		CheckNeoOnnxSupport( minValueBlob != nullptr, "user-provided clip min value", OnnxNode );
		if( minValueBlob->GetDataType() == CT_Float ) {
			minValue = minValueBlob->GetData<float>().GetValue();
		} else {
			minValue = static_cast<float>( minValueBlob->GetData<int>().GetValue() );
		}

		if( InputCount() > 2 ) {
			const CDnnBlob* maxValueBlob = tensors[Input[2]].Data;
			CheckNeoOnnxSupport( maxValueBlob != nullptr, "user-provided clip min value", OnnxNode );
			if( maxValueBlob->GetDataType() == CT_Float ) {
				maxValue = maxValueBlob->GetData<float>().GetValue();
			} else {
				maxValue = static_cast<float>( maxValueBlob->GetData<int>().GetValue() );
			}
		}
	}
}

} // namespace NeoOnnx
