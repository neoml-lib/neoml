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

#include "SqueezeNode.h"
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CSqueezeNode::CSqueezeNode( int nodeIndex, const onnx::NodeProto& squeeze, int opsetVersion ) :
	COpNode( nodeIndex, squeeze, opsetVersion )
{
	// Newer versions have negiative axes support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", squeeze );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", squeeze );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", squeeze );

	Attributes.GetOptionalIntArray( "axes", axes );
}

void CSqueezeNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	const CTensorShape& inputShape = tensors[Input[0]].Shape;
	CTensorShape& outputShape = tensors[Output[0]].Shape;
	outputShape.SetBufferSize( inputShape.Size() - axes.Size() );
	
	int axisIndex = 0;
	for( int i = 0; i < inputShape.Size(); ++i ) {
		if( axisIndex < axes.Size() && ( i == axes[axisIndex] || i == axes[axisIndex] + inputShape.Size() ) ) {
			CheckOnnxProtocol( inputShape[i] == 1, "squeezed dimensions must be of length 1", OnnxNode );
			++axisIndex;
		} else if( !axes.IsEmpty() || inputShape[i] != 1 ) {
			// If axes array is empty we should remove all of the dims with size == 1
			outputShape.Add( inputShape[i] );
		}
	}

	tensors[Output[0]].Data = tensors[Input[0]].Data;
}

void CSqueezeNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( tensors[Output[0]].Data != nullptr ) {
		return;
	}

	const CTensorShape& inputShape = tensors[Input[0]].Shape;
	const CTensorDim& inputDim = dims[Input[0]];

	CTensorDim outputDim;
	int axisIndex = 0;
	for( int i = 0; i < inputDim.Size(); ++i ) {
		if( axisIndex < axes.Size() && ( i == axes[axisIndex] || i == axes[axisIndex] + inputShape.Size() ) ) {
			++axisIndex;
		} else if( !axes.IsEmpty() || inputShape[i] != 1 ) {
			outputDim.Add( inputDim[i] );
		}
	}

	CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, outputDim, dims[Output[0]] ),
		"labeling output dimensions failed", OnnxNode );
}

void CSqueezeNode::AddLayers( const CGraph& /* graph */, const CTensorCache& tensors, const CDimCache& /* dims */,
	CNeoMLLinkCache& neoMLLinks, CDnn& /* dnn */ )
{
	if( tensors[Output[0]].Data != nullptr ) {
		return;
	}

	neoMLLinks[Output[0]] = neoMLLinks[Input[0]];
}

} // namespace NeoOnnx
