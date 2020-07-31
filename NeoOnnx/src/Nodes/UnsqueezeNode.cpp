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

#include "UnsqueezeNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CUnsqueezeNode::CUnsqueezeNode( int nodeIndex, const onnx::NodeProto& unsqueeze, int opsetVersion ) :
	COpNode( nodeIndex, unsqueeze, opsetVersion )
{
	// Newer versions have negative axes support
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= 10, "opset version", unsqueeze);

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", unsqueeze );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", unsqueeze );

	attributes.GetRequiredIntArray( "axes", axes );
}

void CUnsqueezeNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	const CTensorShape& inputShape = tensors[Input[0]].Shape;

	CTensorShape& outputShape = tensors[Output[0]].Shape;
	outputShape.SetSize( inputShape.Size() + axes.Size() );
	int axisIndex = 0;
	for( int i = 0; i < outputShape.Size(); ++i ) {
		if( axisIndex < axes.Size() && i == axes[axisIndex] ) {
			outputShape[i] = 1;
			axisIndex++;
		} else {
			outputShape[i] = inputShape[i - axisIndex];
		}
	}

	tensors[Output[0]].Data = tensors[Input[0]].Data;
}

void CUnsqueezeNode::MarkTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( tensors[Output[0]].Data != nullptr ) {
		return;
	}

	const CTensorDim& outputDim = ( dims[Output[0]] );

	if( outputDim.IsEmpty() ) {
		return;
	}

	CTensorDim inputDim;
	int axisIndex = 0;
	for( int i = 0; i < outputDim.Size(); ++i ) {
		if( axisIndex < axes.Size() && i == axes[axisIndex] ) {
			++axisIndex;
		} else {
			inputDim.Add( outputDim[i] );
		}
	}

	CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, inputDim, dims[Input[0]] ),
		"marking input dimensions failed", onnxNode );
}

void CUnsqueezeNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims, CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	if( tensors[Output[0]].Data != nullptr ) {
		return;
	}

	neoMLLinks[Output[0]] = neoMLLinks[Input[0]];
}

} // namespace NeoOnnx
