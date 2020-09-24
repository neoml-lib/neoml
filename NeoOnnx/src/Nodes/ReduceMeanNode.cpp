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

#include "ReduceMeanNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CReduceMeanNode::CReduceMeanNode( int nodeIndex, const onnx::NodeProto& reduceMean, int opsetVersion ) :
	CGlobalPoolNodeBase( nodeIndex, reduceMean, opsetVersion ),
	keepDims( Attributes.GetOptionalInt( "keepdims", 1 ) )
{
	// The differences between versions are in negative indices support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", reduceMean );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", reduceMean );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", reduceMean );

	Attributes.GetRequiredIntArray( "axes", axes );
}

void CReduceMeanNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "constant input", OnnxNode );
	const CTensorShape& inputShape = tensors[Input[0]].Shape;
	CTensorShape& outputShape = tensors[Output[0]].Shape;

	int axisIndex = 0;
	for( int i = 0; i < inputShape.Size(); ++i ) {
		if( axisIndex < axes.Size() && axes[axisIndex] == i ) {
			++axisIndex;
			if( keepDims != 0 ) {
				outputShape.Add( 1 );
			}
		} else {
			outputShape.Add( inputShape[i] );
		}
	}

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
}

void CReduceMeanNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	const CTensorDim& inputDim = dims[Input[0]];
	CheckNeoOnnxInternal( inputDim.Size() == tensors[Input[0]].Shape.Size(),
		"input's dimensions must be marked", OnnxNode );

	if( keepDims != 0 ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, inputDim, dims[Output[0]] ),
			"labeling output dimensions failed", OnnxNode );
		return;
	}

	CTensorDim outputDim;
	int axisIndex = 0;
	for( int i = 0; i < inputDim.Size(); ++i ) {
		if( axisIndex < axes.Size() && axes[axisIndex] == i ) {
			++axisIndex;
		} else {
			outputDim.Add( inputDim[i] );
		}
	}

	CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, outputDim, dims[Output[0]] ),
		"labeling output dimensions failed", OnnxNode );
}

void CReduceMeanNode::AddLayers( const CGraph& /* graph */, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CTensorDim dimsToPool;
	const CTensorDim& inputDim = dims[Input[0]];
	for( int axisIndex = 0; axisIndex < axes.Size(); ++axisIndex ) {
		dimsToPool.Add( inputDim[axes[axisIndex]] );
	}

	AddPoolingLayer( PT_Mean, dimsToPool, tensors[Input[0]].Shape, inputDim, neoMLLinks, dnn );
}

} // namespace NeoOnnx
