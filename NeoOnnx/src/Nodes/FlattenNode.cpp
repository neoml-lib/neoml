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

#include "common.h"
#pragma hdrstop

#include "FlattenNode.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CFlattenNode::CFlattenNode( int nodeIndex, const onnx::NodeProto& flatten, int opsetVersion ) :
	COpNode( nodeIndex, flatten, opsetVersion ),
	axis( Attributes.GetOptionalInt( "axis", 1 ) )
{
	// The differences between versions are in supported data types and negative axis index
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", flatten );
	
	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", flatten );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", flatten );
}

void CFlattenNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	const CTensorShape& inputShape = tensors[Input[0]].Shape;
	CTensorShape& outputShape = tensors[Output[0]].Shape;
	outputShape = { 1, 1 };

	for( int dimIndex = 0; dimIndex < inputShape.Size(); ++dimIndex ) {
		outputShape[dimIndex < axis ? 0 : 1] *= inputShape[dimIndex];
	}

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
}

void CFlattenNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	const CTensorDim& inputDims = dims[Input[0]];

	if( !inputDims.IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, { inputDims[axis - 1], inputDims[axis] }, dims[Output[0]] ),
			"labeling output dimensions failed", OnnxNode );
	}
}

void CFlattenNode::AddLayers( const CGraph&, const CTensorCache&, const CDimCache&,
	CNeoMLLinkCache& neoMLLinks, CDnn& )
{
	neoMLLinks[Output[0]] = neoMLLinks[Input[0]];
}

} // namespace NeoOnnx
