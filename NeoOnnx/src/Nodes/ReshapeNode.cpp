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

#include "ReshapeNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CReshapeNode::CReshapeNode( int nodeIndex, const onnx::NodeProto& reshape, int opsetVersion ) :
	COpNode( nodeIndex, reshape, opsetVersion ),
	hasFixedShape( false ),
	hasRemainder( false )
{
	// The differences between versions are in supported data types and legacy optimization attributes
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", reshape );

	CheckOnnxProtocol( InputCount() == 2, "node must have 2 inputs", reshape );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", reshape );
}

void CReshapeNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "constant first input", onnxNode );
	CheckNeoOnnxSupport( tensors[Input[1]].Data != nullptr, "non-constant second input", onnxNode );

	const CTensorShape& inputShape = tensors[Input[0]].Shape;

	shape.SetSize( tensors[Input[1]].Data->GetDataSize() );
	tensors[Input[1]].Data->CopyTo( shape.GetPtr() );

	hasFixedShape = false;
	hasRemainder = false;

	CTensorShape& outputShape = tensors[Output[0]].Shape;
	outputShape.SetSize( shape.Size() );

	int remDim = -1;
	size_t rem = 1;

	for( int i = 0; i < inputShape.Size(); ++i ) {
		rem *= inputShape[i];
	}

	for( int i = 0; i < shape.Size(); ++i ) {
		switch( shape[i] ) {
			case 0:
				// Don't change dim size
				CheckOnnxProtocol( rem % inputShape[i] == 0, "input's elements count isn't divisible by shape", onnxNode );
				rem /= inputShape[i];
				outputShape[i] = inputShape[i];
				break;
			case -1:
				// Remainder dim
				CheckOnnxProtocol( remDim == -1, "only one dimension can be -1", onnxNode );
				outputShape[i] = 1;
				remDim = i;
				hasRemainder = true;
				break;
			default:
				// Fixed dim size
				CheckOnnxProtocol( shape[i] > 0, "negative shape value", onnxNode );
				CheckOnnxProtocol( rem % shape[i] == 0, "input's elements count isn't divisible by shape", onnxNode );
				rem /= shape[i];
				outputShape[i] = shape[i];
				hasFixedShape = true;
		}
	}

	if( remDim != -1 ) {
		outputShape[remDim] = static_cast<int>( rem );
	}

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", onnxNode );
	// The tensors[Output[0]].Data was already set to nullptr in default constructor.
}

void CReshapeNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims, CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	if( !hasRemainder && !hasFixedShape ) {
		// Strange case, reshape doesn't do anything...
		neoMLLinks[Output[0]] = neoMLLinks[Input[0]];
		return;
	}

	CPtr<CTransformLayer> transform = new CTransformLayer( dnn.GetMathEngine() );
	transform->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	// This layer can't broadcast dimensions
	// Expecting at least one of dims to be marked
	// And (if only input is marked) it must have at least the same amount of dimensions
	CheckNeoOnnxInternal( dims[Output[0]].Size() == shape.Size() || dims[Input[0]].Size() >= shape.Size(),
		"failed to calculate output blob dimensions", onnxNode );

	// If both input and output dims were marked, output dims have higher priority
	const CTensorDim& preferredDim = dims[Output[0]].IsEmpty() ? dims[Input[0]] : dims[Output[0]];

	for( int i = 0; i < shape.Size(); ++i ) {
		CTransformLayer::CDimensionRule rule;
		switch( shape[i] ) {
			case 0:
				// Unchanged
				rule.Operation = CTransformLayer::O_Multiply;
				rule.Parameter = 1;
				break;
			case -1:
				// Remainder dimension
				rule.Operation = CTransformLayer::O_Remainder;
				rule.Parameter = 1; // Doesn't matter
				break;
			default:
				// Fixed size dimension
				rule.Operation = CTransformLayer::O_SetSize;
				rule.Parameter = shape[i];
		}
		transform->SetDimensionRule( preferredDim[i], rule );
	}

	transform->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	
	dnn.AddLayer( *transform );

	neoMLLinks[Output[0]] = CNeoMLLink( transform, 0 );
}

} // namespace NeoOnnx
