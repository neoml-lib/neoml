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

CReshapeNode::CReshapeNode( const onnx::NodeProto& reshape, int opsetVersion, IMathEngine& /*mathEngine*/ ) :
	CNode( reshape, opsetVersion ),
	hasFixedShape( false ),
	hasRemainder( false )
{
	// The differences between versions are in supported data types and legacy optimization attributes
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", reshape );

	CheckOnnxProtocol( input.Size() == 2, "node must have 2 inputs", reshape );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", reshape );
}

void CReshapeNode::CalcOutputShape()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).Data == nullptr, "constant first input", onnxNode );
	CheckNeoOnnxSupport( InputTensor( 1 ).Data != nullptr, "non-constant second input", onnxNode );

	const CTensorShape& inputShape = InputTensor( 0 ).Shape;

	shape.SetSize( InputTensor( 1 ).Data->GetDataSize() );
	InputTensor( 1 ).Data->CopyTo( shape.GetPtr() );

	hasFixedShape = false;
	hasRemainder = false;

	CTensorShape& outputShape = output[0].Shape;
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
}

void CReshapeNode::CalcOutputData()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).Data == nullptr, "output pre-calculation", onnxNode );
	// The output[0].Data was already set to nullptr in default constructor.
}

void CReshapeNode::AddLayers( CDnn& dnn )
{
	if( !hasRemainder && !hasFixedShape ) {
		// Strange case, reshape doesn't do anything...
		neoMLInputInfo.Add( InputInfo( 0 ) );
		return;
	}

	CPtr<CTransformLayer> transform = new CTransformLayer( dnn.GetMathEngine() );
	transform->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	// This layer can't broadcast dimensions
	// Expecting at least one of dims to be marked
	// And (if only input is marked) it must have at least the same amount of dimensions
	CheckNeoOnnxInternal( output[0].Dim.Size() == shape.Size() || InputTensor( 0 ).Dim.Size() >= shape.Size(),
		"failed to calculate output blob dimensions", onnxNode );

	// If both input and output dims were marked, output dims have higher priority
	const CTensorDim& preferredDim = output[0].Dim.IsEmpty() ? InputTensor( 0 ).Dim : output[0].Dim;

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

	transform->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	
	dnn.AddLayer( *transform );

	neoMLInputInfo.Add( CNeoMLInputInfo( transform, 0 ) );
}

} // namespace NeoOnnx
