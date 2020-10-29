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
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

// Attempts to greedily mark dimensions of dstTensor
// Returns false is failed
static bool createTensorLabeling( const CTensorShape& srcShape, const CTensorDim& srcDim,
	const CTensorShape& dstShape, CTensorDim& dstDim )
{
	const int len = min( srcShape.Size(), dstShape.Size() );
	dstDim.SetSize( dstShape.Size() );
	int usedDims = 0;

	// Marking first unchanged dimensions
	int left = 0;
	while( left < len && srcShape[left] == dstShape[left] ) {
		dstDim[left] = srcDim[left];
		usedDims |= ( 1 << dstDim[left] );
		++left;
	}

	// Marking last unchanged dimensions
	int right = 0;
	while( right + left < len && srcShape[srcShape.Size() - 1 - right] == dstShape[dstShape.Size() - 1 - right] ) {
		dstDim[dstShape.Size() - 1 - right] = srcDim[srcShape.Size() - 1 - right];
		usedDims |= ( 1 << dstDim[dstShape.Size() - 1 - right] );
		++right;
	}

	if( left + right == dstDim.Size() ) {
		// Whole tensor is labeled
		return true;
	}

	// Trying to extrapolate unused channel-first dimensions along dims between left and right indices
	CTensorDim channelFirst = { BD_BatchLength, BD_BatchWidth, BD_ListSize, BD_Channels, BD_Depth, BD_Height, BD_Width };
	int currDimIndex = left == 0 ? 0 : channelFirst.Find( dstDim[left - 1] ) + 1;
	int lastDimIndex = right == 0 ? channelFirst.Size() : channelFirst.Find( dstDim[right + 1] );
	while( left + right < dstDim.Size() && currDimIndex < lastDimIndex ) {
		TBlobDim currDim = channelFirst[--lastDimIndex];
		if( ( ( 1 << currDim ) & usedDims ) != 0 ) {
			return false;
		}
		dstDim[dstDim.Size() - 1 - right] = currDim;
		right++;
	}

	return left + right == dstDim.Size();
}

static CPtr<CDnnBlob> reshapeBlob( CDnnBlob& inputBlob, const CTensorShape& newShape )
{
	CBlobDesc desc( inputBlob.GetDataType() );
	for( int i = 0; i < newShape.Size(); ++i ) {
		desc.SetDimSize( i, newShape[i] );
	}
	CPtr<CDnnBlob> result = inputBlob.GetCopy();
	result->ReinterpretDimensions( desc );
	return result;
}

//---------------------------------------------------------------------------------------------------------------------

CReshapeNode::CReshapeNode( int nodeIndex, const onnx::NodeProto& reshape, int opsetVersion ) :
	COpNode( nodeIndex, reshape, opsetVersion )
{
	// In opsetVersion == 1 new shape is given as node attribute
	// Since opsetVersion == 5 new shape is acquired from the second input
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", reshape );

	if( OpsetVersion < 5 ) {
		CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", reshape );
	} else {
		CheckOnnxProtocol( InputCount() == 2, "node must have 2 inputs", reshape );
	}
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", reshape );
}

void CReshapeNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	if( OpsetVersion >= 5 ) {
		CheckNeoOnnxSupport( tensors[Input[1]].Data != nullptr, "non-constant second input", OnnxNode );
		shape.SetSize( tensors[Input[1]].Data->GetDataSize() );
		tensors[Input[1]].Data->CopyTo( shape.GetPtr() );
	} else {
		Attributes.GetRequiredIntArray( "shape", shape );
	}

	CTensorShape& outputShape = tensors[Output[0]].Shape;
	outputShape.SetSize( shape.Size() );

	int remDim = -1;
	size_t rem = 1;

	const CTensorShape& inputShape = tensors[Input[0]].Shape;
	for( int i = 0; i < inputShape.Size(); ++i ) {
		rem *= inputShape[i];
	}

	for( int i = 0; i < shape.Size(); ++i ) {
		switch( shape[i] ) {
			case 0:
				// Don't change dim size
				CheckOnnxProtocol( rem % inputShape[i] == 0, "input's elements count isn't divisible by shape", OnnxNode );
				rem /= inputShape[i];
				outputShape[i] = inputShape[i];
				break;
			case -1:
				// Remainder dim
				CheckOnnxProtocol( remDim == -1, "only one dimension can be -1", OnnxNode );
				outputShape[i] = 1;
				remDim = i;
				break;
			default:
				// Fixed dim size
				CheckOnnxProtocol( shape[i] > 0, "negative shape value", OnnxNode );
				CheckOnnxProtocol( rem % shape[i] == 0, "input's elements count isn't divisible by shape", OnnxNode );
				rem /= shape[i];
				outputShape[i] = shape[i];
		}
	}

	if( remDim != -1 ) {
		outputShape[remDim] = static_cast<int>( rem );
	}

	if( tensors[Input[0]].Data != nullptr ) {
		tensors[Output[0]].Data = reshapeBlob( *tensors[Input[0]].Data, outputShape );
	}
}

void CReshapeNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( tensors[Output[0]].Data != nullptr ) {
		// The data already has been pre-calculated
		return;
	}

	if( !dims[Output[0]].IsEmpty() && dims[Input[0]].IsEmpty() ) {
		CTensorDim inputDim;
		if( createTensorLabeling( tensors[Output[0]].Shape, dims[Output[0]], tensors[Input[0]].Shape, inputDim ) ) {
			CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, inputDim, dims[Input[0]] ),
				"labeling input dimensions failed", OnnxNode );
		}
	}
}

void CReshapeNode::AddLayers( const CGraph& /* graph */, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	if( tensors[Output[0]].Data != nullptr ) {
		// The data already has been pre-calculated
		return;
	}

	CPtr<CTransformLayer> transform = new CTransformLayer( dnn.GetMathEngine() );
	transform->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	// This layer can't broadcast dimensions
	// Expects at least one of dims to be marked
	// And (if only input is marked) it must have at least the same amount of dimensions
	CheckNeoOnnxInternal( dims[Output[0]].Size() == shape.Size() || dims[Input[0]].Size() >= shape.Size(),
		"failed to calculate output blob dimensions", OnnxNode );

	// If both input and output dims were marked, output dims have higher priority
	const CTensorDim& preferredDim = dims[Output[0]].IsEmpty() ? dims[Input[0]] : dims[Output[0]];

	for( TBlobDim dim = static_cast< TBlobDim >( 0 ); dim < BD_Count; ++dim ) {
		CTransformLayer::CDimensionRule rule;
		int index = preferredDim.Find( dim );
		if( index >= 0 && index < shape.Size() ) {
			switch( shape[index] ) {
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
					rule.Parameter = shape[index];
			}
		} else {
			rule.Operation = CTransformLayer::O_SetSize;
			rule.Parameter = 1;
		}
		transform->SetDimensionRule( dim, rule );
	}

	transform->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	
	dnn.AddLayer( *transform );

	neoMLLinks[Output[0]] = CNeoMLLink( transform, 0 );
}

} // namespace NeoOnnx
