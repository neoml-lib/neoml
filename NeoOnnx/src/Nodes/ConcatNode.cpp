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

#include "ConcatNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CConcatNode::CConcatNode( int nodeIndex, const onnx::NodeProto& concat, int opsetVersion ) :
	COpNode( nodeIndex, concat, opsetVersion ),
	axis( Attributes.GetRequiredInt( "axis" ) )
{
	// Older versions have "axis" attribute as optional, not as required
	CheckNeoOnnxSupport( OpsetVersion >= 4 && OpsetVersion <= MaxOpsetVersion, "opset version", concat );

	CheckOnnxProtocol( InputCount() > 1, "node must have more than 1 inputs", concat );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", concat );
}

void CConcatNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	CTensorShape& outputShape = tensors[Output[0]].Shape;

	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		const CTensorShape& inputShape = tensors[Input[inputIndex]].Shape;

		if( outputShape.IsEmpty() ) {
			inputShape.CopyTo( outputShape );
		} else {
			CheckOnnxProtocol( axis < inputShape.Size(),"'axis' must be less than input's dimensions count", OnnxNode );
			outputShape[axis] += inputShape[axis];
		}
	}

	TBlobType outputBlobType = CT_Invalid;
	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		const CTensor& input = tensors[Input[inputIndex]];
		if( input.Data == nullptr ) {
			// Data can't be pre-calculated
			return;
		}
		CheckOnnxProtocol( outputBlobType == CT_Invalid || outputBlobType == input.Data->GetDataType(),
			"inputs with different data types", OnnxNode );
		outputBlobType = input.Data->GetDataType();
	}
	if( outputBlobType == CT_Invalid ) {
		outputBlobType = CT_Float;
	}

	// Precalculate node's output
	// Allocate output blob
	CBlobDesc outputDesc( outputBlobType );
	for( int dim = 0; dim < static_cast<int>( tensors[Output[0]].Shape.Size() ); ++dim ) {
		outputDesc.SetDimSize( dim, tensors[Output[0]].Shape[dim] );
	}
	tensors[Output[0]].Data = CDnnBlob::CreateBlob( mathEngine, outputBlobType, outputDesc );

	// Collect input blobs for concatenation
	CObjectArray<CDnnBlob> inputBlobs;
	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		inputBlobs.Add( tensors[Input[inputIndex]].Data );
	}

	// Precalculation
	CDnnBlob::MergeByDim( mathEngine, static_cast<TBlobDim>( axis ), inputBlobs, tensors[Output[0]].Data );
}

void CConcatNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		if( !dims[Input[inputIndex]].IsEmpty() ) {
			CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, dims[Input[0]], dims[Output[0]] ),
				"labeling output dimensions failed", OnnxNode );
		}
	}

	if( !dims[Output[0]].IsEmpty() ) {
		for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
			CheckNeoOnnxInternal( SetTensorDim( tensors[Input[inputIndex]].Shape, dims[Output[0]], dims[Input[inputIndex]] ),
				"labeling input dimensions failed", OnnxNode );
		}
	}
}

void CConcatNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	if( tensors[Input[0]].Data != nullptr ) {
		return;
	}

	IMathEngine& mathEngine = dnn.GetMathEngine();
	const TBlobDim concatDim = dims[Output[0]][axis];

	CPtr<CBaseLayer> concatLayer;
	switch( concatDim ) {
		case BD_BatchWidth:
			concatLayer = new CConcatBatchWidthLayer( mathEngine );
			break;
		case BD_Height:
			concatLayer = new CConcatHeightLayer( mathEngine );
			break;
		case BD_Width:
			concatLayer = new CConcatWidthLayer( mathEngine );
			break;
		case BD_Depth:
			concatLayer = new CConcatDepthLayer( mathEngine );
			break;
		case BD_Channels:
			concatLayer = new CConcatChannelsLayer( mathEngine );
			break;
		case BD_BatchLength:
		case BD_ListSize:
		default:
			CheckNeoOnnxSupport( false, "unsupported Concat dimension", OnnxNode );
	}
	concatLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		concatLayer->Connect( inputIndex, *neoMLLinks[Input[inputIndex]].Layer, neoMLLinks[Input[inputIndex]].OutputIndex );
	}
	dnn.AddLayer( *concatLayer );

	neoMLLinks[Output[0]] = CNeoMLLink( concatLayer, 0 );
}

} // namespace NeoOnnx
