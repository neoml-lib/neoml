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
	axis( attributes.GetRequiredInt( "axis" ) )
{
	// Older versions have "axis" attribute as optional, not as required
	CheckNeoOnnxSupport( opsetVersion >= 4 && opsetVersion <= MaxOpsetVersion, "opset version", concat );

	CheckOnnxProtocol( InputCount() > 1, "node must have more than 1 inputs", concat );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", concat );
}

void CConcatNode::CalcOutputTensors( CGraphTensors& tensors, IMathEngine& mathEngine )
{
	CTensorShape& outputShape = OutputTensor( tensors, 0 ).Shape;

	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		const CTensorShape& inputShape = InputTensor( tensors, inputIndex ).Shape;

		if( outputShape.IsEmpty() ) {
			inputShape.CopyTo( outputShape );
		} else {
			CheckOnnxProtocol( axis < inputShape.Size(),"'axis' must be less than input's dimensions count", onnxNode );
			outputShape[axis] += inputShape[axis];
		}
	}

	TBlobType outputBlobType = CT_Invalid;

	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		const CTensor& input = InputTensor( tensors, inputIndex );
		if( input.Data == nullptr ) {
			// Data can't be pre-calculated.
			return;
		}

		CheckOnnxProtocol( outputBlobType == CT_Invalid || outputBlobType == input.Data->GetDataType(),
			"inputs with different data types", onnxNode );
		outputBlobType = input.Data->GetDataType();
	}

	if( outputBlobType == CT_Invalid ) {
		outputBlobType = CT_Float;
	}

	// Precalculating node's output.

	// Allocating output blob.
	CBlobDesc outputDesc( outputBlobType );
	for( int dim = 0; dim < static_cast<int>( OutputTensor( tensors, 0 ).Shape.Size() ); ++dim ) {
		outputDesc.SetDimSize( dim, OutputTensor( tensors, 0 ).Shape[dim] );
	}

	OutputTensor( tensors, 0 ).Data = CDnnBlob::CreateBlob( mathEngine, outputBlobType, outputDesc );

	// Collecting input blobs for concatenation.
	CObjectArray<CDnnBlob> inputBlobs;
	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		inputBlobs.Add( InputTensor( tensors, inputIndex ).Data );
	}

	// Precalculation.Shape.size()
	CDnnBlob::MergeByDim( mathEngine, static_cast<TBlobDim>( axis ), inputBlobs, OutputTensor( tensors, 0 ).Data );
}

void CConcatNode::MarkTensorDims( const CGraphTensors& tensors, CGraphDims& dims )
{
	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		if( !InputDim( dims, inputIndex ).IsEmpty() ) {
			CheckNeoOnnxInternal( SetTensorDim( OutputTensor( tensors, 0 ).Shape, InputDim( dims, 0 ), OutputDim( dims, 0 ) ),
				"marking output dimensions failed", onnxNode );
		}
	}

	if( !OutputDim( dims, 0 ).IsEmpty() ) {
		for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
			CheckNeoOnnxInternal( SetTensorDim( InputTensor( tensors, inputIndex ).Shape, OutputDim( dims, 0 ), InputDim( dims, inputIndex ) ),
				"marking input dimensions failed", onnxNode );
		}
	}
}

void CConcatNode::AddLayers( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn )
{
	if( InputTensor( tensors, 0 ).Data != nullptr ) {
		return;
	}

	IMathEngine& mathEngine = dnn.GetMathEngine();
	const TBlobDim concatDim = OutputDim( dims, 0 )[axis];

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
			NeoAssert( false ); // not supported by NeoML.
	}
	concatLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		concatLayer->Connect( inputIndex, *InputMapping( mappings, inputIndex ).Layer, InputMapping( mappings, inputIndex ).OutputIndex );
	}
	dnn.AddLayer( *concatLayer );

	OutputMapping( mappings, 0 ) = CNeoMLMapping( concatLayer, 0 );
}

} // namespace NeoOnnx
