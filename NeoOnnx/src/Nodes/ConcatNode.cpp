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

CConcatNode::CConcatNode( const onnx::NodeProto& concat ) :
	CNode( concat ),
	axis( attributes.GetRequiredInt( "axis" ) )
{
	CheckOnnxProtocol( input.Size() > 1, "node must have more than 1 inputs", concat );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", concat );
}

void CConcatNode::CalcOutputShape()
{
	CTensorShape& outputShape = output[0].Shape;

	for( int inputIndex = 0; inputIndex < input.Size(); ++inputIndex ) {
		const CTensorShape& inputShape = InputTensor( inputIndex ).Shape;

		if( outputShape.IsEmpty() ) {
			inputShape.CopyTo( outputShape );
		} else {
			CheckOnnxProtocol( axis < inputShape.Size(),"'axis' must be less than input's dimensions count", onnxNode );
			outputShape[axis] += inputShape[axis];
		}
	}
}

void CConcatNode::CalcOutputData()
{
	TBlobType outputBlobType = CT_Invalid;

	for( int inputIndex = 0; inputIndex < input.Size(); ++inputIndex ) {
		const CTensor& input = InputTensor( inputIndex );
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
	for( int dim = 0; dim < static_cast<int>( output[0].Shape.Size() ); ++dim ) {
		outputDesc.SetDimSize( dim, output[0].Shape[dim] );
	}

	IMathEngine& mathEngine = InputTensor( 0 ).Data->GetMathEngine();
	output[0].Data = CDnnBlob::CreateBlob( mathEngine, outputBlobType, outputDesc );

	// Collecting input blobs for concatenation.
	CObjectArray<CDnnBlob> inputBlobs;
	for( int inputIndex = 0; inputIndex < input.Size(); ++inputIndex ) {
		inputBlobs.Add( InputTensor( inputIndex ).Data );
	}

	// Precalculation.Shape.size()
	CDnnBlob::MergeByDim( mathEngine, static_cast<TBlobDim>( axis ), inputBlobs, output[0].Data );
}

void CConcatNode::MarkTensorDims()
{
	if( output[0].Data != nullptr ) {
		return;
	}

	for( int inputIndex = 0; inputIndex < input.Size(); ++inputIndex ) {
		if( !InputTensor( inputIndex ).Dim.IsEmpty() ) {
			CheckNeoOnnxInternal( output[0].SetTensorDim( InputTensor( inputIndex ).Dim ),
				"marking output dimensions failed", onnxNode );
		}
	}

	if( !output[0].Dim.IsEmpty() ) {
		for( int inputIndex = 0; inputIndex < input.Size(); ++inputIndex ) {
			CheckNeoOnnxInternal( InputTensor( inputIndex ).SetTensorDim( output[0].Dim ),
				"marking input dimensions failed", onnxNode );
		}
	}
}

void CConcatNode::AddLayers( CDnn& dnn )
{
	if( output[0].Data != nullptr ) {
		return;
	}

	IMathEngine& mathEngine = dnn.GetMathEngine();
	const TBlobDim concatDim = output[0].Dim[axis];

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

	for( int inputIndex = 0; inputIndex < input.Size(); ++inputIndex ) {
		concatLayer->Connect( inputIndex, InputLayer( inputIndex ), InputLayerIndex( inputIndex ) );
	}
	dnn.AddLayer( *concatLayer );

	neoMLInputInfo.Add( CNeoMLInputInfo( concatLayer, 0 ) );
}

} // namespace NeoOnnx
