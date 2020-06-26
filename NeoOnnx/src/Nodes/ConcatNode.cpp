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

CConcatNode::CConcatNode( const onnx::NodeProto& concat, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( concat, nodeOutputs ),
	axis( attributes.GetRequiredInt( "axis" ) )
{
	CheckOnnxProtocol( input.Size() > 1, "node must have more than 1 inputs", concat );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", concat );
}

void CConcatNode::OnnxReshape()
{
	CTensorShape outputShape;
	TTensorType outputDataType = TT_ConstantTensor;
	TBlobType outputBlobType = CT_Invalid;
	for( int inputIndex = 0; inputIndex < input.Size(); ++inputIndex ) {

		const CTensor& input = InputTensor( inputIndex );
		switch( input.GetType() ) {
			case TT_DataTensor:
				outputDataType = TT_DataTensor;
				break;
			case TT_ConstantTensor:
				CheckOnnxProtocol( outputBlobType == CT_Invalid || outputBlobType == input.GetData()->GetDataType(),
					"inputs with different data types", onnxNode );
				outputBlobType = input.GetData()->GetDataType();
				break;
			default:
				NeoAssert( false );
		}

		if( outputShape.IsEmpty() ) {
			input.GetShape().CopyTo( outputShape );
		} else {
			CheckOnnxProtocol( axis < input.GetShape().Size(),
				"'axis' must be less than input's dimensions count", onnxNode );
			outputShape[axis] += input.GetShape()[axis];
		}

		if( input.GetType() == TT_DataTensor ) {
			outputDataType = TT_DataTensor;
		}
	}

	if( outputBlobType == CT_Invalid ) {
		outputBlobType = CT_Float;
	}

	CPtr<CDnnBlob> outputBlob( nullptr );

	if( outputDataType == TT_ConstantTensor ) {
		// Precalculating node's output.

		// Allocating output blob.
		CBlobDesc outputDesc( outputBlobType );
		for( int dim = 0; dim < static_cast<int>( outputShape.Size() ); ++dim ) {
			outputDesc.SetDimSize( dim, outputShape[dim] );
		}

		IMathEngine& mathEngine = InputTensor( 0 ).GetData()->GetMathEngine();
		outputBlob = CDnnBlob::CreateBlob( mathEngine, outputBlobType, outputDesc );

		// Collecting input blobs for concatenation.
		CObjectArray<CDnnBlob> inputBlobs;
		for( int inputIndex = 0; inputIndex < input.Size(); ++inputIndex ) {
			inputBlobs.Add( InputTensor( inputIndex ).GetData() );
		}

		// Precalculation.GetShape().size()
		CDnnBlob::MergeByDim( mathEngine, static_cast<TBlobDim>( axis ), inputBlobs, outputBlob );
	}

	outputData.Add( CTensor( outputDataType, outputShape, outputBlob ) );
}

void CConcatNode::MarkTensorDims()
{
	if( outputData[0].GetType() == TT_ConstantTensor ) {
		return;
	}

	for( int inputIndex = 0; inputIndex < input.Size(); ++inputIndex ) {
		if( !InputTensor( inputIndex ).GetTensorDim().IsEmpty() ) {
			CheckNeoOnnxInternal( outputData[0].SetTensorDim( InputTensor( inputIndex ).GetTensorDim() ),
				"marking output dimensions failed", onnxNode );
		}
	}

	if( !outputData[0].GetTensorDim().IsEmpty() ) {
		for( int inputIndex = 0; inputIndex < input.Size(); ++inputIndex ) {
			CheckNeoOnnxInternal( InputTensor( inputIndex ).SetTensorDim( outputData[0].GetTensorDim() ),
				"marking input dimensions failed", onnxNode );
		}
	}
}

void CConcatNode::AddLayers( CDnn& dnn )
{
	if( outputData[0].GetType() == TT_ConstantTensor ) {
		return;
	}

	IMathEngine& mathEngine = dnn.GetMathEngine();
	const TBlobDim concatDim = outputData[0].GetTensorDim()[axis];

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

	outputInfo.Add( COutputInfo( concatLayer, 0 ) );
}

} // namespace NeoOnnx
