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

#include "AveragePoolNode.h"
#include "NodeUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CAveragePoolNode::CAveragePoolNode( const onnx::NodeProto& averagePool, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( averagePool, nodeOutputs ),
	autoPad( attributes.GetOptionalString( "auto_pad", "NOTSET" ) )
{
	CheckOnnxProtocol( input.Size() == 1, "node must have 1 input", averagePool );
	CheckOnnxProtocol( OutputCount() == 1 || OutputCount() == 2, "node must have 1 or 2 outputs", averagePool );

	attributes.GetRequiredIntArray( "kernel_shape", kernelShape );
	attributes.GetOptionalIntArray( "strides", strides );
	attributes.GetOptionalIntArray( "pads", pads );

	CheckNeoOnnxSupport( kernelShape.Size() == 2, "non 2-dimensional max pooling", averagePool );
}

void CAveragePoolNode::OnnxReshape()
{
	// Checking input
	const CTensor& inputTensor = InputTensor( 0 );
	CheckNeoOnnxSupport( inputTensor.GetShape().Size() > 2 && inputTensor.GetShape().Size() <= 4,
		"wrong input tensor's dimensions number", onnxNode );
	const CTensorShape& inputShape = inputTensor.GetShape();
	const int poolDims = static_cast<int>( inputShape.Size() ) - 2;

	// Initializing strides, pads and dilations (if not given).
	if( strides.IsEmpty() ) {
		strides.Add( 1, poolDims );
	}
	if( pads.IsEmpty() ) {
		pads.Add( 0, 2 * poolDims );
	}

	if( autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER" ) {
		CalculatePadding( autoPad, inputShape, kernelShape, pads );
	}

	for( int padIndex = 0; padIndex < pads.Size(); ++padIndex ) {
		CheckNeoOnnxSupport( pads[padIndex] == 0, "average pooling with padding", onnxNode );
	}

	// Calculating output shape.
	CTensorShape outputShape;
	inputShape.CopyTo( outputShape );
	for( int dimIndex = 0; dimIndex < poolDims; ++dimIndex ) {
		outputShape[dimIndex + 2] = ( inputShape[dimIndex + 2] + pads[dimIndex] + pads[dimIndex + poolDims]
			- kernelShape[dimIndex] ) / strides[dimIndex] + 1;
	}
	outputData.Add( CTensor( TT_DataTensor, outputShape ) );
}

void CAveragePoolNode::MarkTensorDims()
{
	if( !InputTensor( 0 ).GetTensorDim().IsEmpty() ) {
		CheckNeoOnnxInternal( outputData[0].SetTensorDim( InputTensor( 0 ).GetTensorDim() ),
			"marking output dimensions failed", onnxNode );
	}

	if( !outputData[0].GetTensorDim().IsEmpty() ) {
		CheckNeoOnnxInternal( InputTensor( 0 ).SetTensorDim( outputData[0].GetTensorDim() ),
			"marking input dimensions failed", onnxNode );
	}
}

void CAveragePoolNode::AddLayers( CDnn& dnn )
{
	CPtr<CMeanPoolingLayer> meanPooling = new CMeanPoolingLayer( dnn.GetMathEngine() );
	meanPooling->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	CheckNeoOnnxSupport( InputTensor( 0 ).GetTensorDim()[2] == BD_Height, "wrong pooling dimension", onnxNode );
	CheckNeoOnnxSupport( InputTensor( 0 ).GetTensorDim()[3] == BD_Width, "wrong pooling dimension", onnxNode );

	meanPooling->SetFilterHeight( kernelShape[0] );
	meanPooling->SetFilterWidth( kernelShape[1] );

	meanPooling->SetStrideHeight( strides[0] );
	meanPooling->SetStrideWidth( strides[1] );

	meanPooling->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	dnn.AddLayer( *meanPooling );

	outputInfo.Add( COutputInfo( meanPooling, 0 ) );
}

} // namespace NeoOnnx
