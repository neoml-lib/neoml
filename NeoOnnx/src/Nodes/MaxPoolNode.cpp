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

#include "MaxPoolNode.h"
#include "NodeUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CMaxPoolNode::CMaxPoolNode( int nodeIndex, const onnx::NodeProto& maxPool, int opsetVersion ) :
	COpNode( nodeIndex, maxPool, opsetVersion ),
	autoPad( attributes.GetOptionalString( "auto_pad", "NOTSET" ) )
{
	// The difference between versions are in rarely used attributes (not supported by NeoOnnx): ceil_mode, storage_order etc.
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", maxPool );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", maxPool );
	CheckOnnxProtocol( OutputCount() == 1 || OutputCount() == 2, "node must have 1 or 2 outputs", maxPool );

	attributes.GetRequiredIntArray( "kernel_shape", kernelShape );
	attributes.GetOptionalIntArray( "strides", strides );
	attributes.GetOptionalIntArray( "pads", pads );

	CheckNeoOnnxSupport( kernelShape.Size() == 2, "non 2-dimensional max pooling", maxPool );
}

void CMaxPoolNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	// Checking input
	const CTensor& inputTensor = tensors[Input[0]];
	CheckNeoOnnxSupport( inputTensor.Shape.Size() > 2 && inputTensor.Shape.Size() <= 4,
		"wrong input tensor's dimensions number", onnxNode );
	const CTensorShape& inputShape = inputTensor.Shape;
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
		CheckNeoOnnxSupport( pads[padIndex] == 0, "max pooling with padding", onnxNode );
	}

	// Calculating output shape.
	CTensorShape& outputShape = tensors[Output[0]].Shape;
	inputShape.CopyTo( outputShape );
	for( int dimIndex = 0; dimIndex < poolDims; ++dimIndex ) {
		outputShape[dimIndex + 2] = ( inputShape[dimIndex + 2] + pads[dimIndex] + pads[dimIndex + poolDims]
			- kernelShape[dimIndex] ) / strides[dimIndex] + 1;
	}

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", onnxNode );
	// The tensors[Output[0]].Data was already set to nullptr in default constructor.
}

void CMaxPoolNode::MarkTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( !dims[Input[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, dims[Input[0]], dims[Output[0]] ),
			"marking output dimensions failed", onnxNode );
	}

	if( !dims[Output[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, dims[Output[0]], dims[Input[0]] ),
			"marking input dimensions failed", onnxNode );
	}
}

void CMaxPoolNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims, CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CMaxPoolingLayer> maxPooling = new CMaxPoolingLayer( dnn.GetMathEngine() );
	maxPooling->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	CheckNeoOnnxSupport( ( dims[Input[0]] )[2] == BD_Height, "wrong pooling dimension", onnxNode );
	CheckNeoOnnxSupport( ( dims[Input[0]] )[3] == BD_Width, "wrong pooling dimension", onnxNode );

	maxPooling->SetFilterHeight( kernelShape[0] );
	maxPooling->SetFilterWidth( kernelShape[1] );

	maxPooling->SetStrideHeight( strides[0] );
	maxPooling->SetStrideWidth( strides[1] );

	maxPooling->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *maxPooling );

	neoMLLinks[Output[0]] = CNeoMLLink( maxPooling, 0 );
}

} // namespace NeoOnnx
