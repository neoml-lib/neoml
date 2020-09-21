/* Copyright Â© 2017-2020 ABBYY Production LLC

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

#include "PoolNode.h"
#include "NodeUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CPoolNodeBase::CPoolNodeBase( TPoolingType _poolingType, int nodeIndex, const onnx::NodeProto& poolNode, int opsetVersion ) :
	COpNode( nodeIndex, poolNode, opsetVersion ),
	poolingType( _poolingType ),
	autoPad( Attributes.GetOptionalString( "auto_pad", "NOTSET" ) )
{
	// The difference between versions are in rarely used attributes (not supported by NeoOnnx): ceil_mode, storage_order etc)
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", poolNode );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", poolNode );
	CheckOnnxProtocol( OutputCount() == 1 || OutputCount() == 2, "node must have 1 or 2 outputs", poolNode );

	Attributes.GetRequiredIntArray( "kernel_shape", kernelShape );
	Attributes.GetOptionalIntArray( "strides", strides );
	Attributes.GetOptionalIntArray( "pads", pads );

	CheckNeoOnnxSupport( kernelShape.Size() == 2, "non 2-dimensional max pooling", poolNode );
}

void CPoolNodeBase::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	// Check input
	const CTensor& inputTensor = tensors[Input[0]];
	CheckNeoOnnxSupport( inputTensor.Shape.Size() > 2 && inputTensor.Shape.Size() <= 4,
		"wrong input tensor's dimensions number", OnnxNode );
	const CTensorShape& inputShape = inputTensor.Shape;
	const int poolDims = static_cast<int>( inputShape.Size() ) - 2;

	// Initialize strides, pads and dilations (if not given)
	if( strides.IsEmpty() ) {
		strides.Add( 1, poolDims );
	}
	if( pads.IsEmpty() ) {
		pads.Add( 0, 2 * poolDims );
	}

	if( autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER" ) {
		CalculatePadding( autoPad, inputShape, kernelShape, pads, OnnxNode );
	}

	for( int padIndex = 0; padIndex < pads.Size(); ++padIndex ) {
		CheckNeoOnnxSupport( pads[padIndex] == 0, "max pooling with padding", OnnxNode );
	}

	// Calculate output shape
	CTensorShape& outputShape = tensors[Output[0]].Shape;
	inputShape.CopyTo( outputShape );
	for( int dimIndex = 0; dimIndex < poolDims; ++dimIndex ) {
		outputShape[dimIndex + 2] = ( inputShape[dimIndex + 2] + pads[dimIndex] + pads[dimIndex + poolDims]
			- kernelShape[dimIndex] ) / strides[dimIndex] + 1;
	}

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
}

void CPoolNodeBase::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( !dims[Input[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, dims[Input[0]], dims[Output[0]] ),
			"labeling output dimensions failed", OnnxNode );
	}

	if( !dims[Output[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, dims[Output[0]], dims[Input[0]] ),
			"labeling input dimensions failed", OnnxNode );
	}
}

void CPoolNodeBase::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CPoolingLayer> pooling;
	static_assert( PT_Count == 2, "PT_Count != 2" );
	switch( poolingType ) {
		case PT_Max:
			pooling = new CMaxPoolingLayer( dnn.GetMathEngine() );
			break;
		case PT_Mean:
			pooling = new CMeanPoolingLayer( dnn.GetMathEngine() );
			break;
		default:
			CheckNeoOnnxInternal( false, "unknown pool type", OnnxNode );
	}
	pooling->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	CheckNeoOnnxSupport( dims[Input[0]][2] == BD_Height, "wrong pooling dimension", OnnxNode );
	CheckNeoOnnxSupport( dims[Input[0]][3] == BD_Width, "wrong pooling dimension", OnnxNode );

	pooling->SetFilterHeight( kernelShape[0] );
	pooling->SetFilterWidth( kernelShape[1] );

	pooling->SetStrideHeight( strides[0] );
	pooling->SetStrideWidth( strides[1] );

	pooling->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *pooling );

	neoMLLinks[Output[0]] = CNeoMLLink( pooling, 0 );
}

} // namespace NeoOnnx
