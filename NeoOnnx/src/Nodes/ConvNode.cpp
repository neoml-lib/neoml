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

#include "ConvNode.h"
#include "NodeUtils.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CConvNode::CConvNode( int nodeIndex, const onnx::NodeProto& conv, int opsetVersion ) :
	COpNode( nodeIndex, conv, opsetVersion ),
	group( Attributes.GetOptionalInt( "group", 1 ) ),
	autoPad( Attributes.GetOptionalString( "auto_pad", "NOTSET" ) )
{
	// The differences between versions are in default values of some flags and supported data types
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", conv );

	CheckOnnxProtocol( InputCount() == 2 || InputCount() == 3, "node must have 2 or 3 inputs", conv );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", conv );

	Attributes.GetOptionalIntArray( "strides", strides );
	Attributes.GetOptionalIntArray( "pads", pads );
	Attributes.GetOptionalIntArray( "dilations", dilations );
}

void CConvNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	// Checking input
	const CTensor& inputTensor = tensors[Input[0]];
	CheckNeoOnnxSupport( inputTensor.Shape.Size() > 2 && inputTensor.Shape.Size() <= 4,
		"wrong input tensor's dimensions number", OnnxNode );
	const CTensorShape& inputShape = inputTensor.Shape;
	const int convDims = static_cast<int>( inputShape.Size() ) - 2;

	// Initializing strides, pads and dilations (if not given).
	if( strides.IsEmpty() ) {
		strides.Add( 1, convDims );
	}
	if( pads.IsEmpty() ) {
		pads.Add( 0, 2 * convDims );
	}
	if( dilations.IsEmpty() ) {
		dilations.Add( 1, convDims );
	}

	// Checking groups
	// NeoML supports only 2D, 3D and Channelwise2D convolutions.
	CheckNeoOnnxSupport( group == 1 || ( group == inputShape[1] && convDims < 3 ),
		"grouped convolutiion (non-channelwise)", OnnxNode );

	// Checking weights
	const CTensor& weight = tensors[Input[1]];
	CheckNeoOnnxSupport( weight.Data != nullptr, "non-constant weights", OnnxNode );
	CheckOnnxProtocol( weight.Shape.Size() == convDims + 2,
		"wrong weight tensor's dimensions number", OnnxNode );
	const int filterCount = weight.Shape[0];
	CheckOnnxProtocol( weight.Shape[group == 1 ? 1 : 0] == inputShape[1],
		"wrong weight tensor's size", OnnxNode );

	// Checking bias
	if( InputCount() == 3 ) {
		const CTensor& bias = tensors[Input[2]];
		CheckNeoOnnxSupport( bias.Data != nullptr, "non-constant bias", OnnxNode );
		CheckOnnxProtocol( bias.Shape.Size() == 1, "bias tensor must be 1-dimensional", OnnxNode );
		CheckOnnxProtocol( bias.Shape[0] == filterCount, "bias tensor's size mu be equal to filter count", OnnxNode );
	}

	// Getting kernel shape.
	CTensorShape kernelShape;
	kernelShape.SetBufferSize( inputShape.Size() );
	for( int dimIndex = 2; dimIndex < inputShape.Size(); ++dimIndex ) {
		kernelShape.Add( weight.Shape[dimIndex] );
	}

	if( autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER" ) {
		CalculatePadding( autoPad, inputShape, kernelShape, pads );
	}

	for( int dimIndex = 0; dimIndex < convDims; ++dimIndex ) {
		// NeoML doesn't support dilation in channelwise convolution.
		CheckNeoOnnxSupport( dilations[dimIndex] == 1 || group == 1, "dilations at channelwise conv", OnnxNode );
	}

	// Calculating output shape.
	CTensorShape& outputShape = tensors[Output[0]].Shape;
	inputShape.CopyTo( outputShape );
	if( group == 1 ) {
		outputShape[1] = filterCount;
	}
	for( int dimIndex = 0; dimIndex < convDims; ++dimIndex ) {
		outputShape[dimIndex + 2] = ( inputShape[dimIndex + 2] + pads[dimIndex] + pads[dimIndex + convDims]
			- ( kernelShape[dimIndex] - 1 ) * dilations[dimIndex] - 1 ) / strides[dimIndex] + 1;
	}

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
	// The tensors[Output[0]].Data was already set to nullptr in default constructor.
}

void CConvNode::MarkTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, { BD_BatchWidth, BD_Channels, BD_Height, BD_Width },
		dims[Output[0]] ), "marking output dimensions failed", OnnxNode );
	CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, { BD_BatchWidth, BD_Channels, BD_Height, BD_Width },
		dims[Input[0]] ), "marking output dimensions failed", OnnxNode );
}

void CConvNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CBaseConvLayer> conv = nullptr;

	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<CDnnBlob> filter;
	CPtr<CDnnBlob> freeTerm;

	const int filterCount = tensors[Input[1]].Shape[0];
	const int inputChannels = tensors[Input[0]].Shape[1];
	const int filterHeight = tensors[Input[1]].Shape[2];
	const int filterWidth = tensors[Input[1]].Shape[3];

	if( group == 1 ) {
		conv = new CConvLayer( mathEngine );
		filter = CDnnBlob::Create2DImageBlob( mathEngine, CT_Float, 1,
			filterCount, filterHeight, filterWidth, inputChannels );
		mathEngine.TransposeMatrix( filterCount, tensors[Input[1]].Data->GetData(), inputChannels, 1,
			filterHeight * filterWidth, 1, filter->GetData(), filter->GetDataSize() );
	} else {
		NeoAssert( filterCount == inputChannels );
		NeoAssert( group == inputChannels );
		conv = new CChannelwiseConvLayer( mathEngine );
		filter = CDnnBlob::Create2DImageBlob( mathEngine, CT_Float, 1, 1, filterHeight, filterWidth, filterCount );
		mathEngine.TransposeMatrix( 1, tensors[Input[1]].Data->GetData(), filterCount, 1, filterHeight * filterWidth,
			1, filter->GetData(), filter->GetDataSize() );
	}
	conv->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	conv->SetFilterCount( filterCount );
	conv->SetFilterHeight( filterHeight );
	conv->SetFilterWidth( filterWidth );

	// Getting strides.
	conv->SetStrideHeight( strides[0] );
	conv->SetStrideWidth( strides[1] );

	conv->SetPaddingHeight( pads[0] );
	conv->SetPaddingWidth( pads[1] );

	// Getting dilations.
	conv->SetDilationHeight( dilations[0] );
	conv->SetDilationWidth( dilations[1] );

	if( InputCount() == 3 ) {
		freeTerm = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, 1, filterCount );
		mathEngine.VectorCopy( freeTerm->GetData(), tensors[Input[2]].Data->GetData(), filterCount );
	}
	
	conv->SetFilterData( filter );
	conv->SetFreeTermData( freeTerm );

	conv->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *conv );

	neoMLLinks[Output[0]] = CNeoMLLink( conv, 0 );
}

} // namespace NeoOnnx
