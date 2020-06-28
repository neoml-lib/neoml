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

CConvNode::CConvNode( const onnx::NodeProto& conv, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( conv, nodeOutputs ),
	group( attributes.GetOptionalInt( "group", 1 ) ),
	autoPad( attributes.GetOptionalString( "auto_pad", "NOTSET" ) )
{
	CheckOnnxProtocol( input.Size() == 2 || input.Size() == 3, "node must have 2 or 3 inputs", conv );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", conv );

	attributes.GetOptionalIntArray( "strides", strides );
	attributes.GetOptionalIntArray( "pads", pads );
	attributes.GetOptionalIntArray( "dilations", dilations );
}

void CConvNode::OnnxReshape()
{
	// Checking input
	const CTensor& inputTensor = InputTensor( 0 );
	CheckNeoOnnxSupport( inputTensor.GetShape().Size() > 2 && inputTensor.GetShape().Size() <= 4,
		"wrong input tensor's dimensions number", onnxNode );
	const CTensorShape& inputShape = inputTensor.GetShape();
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
		"grouped convolutiion (non-channelwise)", onnxNode );

	// Checking weights
	const CTensor& weight = InputTensor( 1 );
	CheckNeoOnnxSupport( weight.GetType() == TT_ConstantTensor, "non-constant weights", onnxNode );
	CheckOnnxProtocol( weight.GetShape().Size() == convDims + 2,
		"wrong weight tensor's dimensions number", onnxNode );
	const int filterCount = weight.GetShape()[0];
	CheckOnnxProtocol( weight.GetShape()[group == 1 ? 1 : 0] == inputShape[1],
		"wrong weight tensor's size", onnxNode );

	// Checking bias
	if( input.Size() == 3 ) {
		const CTensor& bias = InputTensor( 2 );
		CheckNeoOnnxSupport( bias.GetType() == TT_ConstantTensor, "non-constant bias", onnxNode );
		CheckOnnxProtocol( bias.GetShape().Size() == 1, "bias tensor must be 1-dimensional", onnxNode );
		CheckOnnxProtocol( bias.GetShape()[0] == filterCount, "bias tensor's size mu be equal to filter count", onnxNode );
	}

	// Getting kernel shape.
	CTensorShape kernelShape;
	kernelShape.SetBufferSize( inputShape.Size() );
	for( int dimIndex = 2; dimIndex < inputShape.Size(); ++dimIndex ) {
		kernelShape.Add( weight.GetShape()[dimIndex] );
	}

	if( autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER" ) {
		CalculatePadding( autoPad, inputShape, kernelShape, pads );
	}

	for( int dimIndex = 0; dimIndex < convDims; ++dimIndex ) {
		// NeoML doesn't support dilation in channelwise convolution.
		CheckNeoOnnxSupport( dilations[dimIndex] == 1 || group == 1, "dilations at channelwise conv", onnxNode );
	}

	// Calculating output shape.
	CTensorShape outputShape;
	inputShape.CopyTo( outputShape );
	if( group == 1 ) {
		outputShape[1] = filterCount;
	}
	for( int dimIndex = 0; dimIndex < convDims; ++dimIndex ) {
		outputShape[dimIndex + 2] = ( inputShape[dimIndex + 2] + pads[dimIndex] + pads[dimIndex + convDims]
			- ( kernelShape[dimIndex] - 1 ) * dilations[dimIndex] - 1 ) / strides[dimIndex] + 1;
	}
	outputData.Add( CTensor( TT_DataTensor, outputShape ) );
}

void CConvNode::MarkTensorDims()
{
	CheckNeoOnnxInternal( outputData[0].SetTensorDim( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width } ),
		"marking output dimensions failed", onnxNode );
	CheckNeoOnnxInternal( InputTensor( 0 ).SetTensorDim( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width } ),
		"marking input dimensions failed", onnxNode );
}

void CConvNode::AddLayers( CDnn& dnn )
{
	CPtr<CBaseConvLayer> conv = nullptr;

	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<CDnnBlob> filter;
	CPtr<CDnnBlob> freeTerm;

	const int filterCount = InputTensor( 1 ).GetShape()[0];
	const int inputChannels = InputTensor( 0 ).GetShape()[1];
	const int filterHeight = InputTensor( 1 ).GetShape()[2];
	const int filterWidth = InputTensor( 1 ).GetShape()[3];

	if( group == 1 ) {
		conv = new CConvLayer( mathEngine );
		filter = CDnnBlob::Create2DImageBlob( mathEngine, CT_Float, 1,
			filterCount, filterHeight, filterWidth, inputChannels );
		mathEngine.TransposeMatrix( filterCount, InputTensor( 1 ).GetData()->GetData(), inputChannels, 1,
			filterHeight * filterWidth, 1, filter->GetData(), filter->GetDataSize() );
	} else {
		NeoAssert( filterCount == inputChannels );
		NeoAssert( group == inputChannels );
		conv = new CChannelwiseConvLayer( mathEngine );
		filter = CDnnBlob::Create2DImageBlob( mathEngine, CT_Float, 1, 1, filterHeight, filterWidth, filterCount );
		mathEngine.TransposeMatrix( 1, InputTensor( 1 ).GetData()->GetData(), filterCount, 1, filterHeight * filterWidth,
			1, filter->GetData(), filter->GetDataSize() );
	}
	conv->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	conv->SetFilterCount( filterCount );
	conv->SetFilterHeight( filterHeight );
	conv->SetFilterWidth( filterWidth );

	// Getting strides.
	conv->SetStrideHeight( strides[0] );
	conv->SetStrideWidth( strides[1] );

	// Getting kernel shape.
	const CTensorShape& inputShape = InputTensor( 0 ).GetShape();
	CTensorShape kernelShape;
	kernelShape.SetBufferSize( inputShape.Size() );
	for( int dimIndex = 2; dimIndex < inputShape.Size(); ++dimIndex ) {
		kernelShape.Add( InputTensor( 1 ).GetShape()[dimIndex] );
	}

	conv->SetPaddingHeight( pads[0] );
	conv->SetPaddingWidth( pads[1] );

	// Getting dilations.
	conv->SetDilationHeight( dilations[0] );
	conv->SetDilationWidth( dilations[1] );

	if( input.Size() == 3 ) {
		freeTerm = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, 1, filterCount );
		mathEngine.VectorCopy( freeTerm->GetData(), InputTensor( 2 ).GetData()->GetData(), filterCount );
	}
	
	conv->SetFilterData( filter );
	conv->SetFreeTermData( freeTerm );

	conv->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	dnn.AddLayer( *conv );

	outputInfo.Add( COutputInfo( conv, 0 ) );
}

} // namespace NeoOnnx
