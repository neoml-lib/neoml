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
#include "PadNode.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CConvNode::CConvNode( const onnx::NodeProto& conv, int opsetVersion ) :
	COpNode( conv, opsetVersion ),
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

void CConvNode::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	const CTensorShape& inputShape = inputs[0]->Shape();
	CheckNeoOnnxSupport( inputShape.Size() > 2 && inputShape.Size() <= 5,
		"wrong input tensor's dimensions number", OnnxNode );
	const int convDims = static_cast<int>( inputShape.Size() ) - 2;

	if( strides.IsEmpty() ) {
		strides.Add( 1, convDims );
	}
	if( pads.IsEmpty() ) {
		pads.Add( 0, 2 * convDims );
	}
	if( dilations.IsEmpty() ) {
		dilations.Add( 1, convDims );
	}

	// Check groups (NeoML supports only 2D, 3D and Channelwise2D convolutions)
	CheckNeoOnnxSupport( group == 1 || ( group == inputShape[1] && convDims < 3 ),
		"grouped convolutiion (non-channelwise)", OnnxNode );

	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "user-provided weights", OnnxNode );
	if( InputCount() == 3 && inputs[2] != nullptr ) {
		CheckNeoOnnxSupport( inputs[2]->IsCalculated(), "user-provided bias", OnnxNode );
	}

	// Calculate padding
	CTensorShape kernelShape;
	kernelShape.SetBufferSize( inputShape.Size() - 2 );
	for( int dimIndex = 2; dimIndex < inputShape.Size(); ++dimIndex ) {
		kernelShape.Add( inputs[1]->Shape()[dimIndex] );
	}
	if( autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER" ) {
		CalculatePadding( autoPad, kernelShape, pads );
	}

	inputShape.CopyTo( outputShape );
	if( group == 1 ) {
		outputShape[1] = inputs[1]->Shape()[0];
	}
	for( int dimIndex = 0; dimIndex < convDims; ++dimIndex ) {
		outputShape[dimIndex + 2] = ( inputShape[dimIndex + 2] + pads[dimIndex] + pads[dimIndex + convDims]
			- ( kernelShape[dimIndex] - 1 ) * dilations[dimIndex] - 1 ) / strides[dimIndex] + 1;
	}

	if( inputShape.Size() == 4 ) {
		add2dConvLayer( inputs, outputs, dnn );
	} else if( inputShape.Size() == 5 ) {
		add3dConvLayer( inputs, outputs, dnn );
	} else {
		CheckNeoOnnxSupport( false, "1-dimensional conv", OnnxNode );
	}
}

// Adds 2-dimensional convolution
void CConvNode::add2dConvLayer( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	const CTensorLayout neoML2dLayout( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width } );

	CPtr<CBaseConvLayer> conv = nullptr;
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<const CDataTensor> filter = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() );
	const CTensorShape& filterShape = filter->Shape();
	const int filterCount = filterShape[0];
	const int inputChannels = inputs[0]->Shape()[1];
	const int filterHeight = filterShape[2];
	const int filterWidth = filterShape[3];

	if( group == 1 ) {
		conv = new CConvLayer( mathEngine );
		filter = dynamic_cast<const CDataTensor*>( ConvertTensor( *inputs[1], neoML2dLayout ).Ptr() );
	} else {
		CheckNeoOnnxSupport( filterCount == inputChannels, "non-trivial grouped conv", OnnxNode);
		CheckNeoOnnxSupport( group == inputChannels, "non-trivial grouped conv", OnnxNode );
		conv = new CChannelwiseConvLayer( mathEngine );
		// In channelwise convolution filter has specific layout
		const CTensorLayout filterLayout( { BD_Channels, BD_BatchWidth, BD_Height, BD_Width } );
		filter = dynamic_cast<const CDataTensor*>( ConvertTensor( *inputs[1], filterLayout ).Ptr() );
	}
	
	conv->SetName( Name() );
	conv->SetFilterCount( filterCount );
	conv->SetFilterHeight( filterHeight );
	conv->SetFilterWidth( filterWidth );
	conv->SetStrideHeight( strides[0] );
	conv->SetStrideWidth( strides[1] );
	CPtr<const CUserTensor> currInput = dynamic_cast<const CUserTensor*>( ConvertTensor( *inputs[0], neoML2dLayout ).Ptr() );

	if( pads[0] >= pads[2] && pads[1] >= pads[3] ) {
		// This is a valid case for convolution in NeoML
		conv->SetPaddingHeight( pads[0] );
		conv->SetPaddingWidth( pads[1] );
	} else {
		// In this case we have to add explicit padding layer
		currInput = PadUserTensor( *currInput, pads, 0.f );
	}
	conv->SetDilationHeight( dilations[0] );
	conv->SetDilationWidth( dilations[1] );

	conv->SetFilterData( filter->Data()->GetCopy() );
	if( InputCount() == 3 && inputs[2] != nullptr ) {
		conv->SetFreeTermData( dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data()->GetCopy() );
	} else {
		conv->SetZeroFreeTerm( false );
	}

	conv->Connect( 0, *currInput->Layer(), currInput->OutputIndex() );
	dnn.AddLayer( *conv );

	outputs[0] = new CUserTensor( outputShape, neoML2dLayout, CLayerOutput( conv, 0 ) );
}

// Adds 3-dimensional convolution
void CConvNode::add3dConvLayer( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	const CTensorLayout neoML3dLayout( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width, BD_Depth } );

	const CTensorShape& filterShape = inputs[1]->Shape();
	const int filterCount = filterShape[0];
	const int inputChannels = inputs[0]->Shape()[1];
	const int filterHeight = filterShape[2];
	const int filterWidth = filterShape[3];
	const int filterDepth = filterShape[4];

	CheckNeoOnnxSupport( group == 1, "groupped 3d convolution", OnnxNode );
	for( int dimIndex = 0; dimIndex < dilations.Size(); ++dimIndex ) {
		CheckNeoOnnxSupport( dilations[dimIndex] == 1, "dilated 3d convolution", OnnxNode );
	}

	CPtr<C3dConvLayer> conv = new C3dConvLayer( dnn.GetMathEngine() );
	conv->SetName( Name() );
	conv->SetFilterCount( filterCount );
	conv->SetFilterHeight( filterHeight );
	conv->SetFilterWidth( filterWidth );
	conv->SetFilterDepth( filterDepth );
	conv->SetStrideHeight( strides[0] );
	conv->SetStrideWidth( strides[1] );
	conv->SetStrideDepth( strides[2] );
	CPtr<const CUserTensor> currInput = dynamic_cast<const CUserTensor*>( ConvertTensor( *inputs[0], neoML3dLayout ).Ptr() );
	
	if( pads[0] >= pads[3] && pads[1] >= pads[4] && pads[2] >= pads[5] ) {
		// This is a valid case for convolution NeoML
		conv->SetPaddingHeight( pads[0] );
		conv->SetPaddingWidth( pads[1] );
		conv->SetPaddingDepth( pads[2] );
	} else {
		// In this case we have to add explicit padding layer
		currInput = PadUserTensor( *currInput, pads, 0.f );
	}

	CPtr<const CDataTensor> filter = dynamic_cast<const CDataTensor*>( ConvertTensor( *inputs[1], neoML3dLayout ).Ptr() );
	conv->SetFilterData( filter->Data()->GetCopy() );
	if( InputCount() == 3 && inputs[2] != nullptr ) {
		conv->SetFreeTermData( dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data()->GetCopy() );
	} else {
		conv->SetZeroFreeTerm( false );
	}

	conv->Connect( 0, *currInput->Layer(), currInput->OutputIndex() );
	dnn.AddLayer( *conv );

	outputs[0] = new CUserTensor( outputShape, neoML3dLayout, CLayerOutput( conv, 0 ) );
}

} // namespace NeoOnnx
