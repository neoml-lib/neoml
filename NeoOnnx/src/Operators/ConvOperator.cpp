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

#include "onnx.pb.h"

#include "ConvOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

namespace NeoOnnx {

// Gets kernel shape
static void getConvKernelShape( int dimCount, const CDataTensor& filter, CTensorShape& kernelShape )
{
	kernelShape.SetBufferSize( dimCount - 2 );
	for( int dimIndex = 2; dimIndex < dimCount; ++dimIndex ) {
		kernelShape.Add( filter.DimSize( dimIndex ) );
	}
}

// --------------------------------------------------------------------------------------------------------------------

CConvOperator::CConvOperator( const onnx::NodeProto& conv, int opsetVersion ) :
	CLayerOperator( conv, opsetVersion ),
	group( 1 ),
	autoPad( "NOTSET" )
{
	// v1 - original
	// v11 - default values for 'strides' and 'dilations' attributes are added
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 2 || InputCount() == 3, "operator must have 2 or 3 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );

	GetAttribute( "group", group );
	GetAttribute( "auto_pad", autoPad );
}

void CConvOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoShapeInputs( inputs );

	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );

	CheckNeoOnnxSupport( inputs[0]->DimCount() > 2 && inputs[0]->DimCount() <= 5,
		"wrong input tensor's dimensions number", *this );

	CheckOnnxProtocol( inputs[1] != nullptr, "input can't be optional", *this );
	CheckNeoOnnxSupport( inputs[1]->Type() == TTensorType::Data, "user-provided weights", *this );
	if( InputCount() == 3 && inputs[2] != nullptr ) {
		CheckNeoOnnxSupport( inputs[2]->Type() == TTensorType::Data, "user-provided bias", *this );
	}

	if( inputs[0]->DimCount() == 4 ) {
		add2dConvLayer( inputs, false, dnn, outputs );
	} else if( inputs[0]->DimCount() == 5 ) {
		add3dConvLayer( inputs, dnn, outputs );
	} else if ( inputs[0]->DimCount() == 3 ) {
		add2dConvLayer( inputs, true, dnn, outputs );
	} else {
		CheckNeoOnnxSupport( false, "3+-dimensional convolution", *this );
	}
}

// Adds 2-dimensional convolution (also used for emulation of 1-dimensional convolution)
void CConvOperator::add2dConvLayer( const CTensorArray& inputs, bool is1dConv, CDnn& dnn, CTensorArray& outputs ) const
{
	CPtr<const CDataTensor> filter = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() );

	CTensorShape kernelShape;
	getConvKernelShape( inputs[0]->DimCount(), *filter, kernelShape);
	CFastArray<int, 8> strides;
	getStrides( inputs, strides );
	CFastArray<int, 8> pads;
	getPads( inputs, kernelShape, pads );
	CFastArray<int, 8> dilations;
	getDilations( inputs, dilations );

	CPtr<CBaseConvLayer> conv = nullptr;
	IMathEngine& mathEngine = dnn.GetMathEngine();

	const int filterCount = filter->DimSize( 0 );
	if( group == 1 ) {
		// Non-groupped convolution can be calculated via CConvLayer
		CTensorLayout neoMLLayout( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width } );
		if( is1dConv ) {
			neoMLLayout.SetSize( 3 );
		}

		conv = new CConvLayer( mathEngine );
		filter = dynamic_cast<const CDataTensor*>( ConvertTensor( *inputs[1], neoMLLayout ).Ptr() );
	} else {
		// Other cases of groupped convolution aren't supported by NeoML
		CheckNeoOnnxSupport( filter->DimSize( 1 ) == 1, "non-trivial groupped conv", *this );
		CheckNeoOnnxSupport( inputs[0]->DimCount() <= 4, "3D+ channelwise convolution");
		conv = new CChannelwiseConvLayer( mathEngine );
		// In channelwise convolution filter has specific layout
		CTensorLayout filterLayout( { BD_Channels, BD_BatchWidth, BD_Height, BD_Width } );
		if( is1dConv ) {
			filterLayout.SetSize( 3 );
		}
		filter = dynamic_cast<const CDataTensor*>( ConvertTensor( *inputs[1], filterLayout ).Ptr() );
	}
	
	conv->SetName( Name() );
	conv->SetFilterCount( filterCount );
	conv->SetFilterHeight( kernelShape[0] );
	conv->SetFilterWidth( is1dConv ? 1 : kernelShape[1] );
	conv->SetStrideHeight( strides[0] );
	conv->SetStrideWidth( is1dConv ? 1 : strides[1] );

	CPtr<const CUserTensor> currInput = AsUserTensor( *ConvertTensor( *inputs[0], CNeoMLImageLayoutValidator() ),
		Name() + "_Source", dnn );
	if( ( is1dConv && pads[0] >= pads[1] )
		|| ( !is1dConv && pads[0] >= pads[2] && pads[1] >= pads[3] ) )
	{
		// This is a valid padding for a convolution in NeoML
		conv->SetPaddingHeight( pads[0] );
		conv->SetPaddingWidth( is1dConv ? 0 : pads[1] );
	} else {
		// In NeoML convolution doesn't support cases when bottom padding is larger than upper padding
		// (the same goes for other spatial dimensions)
		// In this case we have to add explicit padding layer
		currInput = PadUserTensor( *currInput, pads, TBlobResizePadding::Constant, 0.f );
	}
	conv->SetDilationHeight( dilations[0] );
	conv->SetDilationWidth( is1dConv ? 1 : dilations[1] );

	conv->SetFilterData( filter->Data()->GetCopy() );
	if( InputCount() == 3 && inputs[2] != nullptr ) {
		conv->SetFreeTermData( dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data()->GetCopy() );
	} else {
		conv->SetZeroFreeTerm( true );
	}

	conv->Connect( 0, *currInput->Layer(), currInput->OutputIndex() );
	dnn.AddLayer( *conv );

	outputs.Add( new CUserTensor( currInput->Layout(), CLayerOutput( conv, 0 ) ) );
}

// Adds 3-dimensional convolution
void CConvOperator::add3dConvLayer( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CTensorLayout neoMLLayout( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width, BD_Depth } );
	CPtr<const CDataTensor> filter = dynamic_cast<const CDataTensor*>(
		ConvertTensor( *inputs[1], neoMLLayout ).Ptr() );

	CTensorShape kernelShape;
	getConvKernelShape( inputs[0]->DimCount(), *filter, kernelShape );
	CFastArray<int, 8> strides;
	getStrides( inputs, strides );
	CFastArray<int, 8> pads;
	getPads( inputs, kernelShape, pads );
	CFastArray<int, 8> dilations;
	getDilations( inputs, dilations );

	// In NeoML there is no channelwise convolution for 3-dimensional images
	CheckNeoOnnxSupport( group == 1, "groupped 3d convolution", *this );
	for( int dimIndex = 0; dimIndex < dilations.Size(); ++dimIndex ) {
		CheckNeoOnnxSupport( dilations[dimIndex] == 1, "dilated 3d convolution", *this );
	}

	CPtr<C3dConvLayer> conv = new C3dConvLayer( dnn.GetMathEngine() );
	conv->SetName( Name() );
	conv->SetFilterCount( filter->DimSize( 0 ) );
	conv->SetFilterHeight( kernelShape[0] );
	conv->SetFilterWidth( kernelShape[1] );
	conv->SetFilterDepth( kernelShape[2] );
	conv->SetStrideHeight( strides[0] );
	conv->SetStrideWidth( strides[1] );
	conv->SetStrideDepth( strides[2] );
	
	CPtr<const CUserTensor> currInput = AsUserTensor( *ConvertTensor( *inputs[0], CNeoMLImageLayoutValidator() ),
		Name() + "_Source", dnn );
	if( pads[0] >= pads[3] && pads[1] >= pads[4] && pads[2] >= pads[5] ) {
		// This is a valid padding for a convolution in NeoML
		conv->SetPaddingHeight( pads[0] );
		conv->SetPaddingWidth( pads[1] );
		conv->SetPaddingDepth( pads[2] );
	} else {
		// In NeoML convolution doesn't support cases when bottom padding is larger than upper padding
		// (the same goes for other spatial dimensions)
		// In this case we have to add explicit padding layer
		currInput = PadUserTensor( *currInput, pads, TBlobResizePadding::Constant, 0.f );
	}

	conv->SetFilterData( filter->Data()->GetCopy() );
	if( InputCount() == 3 && inputs[2] != nullptr ) {
		conv->SetFreeTermData( dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data()->GetCopy() );
	} else {
		conv->SetZeroFreeTerm( true );
	}

	conv->Connect( 0, *currInput->Layer(), currInput->OutputIndex() );
	dnn.AddLayer( *conv );

	outputs.Add( new CUserTensor( currInput->Layout(), CLayerOutput( conv, 0 ) ) );
}

// Gets strides
void CConvOperator::getStrides( const CTensorArray& inputs, CFastArray<int, 8>& strides ) const
{
	GetAttribute( "strides", strides );
	if( strides.IsEmpty() ) {
		const int convDims = static_cast<int>( inputs[0]->DimCount() ) - 2;
		strides.Add( 1, convDims );
	}
}

// Gets padding sizes
void CConvOperator::getPads( const CTensorArray& inputs, const CTensorShape& kernelShape, CFastArray<int, 8>& pads ) const
{
	GetAttribute( "pads", pads );
	if( pads.IsEmpty() ) {
		const int convDims = static_cast<int>( inputs[0]->DimCount() ) - 2;
		pads.Add( 0, 2 * convDims );
	}
	if( autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER" ) {
		CalculatePadding( autoPad, kernelShape, pads );
	}
}

// Gets dilation sizes
void CConvOperator::getDilations( const CTensorArray& inputs, CFastArray<int, 8>& dilations ) const
{
	GetAttribute( "dilations", dilations );
	if( dilations.IsEmpty() ) {
		const int convDims = static_cast<int>( inputs[0]->DimCount() ) - 2;
		dilations.Add( 1, convDims );
	}
}

} // namespace NeoOnnx

