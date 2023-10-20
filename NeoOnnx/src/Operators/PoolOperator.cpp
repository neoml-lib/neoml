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

#include "onnx.pb.h"

#include "PoolOperator.h"
#include "TensorUtils.h"

namespace NeoOnnx {

CPoolOperatorBase::CPoolOperatorBase( const onnx::NodeProto& poolNode, int opsetVersion ) :
	CLayerOperator( poolNode, opsetVersion ),
	autoPad( "NOTSET" )
{
	// The difference between versions are in rarely used attributes (not supported by NeoOnnx): ceil_mode, storage_order etc)
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1 || OutputCount() == 2, "operator must have 1 or 2 outputs", *this );

	CheckOnnxProtocol( GetAttribute( "kernel_shape", kernelShape ), "'kernel_shape' attribute is missing", *this );
	CheckNeoOnnxSupport( kernelShape.Size() == 2, "non 2-dimensional max pooling", *this );

	GetAttribute( "auto_pad", autoPad );
}

void CPoolOperatorBase::GetPads( const CTensorArray& inputs, CFastArray<int, 8>& pads ) const
{
	GetAttribute( "pads", pads );

	if( pads.IsEmpty() ) {
		pads.Add( 0, 2 * ( inputs[0]->DimCount() - 2 ) );
	}

	if( autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER" ) {
		CalculatePadding( autoPad, kernelShape, pads );
	}
}

void CPoolOperatorBase::AddLayersImpl( const CTensorArray& inputs, float padValue,
	CPoolingLayer& pooling, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNoShapeInputs( inputs );
	// Check input
	CheckNeoOnnxSupport( inputs[0]->DimCount() > 2 && inputs[0]->DimCount() <= 4,
		"wrong input tensor's dimensions number", *this );

	// Initialize strides and pads(if not given)
	CFastArray<int, 8> strides;
	getStrides( inputs, strides );
	CFastArray<int, 8> pads;
	GetPads( inputs, pads );

	pooling.SetName( Name() );

	CPtr<const CUserTensor> input = AsUserTensor( *ConvertTensor( *inputs[0], CNeoMLImageLayoutValidator() ),
		Name() + "_Source", dnn );
	input = PadUserTensor( *input, pads, TBlobResizePadding::Constant, padValue );

	pooling.SetFilterHeight( kernelShape[0] );
	pooling.SetFilterWidth( kernelShape[1] );

	pooling.SetStrideHeight( strides[0] );
	pooling.SetStrideWidth( strides[1] );

	pooling.Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( pooling );

	outputs.Add( new CUserTensor( input->Layout(), CLayerOutput( &pooling, 0 ) ) );
	if( OutputCount() > outputs.Size() ) {
		outputs.Add( nullptr, OutputCount() - outputs.Size() );
	}
}

// Gets pool strides
void CPoolOperatorBase::getStrides( const CTensorArray& inputs, CFastArray<int, 8>& strides ) const
{
	GetAttribute( "strides", strides );

	if( strides.IsEmpty() ) {
		strides.Add( 1, inputs[0]->DimCount() - 2 );
	}
}

// --------------------------------------------------------------------------------------------------------------------

void CMaxPoolOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CPtr<CPoolingLayer> pooling( new CMaxPoolingLayer( dnn.GetMathEngine() ) );
	CPoolOperatorBase::AddLayersImpl( inputs, -FLT_MAX, *pooling, dnn, outputs );
}

// --------------------------------------------------------------------------------------------------------------------

CAveragePoolOperator::CAveragePoolOperator( const onnx::NodeProto& averagePool, int opsetVersion ) :
	CPoolOperatorBase( averagePool, opsetVersion ),
	includePad( false )
{
	// v1 - initial version
	// v7 - added count_include_pad
	// v10 - added ceil mode
	// v19 - added dilations
	if( OpsetVersion >= 7 ) {
		int countIncludePad = 0;
		GetAttribute( "count_include_pad", countIncludePad );
		includePad = countIncludePad != 0;
	}
	if( OpsetVersion >= 10 ) {
		int ceilMode = 0;
		GetAttribute( "ceil_mode", ceilMode );
		CheckNeoOnnxSupport( ceilMode == 0, "ceil_mode", *this );
	}
	if( OpsetVersion >= 19 ) {
		CArray<int> dilations;
		GetAttribute( "dilations", dilations );
		for( const int dilation : dilations ) {
			CheckNeoOnnxSupport( dilation == 1, "non-trivial dilation", *this );
		}
	}
}

void CAveragePoolOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	if( !includePad ) {
		// There are 2 ways to calculate average pooling with padding:
		//     1. padding affects only output size, but padding values aren't included in the averages
		//     2. padding affects both output size and average calculation
		// In NeoML pooling doesn't support paddings, that's why only second option is available (via explicit padding of input tensor)
		CFastArray<int, 8> pads;
		GetPads( inputs, pads );
		for( int padIndex = 0; padIndex < pads.Size(); ++padIndex ) {
			CheckNeoOnnxSupport( pads[padIndex] == 0, "average pooling with padding not included in calc", *this );
		}
	}

	CPtr<CPoolingLayer> pooling( new CMeanPoolingLayer( dnn.GetMathEngine() ) );
	CPoolOperatorBase::AddLayersImpl( inputs, 0.f, *pooling, dnn, outputs );
}

} // namespace NeoOnnx

