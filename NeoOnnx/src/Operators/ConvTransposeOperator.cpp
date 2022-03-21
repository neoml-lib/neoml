/* Copyright © 2017-2022 ABBYY Production LLC

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

#include "ConvTransposeOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

namespace NeoOnnx {

// Gets kernel shape
static void getKernelShape( const CTensorArray& inputs, CTensorShape& kernelShape )
{
	const CTensorShape& inputShape = inputs[0]->Shape();
	kernelShape.SetBufferSize( inputShape.Size() - 2 );
	for( int dimIndex = 2; dimIndex < inputShape.Size(); ++dimIndex ) {
		kernelShape.Add( inputs[1]->Shape()[dimIndex] );
	}
}

// Calculates output shape based on the convTranspose parameters
static void calcOutputShape( const CTensorArray& inputs, const CTensorShape& kernelShape, const CFastArray<int, 8>& strides,
	const CFastArray<int, 8>& pads, const CFastArray<int, 8>& dilations, CTensorShape& outputShape )
{
	const CTensorShape& inputShape = inputs[0]->Shape();
	inputShape.CopyTo( outputShape );
	outputShape[1] = inputs[1]->Shape()[1];
	const int convDims = inputShape.Size() - 2;
	for( int dimIndex = 0; dimIndex < convDims; ++dimIndex ) {
		outputShape[dimIndex + 2] = strides[dimIndex] * ( inputShape[dimIndex + 2] - 1 )
			+ ( kernelShape[dimIndex] - 1 ) * dilations[dimIndex] + 1
			- pads[dimIndex] - pads[dimIndex + convDims];
	}
}

// --------------------------------------------------------------------------------------------------------------------

CConvTransposeOperator::CConvTransposeOperator( const onnx::NodeProto& convTranspose, int opsetVersion ) :
	CLayerOperator( convTranspose, opsetVersion )
{
	// v1 - initial version of the operator
	// v11 - changes in 'auto_pad' processing and new data types are supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 2 || InputCount() == 3, "operator must have 2 or 3 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );

	int group = 1;
	GetAttribute( "group", group );
	CheckNeoOnnxSupport( group == 1, "groupped ConvTranspose", *this );

	CString autoPad = "NOTSET";
	GetAttribute( "auto_pad", autoPad );
	CheckNeoOnnxSupport( autoPad == "NOTSET", "auto_pad", *this );
}

void CConvTransposeOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	const CTensorShape& inputShape = inputs[0]->Shape();
	CheckNeoOnnxSupport( inputShape.Size() == 4, "non-2d convTranspose", *this );
	CheckOnnxProtocol( inputs[1] != nullptr, "input can't be optional", *this );
	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "user-provided weights", *this );
	if( InputCount() == 3 && inputs[2] != nullptr ) {
		CheckNeoOnnxSupport( inputs[2]->IsCalculated(), "user-provided bias", *this );
	}

	// ConvTranspose has 2 ways of setting convolution parameters
	// 1. Padding is given, output size is calculated
	// 2. Output size is given, padding is calculated
	// NeoOnnx supports only first scenario
	CTensorShape kernelShape;
	getKernelShape( inputs, kernelShape );
	CFastArray<int, 8> strides;
	getStrides( inputs, strides );
	CFastArray<int, 8> pads;
	getPads( inputs, kernelShape, pads );
	CFastArray<int, 8> dilations;
	getDilations( inputs, dilations );
	CTensorShape outputShape;
	calcOutputShape( inputs, kernelShape, strides, pads, dilations, outputShape );

	CTensorLayout neoMLLayout( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width } );
	CPtr<const CDataTensor> filter = dynamic_cast<const CDataTensor*>( ConvertTensor( *inputs[1], neoMLLayout ).Ptr() );
	const int filterCount = filter->Shape()[1];
	const int inputChannels = inputs[0]->Shape()[1];

	IMathEngine& mathEngine = dnn.GetMathEngine();
	CPtr<CTransposedConvLayer> transposedConv = new CTransposedConvLayer( mathEngine );
	transposedConv->SetName( Name() );
	transposedConv->SetFilterCount( filterCount );
	transposedConv->SetFilterHeight( kernelShape[0] );
	transposedConv->SetFilterWidth( kernelShape[1] );
	transposedConv->SetStrideHeight( strides[0] );
	transposedConv->SetStrideWidth( strides[1] );
	CPtr<const CUserTensor> currInput = AsUserTensor( *ConvertTensor( *inputs[0], neoMLLayout ), Name() + "_Source", dnn );

	if( pads[0] >= pads[2] && pads[1] >= pads[3] ) {
		// This is a valid padding for a convolution in NeoML
		transposedConv->SetPaddingHeight( pads[0] );
		transposedConv->SetPaddingWidth( pads[1] );
	} else {
		// In NeoML convolution doesn't support cases when bottom padding is larger than upper padding
		// (the same goes for other spatial dimensions)
		// In this case we have to add explicit padding layer
		currInput = PadUserTensor( *currInput, pads, 0.f );
	}
	transposedConv->SetDilationHeight( dilations[0] );
	transposedConv->SetDilationWidth( dilations[1] );

	transposedConv->SetFilterData( filter->Data()->GetCopy() );
	if( InputCount() == 3 && inputs[2] != nullptr ) {
		transposedConv->SetFreeTermData( dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data()->GetCopy() );
	} else {
		transposedConv->SetZeroFreeTerm( true );
	}

	transposedConv->Connect( 0, *currInput->Layer(), currInput->OutputIndex() );
	dnn.AddLayer( *transposedConv );
	outputs.Add( new CUserTensor( outputShape, neoMLLayout, CLayerOutput( transposedConv, 0 ) ) );
}

// Gets strides
void CConvTransposeOperator::getStrides( const CTensorArray& inputs, CFastArray<int, 8>& strides ) const
{
	GetAttribute( "strides", strides );
	if( strides.IsEmpty() ) {
		const int convDims = static_cast<int>( inputs[0]->Shape().Size() ) - 2;
		strides.Add( 1, convDims );
	}
}

// Gets padding sizes
void CConvTransposeOperator::getPads( const CTensorArray& inputs, const CTensorShape& kernelShape, CFastArray<int, 8>& pads ) const
{
	GetAttribute( "pads", pads );
	if( pads.IsEmpty() ) {
		const int convDims = static_cast<int>( inputs[0]->Shape().Size() ) - 2;
		pads.Add( 0, 2 * convDims );
	}
}

// Gets dilation sizes
void CConvTransposeOperator::getDilations( const CTensorArray& inputs, CFastArray<int, 8>& dilations ) const
{
	GetAttribute( "dilations", dilations );
	if( dilations.IsEmpty() ) {
		const int convDims = inputs[0]->Shape().Size() - 2;
		dilations.Add( 1, convDims );
	}
}

} // namespace NeoOnnx
