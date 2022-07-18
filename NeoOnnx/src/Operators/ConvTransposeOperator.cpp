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
static void getConvTransposeKernelShape( const CTensorArray& inputs, CTensorShape& kernelShape )
{
	const CTensorShape& inputShape = inputs[0]->Shape();
	kernelShape.SetBufferSize( inputShape.Size() - 2 );
	for( int dimIndex = 2; dimIndex < inputShape.Size(); ++dimIndex ) {
		kernelShape.Add( inputs[1]->Shape()[dimIndex] );
	}
}

// Calculates output shape based on the convTranspose parameters
static void calcOutputShape( const CTensorArray& inputs, const CTensorShape& kernelShape, const CFastArray<int, 8>& strides,
	const CFastArray<int, 8>& dilations, CTensorShape& outputShape )
{
	const CTensorShape& inputShape = inputs[0]->Shape();
	inputShape.CopyTo( outputShape );
	outputShape[1] = inputs[1]->Shape()[1];
	const int convDims = inputShape.Size() - 2;
	for( int dimIndex = 0; dimIndex < convDims; ++dimIndex ) {
		outputShape[dimIndex + 2] = strides[dimIndex] * ( inputShape[dimIndex + 2] - 1 )
			+ ( kernelShape[dimIndex] - 1 ) * dilations[dimIndex];
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

	// Apply output_padding (in conv_transpose it's a padding which is applied before de-convolution
	CTensorShape kernelShape;
	getConvTransposeKernelShape( inputs, kernelShape );
	CFastArray<int, 8> strides;
	getStrides( inputs, strides );
	CFastArray<int, 8> dilations;
	getDilations( inputs, dilations );
	CTensorShape outputShape;
	calcOutputShape( inputs, kernelShape, strides, dilations, outputShape);

	CTensorLayout neoMLLayout( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width } );
	CPtr<const CDataTensor> filter = dynamic_cast<const CDataTensor*>( ConvertTensor( *inputs[1], neoMLLayout ).Ptr() );
	const int filterCount = filter->Shape()[1];

	IMathEngine& mathEngine = dnn.GetMathEngine();
	CPtr<CTransposedConvLayer> transposedConv = new CTransposedConvLayer( mathEngine );
	transposedConv->SetName( Name() );
	transposedConv->SetFilterCount( filterCount );
	transposedConv->SetFilterHeight( kernelShape[0] );
	transposedConv->SetFilterWidth( kernelShape[1] );
	transposedConv->SetStrideHeight( strides[0] );
	transposedConv->SetStrideWidth( strides[1] );
	transposedConv->SetDilationHeight( dilations[0] );
	transposedConv->SetDilationWidth( dilations[1] );

	transposedConv->SetFilterData( filter->Data()->GetCopy() );
	if( InputCount() == 3 && inputs[2] != nullptr ) {
		transposedConv->SetFreeTermData( dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data()->GetCopy() );
	} else {
		transposedConv->SetZeroFreeTerm( true );
	}

	CPtr<const CUserTensor> currTensor = AsUserTensor( *ConvertTensor( *inputs[0], neoMLLayout ), Name() + "_Source", dnn );
	transposedConv->Connect( 0, *currTensor->Layer(), currTensor->OutputIndex() );
	dnn.AddLayer( *transposedConv );
	currTensor = new CUserTensor( outputShape, neoMLLayout, CLayerOutput( transposedConv, 0 ) );
	currTensor = applyOutputPadding( *currTensor );
	outputs.Add( applyPads( *currTensor ).Ptr() );
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

// Gets dilation sizes
void CConvTransposeOperator::getDilations( const CTensorArray& inputs, CFastArray<int, 8>& dilations ) const
{
	GetAttribute( "dilations", dilations );
	if( dilations.IsEmpty() ) {
		const int convDims = inputs[0]->Shape().Size() - 2;
		dilations.Add( 1, convDims );
	}
}

// Applies output_padding to the result of the convolution (see ConvTranspose docs for more details)
CPtr<const CUserTensor> CConvTransposeOperator::applyOutputPadding( const CUserTensor& convOutput ) const
{
	CFastArray<int, 8> outputPadding;
	if( !GetAttribute( "output_padding", outputPadding ) ) {
		return &convOutput;
	}

	bool hasOutputPadding = false;
	for( int i = 0; i < outputPadding.Size(); ++i ) {
		if( outputPadding[i] != 0 ) {
			hasOutputPadding = true;
			break;
		}
	}

	if( !hasOutputPadding ) {
		return &convOutput;
	}

	NeoAssert( outputPadding.Size() == 2 );
	outputPadding.InsertAt( { 0, 0 }, 0 );
	return PadUserTensor( convOutput, outputPadding, 0.f ).Ptr();
}

// Applies pads on the result of the transposed conv (see ConvTranspose docs for more details)
CPtr<const CUserTensor> CConvTransposeOperator::applyPads( const CUserTensor& paddedOutput ) const
{
	const int convDims = paddedOutput.DimCount() - 2;
	CFastArray<int, 8> outputShape;
	CFastArray<int, 8> pads;

	// Getting pads sizes (one way, or another)
	const bool hasPadsAttribute = GetAttribute( "pads", pads );
	if( !hasPadsAttribute ) {
		if( !GetAttribute( "output_shape", outputShape ) ) {
			return &paddedOutput;
		}

		// Pads must be automatically calculated based on the output_shape
		NeoAssert( outputShape.Size() == paddedOutput.DimCount() );
		CString autoPad = "NOTSET";
		GetAttribute( "auto_pad", autoPad );

		pads.SetSize( 2 * convDims );
		for( int i = 0; i < convDims; ++i ) {
			const int totalPadding = paddedOutput.Shape()[i + 2] - outputShape[i + 2];
			pads[i] = autoPad != "SAME_UPPER" ? totalPadding / 2 : ( totalPadding + 1 ) / 2;
			pads[convDims + i] = totalPadding - pads[i];
		}
	}

	// Check that pads are not trivial
	NeoAssert( pads.Size() == 2 * convDims );
	bool hasPads = false;
	for( int i = 0; i < pads.Size(); ++i ) {
		if( pads[i] != 0 ) {
			hasPads = true;
			break;
		}
	}
	if( !hasPads ) {
		return &paddedOutput;
	}

	// Apply pads based on formula of the node output shape
	NeoAssert( pads.Size() == 4 );
	NeoAssert( paddedOutput.Layout()[2] == BD_Height && paddedOutput.Layout()[3] == BD_Width );
	for( int i = 0; i < pads.Size(); ++i ) {
		pads[i] = -pads[i];
	}
	return PadUserTensor( paddedOutput, pads, 0.f );
}

} // namespace NeoOnnx
