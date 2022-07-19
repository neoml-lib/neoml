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

	CTensorShape kernelShape;
	getConvTransposeKernelShape( inputs, kernelShape );
	CFastArray<int, 8> strides;
	getStrides( inputs, strides );
	CFastArray<int, 8> dilations;
	getDilations( inputs, dilations );
	CTensorShape outputShape;
	calcOutputShape( inputs, kernelShape, strides, dilations, outputShape );
	// total padding is a combined result of applying output_padding and pads attributes
	CFastArray<int, 8> totalPadding;
	getTotalPadding( outputShape, totalPadding );
	const int convDims = outputShape.Size() - 2;
	// check if total padding can be fused into CTransposedConvLayer
	bool doesConvLayerSupportPadding = true;
	for( int i = 0; i < convDims; ++i ) {
		if( totalPadding[i] < 0 || totalPadding[i] != totalPadding[i + convDims] ) {
			doesConvLayerSupportPadding = false;
			break;
		}
	}

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
	// apply padding via layer (if possible)
	if( doesConvLayerSupportPadding ) {
		transposedConv->SetPaddingHeight( totalPadding[0] );
		transposedConv->SetPaddingWidth( totalPadding[1] );
		outputShape[2] -= 2 * totalPadding[0];
		outputShape[3] -= 2 * totalPadding[1];
	}

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
	// apply padding if it hasn't been applied via layer
	if( !doesConvLayerSupportPadding ) {
		// padding in transposed conv works in the opposite direction
		for( int i = 0; i < 2 * convDims; ++i ) {
			totalPadding[i] = -totalPadding[i];
		}
		currTensor = PadUserTensor( *currTensor, totalPadding, 0 ).Ptr();
	}
	outputs.Add( currTensor.Ptr() );
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

// Gets output_padding values
void CConvTransposeOperator::getOutputPadding( const CTensorShape& convShape, CFastArray<int, 8>& outputPadding ) const
{
	GetAttribute( "output_padding", outputPadding );
	if( outputPadding.IsEmpty() ) {
		const int convDims = convShape.Size() - 2;
		outputPadding.Add( 0, convDims );
	}
}

// Gets pads values
void CConvTransposeOperator::getPads( const CTensorShape& convShape, const CFastArray<int, 8>& outputPadding,
	CFastArray<int, 8>& pads ) const
{
	const int convDims = convShape.Size() - 2;
	CFastArray<int, 8> outputShape;

	// Getting pads sizes (one way, or another)
	const bool hasPadsAttribute = GetAttribute( "pads", pads );
	if( !hasPadsAttribute ) {
		if( !GetAttribute( "output_shape", outputShape ) ) {
			pads.Add( 0, 2 * convDims );
			return;
		}

		// Pads must be automatically calculated based on the output_shape
		NeoAssert( outputShape.Size() == convShape.Size() );
		CString autoPad = "NOTSET";
		GetAttribute( "auto_pad", autoPad );

		pads.SetSize( 2 * convDims );
		for( int i = 0; i < convDims; ++i ) {
			const int totalPadding = convShape[i + 2] + outputPadding[i] - outputShape[i + 2];
			pads[i] = autoPad != "SAME_UPPER" ? totalPadding / 2 : ( totalPadding + 1 ) / 2;
			pads[convDims + i] = totalPadding - pads[i];
		}
	}
}

// Calculates total paddings
void CConvTransposeOperator::getTotalPadding( const CTensorShape& convShape, CFastArray<int, 8>& totalPadding ) const
{
	CFastArray<int, 8> outputPadding;
	getOutputPadding( convShape, outputPadding );
	CFastArray<int, 8> pads;
	getPads( convShape, outputPadding, pads );

	const int convDims = convShape.Size() - 2;
	NeoAssert( outputPadding.Size() == convDims );
	NeoAssert( pads.Size() == 2 * convDims );
	totalPadding.SetSize( 2 * convDims );
	for( int i = 0; i < convDims; ++i ) {
		totalPadding[i] = pads[i];
		totalPadding[i + convDims] = pads[i + convDims] - outputPadding[i];
	}
}

} // namespace NeoOnnx
