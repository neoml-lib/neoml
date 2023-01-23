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

#include <NeoML/Dnn/Layers/Onnx/OnnxConvTransposeLayer.h>

namespace NeoOnnx {

// Gets kernel shape
static void getConvTransposeKernelShape( int dimCount, const CDataTensor& filter, CTensorShape& kernelShape )
{
	kernelShape.SetBufferSize( dimCount - 2 );
	for( int dimIndex = 2; dimIndex < dimCount; ++dimIndex ) {
		kernelShape.Add( filter.DimSize( dimIndex ) );
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
	CheckNoShapeInputs( inputs );

	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	CheckNeoOnnxSupport( inputs[0]->DimCount() == 4, "non-2d convTranspose", *this );
	CheckOnnxProtocol( inputs[1] != nullptr, "input can't be optional", *this );
	CheckNeoOnnxSupport( inputs[1]->Type() == TTensorType::Data, "user-provided weights", *this );
	if( InputCount() == 3 && inputs[2] != nullptr ) {
		CheckNeoOnnxSupport( inputs[2]->Type() == TTensorType::Data, "user-provided bias", *this );
	}

	CTensorLayout neoMLLayout( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width } );
	CPtr<const CDataTensor> filter = dynamic_cast<const CDataTensor*>( ConvertTensor( *inputs[1], neoMLLayout ).Ptr() );
	const int filterCount = filter->DimSize( 1 );

	CTensorShape kernelShape;
	getConvTransposeKernelShape( inputs[0]->DimCount(), *filter, kernelShape );
	CFastArray<int, 8> strides;
	getStrides( inputs, strides );
	CFastArray<int, 8> dilations;
	getDilations( inputs, dilations );
	const int convDims = inputs[0]->DimCount() - 2;

	IMathEngine& mathEngine = dnn.GetMathEngine();
	CPtr<COnnxConvTransposeLayer> transposedConv = new COnnxConvTransposeLayer( mathEngine );
	transposedConv->SetName( Name() );
	transposedConv->SetFilterCount( filterCount );
	transposedConv->SetFilterHeight( kernelShape[0] );
	transposedConv->SetFilterWidth( kernelShape[1] );
	transposedConv->SetStrideHeight( strides[0] );
	transposedConv->SetStrideWidth( strides[1] );
	transposedConv->SetDilationHeight( dilations[0] );
	transposedConv->SetDilationWidth( dilations[1] );

	CFastArray<int, 8> pads;
	GetAttribute( "pads", pads );
	pads.CopyTo( transposedConv->Pads() );

	CFastArray<int, 8> outputPadding;
	if( !GetAttribute( "output_padding", outputPadding ) ) {
		outputPadding.Add( 0, convDims );
	}
	outputPadding.CopyTo( transposedConv->OutputPadding() );

	CFastArray<int, 8> outputShape;
	GetAttribute( "output_shape", outputShape );
	outputShape.CopyTo( transposedConv->OutputShape() );

	transposedConv->SetFilterData( filter->Data()->GetCopy() );
	if( InputCount() == 3 && inputs[2] != nullptr ) {
		transposedConv->SetFreeTermData( dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data()->GetCopy() );
	} else {
		transposedConv->SetZeroFreeTerm( true );
	}

	CPtr<const CUserTensor> currTensor = AsUserTensor( *ConvertTensor( *inputs[0], neoMLLayout ), Name() + "_Source", dnn );
	transposedConv->Connect( 0, *currTensor->Layer(), currTensor->OutputIndex() );
	dnn.AddLayer( *transposedConv );
	currTensor = new CUserTensor( neoMLLayout, CLayerOutput( transposedConv, 0 ) );
	outputs.Add( currTensor.Ptr() );
}

// Gets strides
void CConvTransposeOperator::getStrides( const CTensorArray& inputs, CFastArray<int, 8>& strides ) const
{
	GetAttribute( "strides", strides );
	if( strides.IsEmpty() ) {
		const int convDims = static_cast<int>( inputs[0]->DimCount() ) - 2;
		strides.Add( 1, convDims );
	}
}

// Gets dilation sizes
void CConvTransposeOperator::getDilations( const CTensorArray& inputs, CFastArray<int, 8>& dilations ) const
{
	GetAttribute( "dilations", dilations );
	if( dilations.IsEmpty() ) {
		const int convDims = inputs[0]->DimCount() - 2;
		dilations.Add( 1, convDims );
	}
}

} // namespace NeoOnnx
