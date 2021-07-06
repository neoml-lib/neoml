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

#include "onnx.pb.h"

#include "PoolOperator.h"
#include "TensorUtils.h"

namespace NeoOnnx {

// Creates pooling layer for the given pooling type
static CPtr<CPoolingLayer> createPoolingLayer( CPoolOperatorBase::TPoolType poolType, IMathEngine& mathEngine )
{
	static_assert( CPoolOperatorBase::PT_Count == 2, "CPoolOperatorBase::PT_Count != 2" );
	switch( poolType ) {
		case CPoolOperatorBase::PT_Max:
			return new CMaxPoolingLayer( mathEngine );
		case CPoolOperatorBase::PT_Mean:
			return new CMeanPoolingLayer( mathEngine );
		default:
			NeoAssert( false );
	}

	return nullptr;
}

CPoolOperatorBase::CPoolOperatorBase( TPoolType _poolType, const onnx::NodeProto& poolNode, int opsetVersion ) :
	CLayerOperator( poolNode, opsetVersion ),
	poolType( _poolType ),
	autoPad( "NOTSET" ),
	includePad( false )
{
	// The difference between versions are in rarely used attributes (not supported by NeoOnnx): ceil_mode, storage_order etc)
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1 || OutputCount() == 2, "operator must have 1 or 2 outputs", *this );

	if( poolType == PT_Mean && OpsetVersion >= 7 ) {
		int countIncludePad = 0;
		GetAttribute( "count_include_pad", countIncludePad );
		includePad = countIncludePad != 0;
	}

	CheckOnnxProtocol( GetAttribute( "kernel_shape", kernelShape ), "'kernel_shape' attribute is missing", *this );
	CheckNeoOnnxSupport( kernelShape.Size() == 2, "non 2-dimensional max pooling", *this );

	GetAttribute( "auto_pad", autoPad );
}

void CPoolOperatorBase::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	// Check input
	CheckNeoOnnxSupport( inputs[0] != nullptr && !inputs[0]->IsCalculated(),
		"", *this );
	const CTensorShape& inputShape = inputs[0]->Shape();
	CheckNeoOnnxSupport( inputShape.Size() > 2 && inputShape.Size() <= 4,
		"wrong input tensor's dimensions number", *this );
	const int poolDims = inputShape.Size() - 2;

	// Initialize strides and pads(if not given)
	CFastArray<int, 8> strides;
	getStrides( inputs, strides );
	CFastArray<int, 8> pads;
	getPads( inputs, pads );

	if( poolType == PT_Mean && !includePad ) {
		// We can't pad image correctly because NeoML's doesn't support paddings in poolings
		// and explicit paddings will be included in calculations
		for( int padIndex = 0; padIndex < pads.Size(); ++padIndex ) {
			CheckNeoOnnxSupport( pads[padIndex] == 0, "average pooling with padding not included in calc", *this );
		}
	}

	// Calculate output shape
	CTensorShape outputShape;
	inputShape.CopyTo( outputShape );
	for( int dimIndex = 0; dimIndex < poolDims; ++dimIndex ) {
		outputShape[dimIndex + 2] = ( inputShape[dimIndex + 2] + pads[dimIndex] + pads[dimIndex + poolDims]
			- kernelShape[dimIndex] ) / strides[dimIndex] + 1;
	}

	CPtr<CPoolingLayer> pooling = createPoolingLayer( poolType, dnn.GetMathEngine() );
	pooling->SetName( Name() );

	CTensorLayout expectedLayout( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width } );
	expectedLayout.SetSize( inputShape.Size() );
	CPtr<const CUserTensor> input = dynamic_cast<const CUserTensor*>( ConvertTensor( *inputs[0], expectedLayout ).Ptr() );
	input = PadUserTensor( *input, pads, poolType == PT_Max ? -FLT_MAX : 0.f );

	pooling->SetFilterHeight( kernelShape[0] );
	pooling->SetFilterWidth( kernelShape[1] );

	pooling->SetStrideHeight( strides[0] );
	pooling->SetStrideWidth( strides[1] );

	pooling->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *pooling );

	outputs.Add( new CUserTensor( outputShape, input->Layout(), CLayerOutput( pooling, 0 ) ) );
}

// Gets pool strides
void CPoolOperatorBase::getStrides( const CTensorArray& inputs, CFastArray<int, 8>& strides ) const
{
	GetAttribute( "strides", strides );

	if( strides.IsEmpty() ) {
		strides.Add( 1, inputs[0]->Shape().Size() - 2 );
	}
}

// Gets pad sizes
void CPoolOperatorBase::getPads( const CTensorArray& inputs, CFastArray<int, 8>& pads ) const
{
	GetAttribute( "pads", pads );

	if( pads.IsEmpty() ) {
		pads.Add( 0, 2 * ( inputs[0]->Shape().Size() - 2 ) );
	}

	if( autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER" ) {
		CalculatePadding( autoPad, kernelShape, pads );
	}
}

} // namespace NeoOnnx

