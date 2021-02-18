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

#include "PoolNode.h"
#include "PadNode.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CPoolNodeBase::CPoolNodeBase( TPoolType _poolType, const onnx::NodeProto& poolNode, int opsetVersion ) :
	COpNode( poolNode, opsetVersion ),
	poolType( _poolType ),
	autoPad( Attributes.GetOptionalString( "auto_pad", "NOTSET" ) )
{
	// The difference between versions are in rarely used attributes (not supported by NeoOnnx): ceil_mode, storage_order etc)
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", poolNode );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", poolNode );
	CheckOnnxProtocol( OutputCount() == 1 || OutputCount() == 2, "node must have 1 or 2 outputs", poolNode );

	Attributes.GetRequiredIntArray( "kernel_shape", kernelShape );
	Attributes.GetOptionalIntArray( "strides", strides );
	Attributes.GetOptionalIntArray( "pads", pads );

	CheckNeoOnnxSupport( kernelShape.Size() == 2, "non 2-dimensional max pooling", poolNode );
}

void CPoolNodeBase::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	// Check input
	CheckNeoOnnxSupport( inputs[0] != nullptr && !inputs[0]->IsCalculated(),
		"", OnnxNode );
	const CTensorShape& inputShape = inputs[0]->Shape();
	CheckNeoOnnxSupport( inputShape.Size() > 2 && inputShape.Size() <= 4,
		"wrong input tensor's dimensions number", OnnxNode );
	const int poolDims = static_cast<int>( inputShape.Size() ) - 2;

	// Initialize strides, pads and dilations (if not given)
	if( strides.IsEmpty() ) {
		strides.Add( 1, poolDims );
	}
	if( pads.IsEmpty() ) {
		pads.Add( 0, 2 * poolDims );
	}

	if( autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER" ) {
		CalculatePadding( autoPad, kernelShape, pads );
	}

	if( poolType == PT_Mean ) {
		for( int padIndex = 0; padIndex < pads.Size(); ++padIndex ) {
			// We can't pad image correctly for average (result will differ from Onnx anyway)
			CheckNeoOnnxSupport( pads[padIndex] == 0, "average pooling with padding", OnnxNode );
		}
	}

	// Calculate output shape
	CTensorShape outputShape;
	inputShape.CopyTo( outputShape );
	for( int dimIndex = 0; dimIndex < poolDims; ++dimIndex ) {
		outputShape[dimIndex + 2] = ( inputShape[dimIndex + 2] + pads[dimIndex] + pads[dimIndex + poolDims]
			- kernelShape[dimIndex] ) / strides[dimIndex] + 1;
	}

	// TODO: add 3d-pooling support
	CPtr<CPoolingLayer> pooling;
	static_assert( PT_Count == 2, "PT_Count != 2" );
	switch( poolType ) {
		case PT_Max:
			pooling = new CMaxPoolingLayer( dnn.GetMathEngine() );
			break;
		case PT_Mean:
			pooling = new CMeanPoolingLayer( dnn.GetMathEngine() );
			break;
		default:
			CheckNeoOnnxInternal( false, "unknown pool type", OnnxNode );
	}
	pooling->SetName( Name() );

	CDimOrder expectedOrder( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width } );
	expectedOrder.SetSize( inputShape.Size() );
	CPtr<const CUserTensor> input = dynamic_cast<const CUserTensor*>( ConvertTensor( *inputs[0], CTensorLayout( expectedOrder ) ).Ptr() );
	input = PadUserTensor( *input, pads, poolType == PT_Max ? -FLT_MAX : 0.f );

	pooling->SetFilterHeight( kernelShape[0] );
	pooling->SetFilterWidth( kernelShape[1] );

	pooling->SetStrideHeight( strides[0] );
	pooling->SetStrideWidth( strides[1] );

	pooling->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *pooling );

	outputs[0] = new CUserTensor( outputShape, input->Layout(), CLayerOutput( pooling, 0 ) );
}

} // namespace NeoOnnx
