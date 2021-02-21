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

#include "SoftmaxNode.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CSoftmaxNode::CSoftmaxNode( const onnx::NodeProto& softmax, int opsetVersion ) :
	COpNode( softmax, opsetVersion ),
	axis( Attributes.GetOptionalInt( "axis", 1 ) )
{
	// The differences between versions are in negative axis support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", softmax );

	// Negative axis index supported since v11
	CheckOnnxProtocol( axis >= 0 || opsetVersion >= 11, "negative axis index", softmax );
	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", softmax );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", softmax );
}

void CSoftmaxNode::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	CheckNeoOnnxInternal( inputs[0] != nullptr && !inputs[0]->IsCalculated(), "Unknown input", OnnxNode );

	const int dimCount = inputs[0]->Shape().Size();
	CheckNeoOnnxSupport( axis <= 3, "more than 3 batch dimensions", OnnxNode );
	CheckNeoOnnxSupport( dimCount - axis + 1 <= 4, "more than 4 object  dimensions", OnnxNode );

	CDimOrder outputDimOrder;
	getDimOrder( dimCount, axis, inputs[0]->Layout().OnnxOrder, outputDimOrder );

	CPtr<const CUserTensor> input = dynamic_cast<const CUserTensor*>( ConvertTensor( *inputs[0], CTensorLayout( outputDimOrder ) ).Ptr() );

	CPtr<CSoftmaxLayer> softmax = new CSoftmaxLayer( dnn.GetMathEngine() );
	softmax->SetName( Name() );
	softmax->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *softmax );

	outputs[0] = new CUserTensor( input->Shape(), input->Layout(), CLayerOutput( softmax, 0 ) );
}

void CSoftmaxNode::getDimOrder( int dimCount, int axis, const CDimOrder& inputDimOrder, CDimOrder& dimOrder )
{
	if( inputDimOrder.IsEmpty() && axis == 3 ) {
		// DT_Onnx is ok if axis == 3
		return;
	}

	if( !inputDimOrder.IsEmpty() ) {
		// Check whether NeoML dim order is compatible with softmax
		bool isCompatible = true;
		for( int i = 0; i < inputDimOrder.Size(); ++i ) {
			if( ( i < axis && inputDimOrder[i] >= BD_Height ) // object dimension before axis
				|| ( i >= axis && inputDimOrder[i] < BD_Height ) ) // batch dimension after axis
			{
				isCompatible = false;
				break;
			}
		}

		if( isCompatible ) {
			inputDimOrder.CopyTo( dimOrder );
			return;
		}
	}

	dimOrder.SetBufferSize( dimCount );
	for( int i = 0; i < dimCount; ++i ) {
		if( i < axis ) {
			dimOrder[i] = static_cast<TBlobDim>( i );
		} else {
			dimOrder[i] = static_cast<TBlobDim>( BD_Height + i - axis );
		}
	}
}

} // namespace NeoOnnx
