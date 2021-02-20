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

#include "TransposeNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CTransposeNode::CTransposeNode( const onnx::NodeProto& transpose, int opsetVersion ) :
	COpNode( transpose, opsetVersion )
{
	// The differences between versions are in supported data types and legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", transpose );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", transpose );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", transpose );
}

// Gets actual dim order (explicitly, even when DT_Onnx)
static void getDimOrder( int dimCount, const CTensorLayout& layout, CDimOrder& order )
{
	if( layout.DimType == DT_NeoML ) {
		layout.OnnxOrder.CopyTo( order );
		return;
	}

	order.SetBufferSize( dimCount );
	for( int i = 0; i < dimCount; ++i ) {
		order.Add( static_cast<TBlobDim>( i ) );
	}
}

// Returns is current (explicit) dim order is DT_Onnx
static bool isOnnxDimOrder( const CDimOrder& order )
{
	for( int i = 0; i < order.Size(); ++i ) {
		if( order[i] != static_cast<TBlobDim>( i ) ) {
			return false;
		}
	}
	return true;
}

void CTransposeNode::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	CheckNeoOnnxInternal( inputs[0] != nullptr && !inputs[0]->IsCalculated(), "unknown input", OnnxNode );

	const CTensorShape& inputShape = inputs[0]->Shape();
	const int dimCount = inputShape.Size();

	CFastArray<int, 8> perm;
	// Default value is reverse order
	perm.SetBufferSize( dimCount );
	for( int i = 0; i < dimCount; ++i ) {
		perm.Add( dimCount - 1 - i );
	}
	Attributes.GetOptionalIntArray( "perm", perm );

	// Working only with layout (converters will be added by next layers when needed)
	CDimOrder inputDimOrder;
	getDimOrder( dimCount, inputs[0]->Layout(), inputDimOrder );

	CDimOrder outputDimOrder;
	outputDimOrder.SetBufferSize( dimCount );
	CTensorShape outputShape;
	outputShape.SetBufferSize( dimCount );

	for( int i = 0; i < dimCount; ++i ) {
		outputDimOrder.Add( inputDimOrder[perm[i]] );
		outputShape.Add( inputShape[perm[i]] );
	}

	outputs[0] = new CUserTensor( outputShape, 
		isOnnxDimOrder( outputDimOrder ) ? CTensorLayout() : CTensorLayout( outputDimOrder ),
		dynamic_cast<const CUserTensor*>( inputs[0].Ptr() )->LayerOutput() );
}

} // namespace NeoOnnx
