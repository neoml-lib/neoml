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

#include "SqueezeNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CSqueezeNode::CSqueezeNode( const onnx::NodeProto& squeeze, int opsetVersion ) :
	COpNode( squeeze, opsetVersion )
{
	// v1 - original
	// v11 - added negative axes index support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", squeeze );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", squeeze );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", squeeze );
}

void CSqueezeNode::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	CheckNeoOnnxInternal( inputs[0] != nullptr && !inputs[0]->IsCalculated(), "user-provided input is expected", OnnxNode );

	CFastArray<int, 8> axes;
	getAxes( inputs[0]->Shape(), axes );

	CTensorShape outputShape;
	calcOutputShape( inputs[0]->Shape(), axes, outputShape );

	CDimOrder outputDimOrder;
	calcOutputDimOrder( inputs[0]->Layout().OnnxOrder, axes, outputDimOrder );

	outputs[0] = new CUserTensor( outputShape, CTensorLayout( outputDimOrder ),
		dynamic_cast<const CUserTensor*>( inputs[0].Ptr() )->LayerOutput() );
}

// Fills array with axes indices to be squeezed
// Returns array of positive indices in sorted order
void CSqueezeNode::getAxes( const CTensorShape& inputShape, CFastArray<int, 8>& axes ) const
{
	axes.Empty();
	Attributes.GetRequiredIntArray( "axes", axes );
	for( int i = 0; i < axes.Size(); ++i ) {
		if( axes[i] < 0 ) {
			CheckOnnxProtocol( OpsetVersion >= 11, "negative axes indices are supported since v11", OnnxNode );
			axes[i] += inputShape.Size();
		}
	}
	axes.QuickSort<Ascending<int>>();
}

// Calculates output tensor's shape
void CSqueezeNode::calcOutputShape( const CTensorShape& inputShape, const CFastArray<int, 8>& axes, CTensorShape& outputShape ) const
{
	outputShape.Empty();
	outputShape.SetBufferSize( inputShape.Size() - axes.Size() );

	int axeIndex = 0;
	int inputDimIndex = 0;
	for( int i = 0; i < inputShape.Size(); ++i ) {
		if( axeIndex < axes.Size() && i == axes[axeIndex] ) {
			++axeIndex;
		} else {
			outputShape.Add( inputShape[i] );
		}
	}
}

// Calculates output tensor's dim order
void CSqueezeNode::calcOutputDimOrder( const CDimOrder& inputDimOrder, const CFastArray<int, 8>& axes, CDimOrder& outputDimOrder ) const
{
	if( inputDimOrder.IsEmpty() ) {
		// Original layout is Onnx
		// No need in conversion
		outputDimOrder.Empty();
		return;
	}

	// NeoML layout
	int axeIndex = 0;
	int inputDimIndex = 0;
	outputDimOrder.SetBufferSize( axes.Size() + inputDimOrder.Size() );

	// Distribute unused blob dimensions among new axes
	for( int i = 0; i < inputDimOrder.Size(); ++i ) {
		if( axeIndex < axes.Size() && i == axes[axeIndex] ) {
			++axeIndex;
		} else {
			outputDimOrder.Add( inputDimOrder[i] );
		}
	}
}

} // namespace NeoOnnx
