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

CSqueezeNode::CSqueezeNode( const onnx::NodeProto& squeeze, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( squeeze, nodeOutputs )
{
	CheckOnnxProtocol( input.Size() == 1, "node must have 1 input", squeeze );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", squeeze );

	attributes.GetRequiredIntArray( "axes", axes );
}

void CSqueezeNode::OnnxReshape()
{
	const CTensorShape& inputShape = InputTensor( 0 ).GetShape();

	CTensorShape outputShape;
	outputShape.SetBufferSize( inputShape.Size() - axes.Size() );
	int axisIndex = 0;
	for( int i = 0; i < inputShape.Size(); ++i ) {
		if( axisIndex < axes.Size() && i == axes[axisIndex] ) {
			CheckOnnxProtocol( inputShape[i] == 1, "squeezed dimensions must be of length 1", onnxNode );
			++axisIndex;
		} else {
			outputShape.Add( inputShape[i] );
		}
	}

	CDnnBlob* outputBlob = InputTensor( 0 ).GetType() == TT_ConstantTensor ? InputTensor( 0 ).GetData() : nullptr;
	outputData.Add( CTensor( InputTensor( 0 ).GetType(), outputShape, outputBlob ) );
}

void CSqueezeNode::MarkTensorDims()
{
	if( outputData[0].GetType() == TT_ConstantTensor ) {
		return;
	}

	const CTensorDim& inputDim = InputTensor( 0 ).GetTensorDim();

	CTensorDim outputDim;
	int axisIndex = 0;
	for( int i = 0; i < inputDim.Size(); ++i ) {
		if( axisIndex < axes.Size() && i == axes[axisIndex] ) {
			++axisIndex;
		} else {
			outputDim.Add( inputDim[i] );
		}
	}

	CheckNeoOnnxInternal( outputData[0].SetTensorDim( outputDim ), "marking output dimensions failed", onnxNode );
}

void CSqueezeNode::AddLayers( CDnn& )
{
	if( outputData[0].GetType() == TT_ConstantTensor ) {
		return;
	}

	outputInfo.Add( InputInfo( 0 ) );
}

} // namespace NeoOnnx
