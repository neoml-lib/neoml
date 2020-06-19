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

#include "UnsqueezeNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CUnsqueezeNode::CUnsqueezeNode( const onnx::NodeProto& unsqueeze, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( unsqueeze, nodeOutputs )
{
	CheckOnnxProtocol( input.Size() == 1, "node must have 1 input", unsqueeze );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", unsqueeze );

	attributes.GetRequiredIntArray( "axes", axes );
}

void CUnsqueezeNode::OnnxReshape()
{
	const CTensorShape& inputShape = InputTensor( 0 ).GetShape();

	CTensorShape outputShape;
	outputShape.SetSize( inputShape.Size() + axes.Size() );
	int axisIndex = 0;
	for( int i = 0; i < outputShape.Size(); ++i ) {
		if( axisIndex < axes.Size() && i == axes[axisIndex] ) {
			outputShape[i] = 1;
			axisIndex++;
		} else {
			outputShape[i] = inputShape[i - axisIndex];
		}
	}

	CDnnBlob* outputBlob = InputTensor( 0 ).GetType() == TT_ConstantTensor ? InputTensor( 0 ).GetData() : nullptr;
	outputData.Add( CTensor( InputTensor( 0 ).GetType(), outputShape, outputBlob ) );
}

void CUnsqueezeNode::MarkTensorDims()
{
	if( outputData[0].GetType() == TT_ConstantTensor ) {
		return;
	}

	const CTensorDim& outputDim = outputData[0].GetTensorDim();

	if( outputDim.IsEmpty() ) {
		return;
	}

	CTensorDim inputDim;
	int axisIndex = 0;
	for( int i = 0; i < outputDim.Size(); ++i ) {
		if( axisIndex < axes.Size() && i == axes[axisIndex] ) {
			++axisIndex;
		} else {
			inputDim.Add( outputDim[i] );
		}
	}

	CheckNeoOnnxInternal( InputTensor( 0 ).SetTensorDim( inputDim ),
		"marking input dimensions failed", onnxNode );
}

void CUnsqueezeNode::AddLayers( CDnn& )
{
	if( outputData[0].GetType() == TT_ConstantTensor ) {
		return;
	}

	outputInfo.Add( InputInfo( 0 ) );
}

} // namespace NeoOnnx
