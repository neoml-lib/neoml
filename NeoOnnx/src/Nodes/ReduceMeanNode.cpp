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

#include "ReduceMeanNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CReduceMeanNode::CReduceMeanNode( const onnx::NodeProto& reduceMean, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( reduceMean, nodeOutputs ),
	keepDims( attributes.GetOptionalInt( "keepdims", 1 ) )
{
	CheckOnnxProtocol( input.Size() == 1, "node must have 1 input", reduceMean );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", reduceMean );

	attributes.GetRequiredIntArray( "axes", axes );
}

void CReduceMeanNode::OnnxReshape()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).GetType() == TT_DataTensor, "constant input", onnxNode );
	const CTensorShape& inputShape = InputTensor( 0 ).GetShape();
	CTensorShape outputShape;

	int axisIndex = 0;
	for( int i = 0; i < inputShape.Size(); ++i ) {
		if( axisIndex < axes.Size() && axes[axisIndex] == i ) {
			++axisIndex;
			if( keepDims != 0 ) {
				outputShape.Add( 1 );
			}
		} else {
			outputShape.Add( inputShape[i] );
		}
	}

	outputData.Add( CTensor( TT_DataTensor, outputShape ) );
}

void CReduceMeanNode::MarkTensorDims()
{
	const CTensorDim& inputDim = InputTensor( 0 ).GetTensorDim();
	CheckNeoOnnxInternal( inputDim.Size() == InputTensor( 0 ).GetShape().Size(),
		"input's dimensions must be marked", onnxNode );

	if( keepDims != 0 ) {
		CheckNeoOnnxInternal( outputData[0].SetTensorDim( inputDim ), "marking output dimensions failed", onnxNode );
		return;
	}

	CTensorDim outputDim;
	int axisIndex = 0;
	for( int i = 0; i < inputDim.Size(); ++i ) {
		if( axisIndex < axes.Size() && axes[axisIndex] == i ) {
			++axisIndex;
		} else {
			outputDim.Add( inputDim[i] );
		}
	}

	CheckNeoOnnxInternal( outputData[0].SetTensorDim( outputDim ), "marking output dimensions failed", onnxNode );
}

static const int pool2dDims = ( 1 << static_cast<int>( BD_Height ) ) | ( 1 << static_cast<int>( BD_Width ) );

void CReduceMeanNode::AddLayers( CDnn& dnn )
{
	int pooledDims = 0;
	CArray<int> axes;
	attributes.GetRequiredIntArray( "axes", axes );

	for( int axisIndex = 0; axisIndex < axes.Size(); ++axisIndex ) {
		pooledDims |= ( 1 << static_cast<int>( InputTensor( 0 ).GetTensorDim()[axes[axisIndex]] ) );
	}

	CheckNeoOnnxSupport( ( pooledDims | pool2dDims ) == pool2dDims,
		"reduce over dimensions other than BD_Height and BD_Width", onnxNode );

	add2dPoolingLayer( dnn, pooledDims );
}

void CReduceMeanNode::add2dPoolingLayer( CDnn& dnn, int pooledDims )
{
	CPtr<CMeanPoolingLayer> poolingLayer = new CMeanPoolingLayer( dnn.GetMathEngine() );
	poolingLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	// Making it global.
	for( int axisIndex = 0; axisIndex < axes.Size(); ++axisIndex ) {
		TBlobDim dim = InputTensor( 0 ).GetTensorDim()[axes[axisIndex]];
		const bool isDimPooled = ( ( ( 1 << static_cast<int>( dim ) ) & pooledDims ) != 0 );
		switch( dim ) {
			case BD_Height:
				poolingLayer->SetFilterHeight( isDimPooled ? InputTensor( 0 ).GetShape()[axes[axisIndex]] : 1 );
				poolingLayer->SetStrideHeight( 1 );
				break;
			case BD_Width:
				poolingLayer->SetFilterWidth( isDimPooled ? InputTensor( 0 ).GetShape()[axes[axisIndex]] : 1 );
				poolingLayer->SetStrideWidth( 1 );
				break;
			default:
				CheckNeoOnnxInternal( false, CString( "dimension " ) + Str( dim ) + " can not be pooled",
					onnxNode );
		}
	}

	poolingLayer->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	dnn.AddLayer( *poolingLayer );

	outputInfo.Add( COutputInfo( poolingLayer, 0 ) );
}

} // namespace NeoOnnx
