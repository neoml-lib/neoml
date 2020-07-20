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

CReduceMeanNode::CReduceMeanNode( int nodeIndex, const onnx::NodeProto& reduceMean, int opsetVersion ) :
	COpNode( nodeIndex, reduceMean, opsetVersion ),
	keepDims( attributes.GetOptionalInt( "keepdims", 1 ) )
{
	// The differences between versions are in negative indices support
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", reduceMean );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", reduceMean );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", reduceMean );

	attributes.GetRequiredIntArray( "axes", axes );
}

void CReduceMeanNode::CalcOutputTensors( CGraphTensors& tensors, IMathEngine& mathEngine )
{
	CheckNeoOnnxSupport( InputTensor( tensors, 0 ).Data == nullptr, "constant input", onnxNode );
	const CTensorShape& inputShape = InputTensor( tensors, 0 ).Shape;
	CTensorShape& outputShape = OutputTensor( tensors, 0 ).Shape;

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

	CheckNeoOnnxSupport( InputTensor( tensors, 0 ).Data == nullptr, "output pre-calculation", onnxNode );
	// The OutputTensor( tensors, 0 ).Data was already set to nullptr in default constructor.
}

void CReduceMeanNode::MarkTensorDims( const CGraphTensors& tensors, CGraphDims& dims )
{
	const CTensorDim& inputDim = InputDim( dims, 0 );
	CheckNeoOnnxInternal( inputDim.Size() == InputTensor( tensors, 0 ).Shape.Size(),
		"input's dimensions must be marked", onnxNode );

	if( keepDims != 0 ) {
		CheckNeoOnnxInternal( SetTensorDim( OutputTensor( tensors, 0 ).Shape, inputDim, OutputDim( dims, 0 ) ),
			"marking output dimensions failed", onnxNode );
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

	CheckNeoOnnxInternal( SetTensorDim( OutputTensor( tensors, 0 ).Shape, outputDim, OutputDim( dims, 0 ) ),
		"marking output dimensions failed", onnxNode );
}

static const int pool2dDims = ( 1 << static_cast<int>( BD_Height ) ) | ( 1 << static_cast<int>( BD_Width ) );

void CReduceMeanNode::AddLayers( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn )
{
	int pooledDims = 0;
	CArray<int> axes;
	attributes.GetRequiredIntArray( "axes", axes );

	for( int axisIndex = 0; axisIndex < axes.Size(); ++axisIndex ) {
		pooledDims |= ( 1 << static_cast<int>( ( InputDim( dims, 0 ) )[axes[axisIndex]] ) );
	}

	CheckNeoOnnxSupport( ( pooledDims | pool2dDims ) == pool2dDims,
		"reduce over dimensions other than BD_Height and BD_Width", onnxNode );

	add2dPoolingLayer( tensors, dims, mappings, dnn, pooledDims );
}

void CReduceMeanNode::add2dPoolingLayer( const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn, int pooledDims )
{
	CPtr<CMeanPoolingLayer> poolingLayer = new CMeanPoolingLayer( dnn.GetMathEngine() );
	poolingLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	// Making it global.
	for( int axisIndex = 0; axisIndex < axes.Size(); ++axisIndex ) {
		TBlobDim dim = ( InputDim( dims, 0 ) )[axes[axisIndex]];
		const bool isDimPooled = ( ( ( 1 << static_cast<int>( dim ) ) & pooledDims ) != 0 );
		switch( dim ) {
			case BD_Height:
				poolingLayer->SetFilterHeight( isDimPooled ? InputTensor( tensors, 0 ).Shape[axes[axisIndex]] : 1 );
				poolingLayer->SetStrideHeight( 1 );
				break;
			case BD_Width:
				poolingLayer->SetFilterWidth( isDimPooled ? InputTensor( tensors, 0 ).Shape[axes[axisIndex]] : 1 );
				poolingLayer->SetStrideWidth( 1 );
				break;
			default:
				CheckNeoOnnxInternal( false, CString( "dimension " ) + Str( dim ) + " can not be pooled",
					onnxNode );
		}
	}

	poolingLayer->Connect( 0, *InputMapping( mappings, 0 ).Layer, InputMapping( mappings, 0 ).OutputIndex );
	dnn.AddLayer( *poolingLayer );

	OutputMapping( mappings, 0 ) = CNeoMLMapping( poolingLayer, 0 );
}

} // namespace NeoOnnx
