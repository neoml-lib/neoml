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
	keepDims( Attributes.GetOptionalInt( "keepdims", 1 ) )
{
	// The differences between versions are in negative indices support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", reduceMean );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", reduceMean );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", reduceMean );

	Attributes.GetRequiredIntArray( "axes", axes );
}

void CReduceMeanNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "constant input", OnnxNode );
	const CTensorShape& inputShape = tensors[Input[0]].Shape;
	CTensorShape& outputShape = tensors[Output[0]].Shape;

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

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
	// The tensors[Output[0]].Data was already set to nullptr in default constructor.
}

void CReduceMeanNode::MarkTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	const CTensorDim& inputDim = dims[Input[0]];
	CheckNeoOnnxInternal( inputDim.Size() == tensors[Input[0]].Shape.Size(),
		"input's dimensions must be marked", OnnxNode );

	if( keepDims != 0 ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, inputDim, dims[Output[0]] ),
			"marking output dimensions failed", OnnxNode );
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

	CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, outputDim, dims[Output[0]] ),
		"marking output dimensions failed", OnnxNode );
}

static const int pool2dDims = ( 1 << static_cast<int>( BD_Height ) ) | ( 1 << static_cast<int>( BD_Width ) );

void CReduceMeanNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims, CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	int pooledDims = 0;
	CArray<int> axes;
	Attributes.GetRequiredIntArray( "axes", axes );

	for( int axisIndex = 0; axisIndex < axes.Size(); ++axisIndex ) {
		pooledDims |= ( 1 << static_cast<int>( ( dims[Input[0]] )[axes[axisIndex]] ) );
	}

	CheckNeoOnnxSupport( ( pooledDims | pool2dDims ) == pool2dDims,
		"reduce over dimensions other than BD_Height and BD_Width", OnnxNode );

	add2dPoolingLayer( tensors, dims, neoMLLinks, dnn, pooledDims );
}

void CReduceMeanNode::add2dPoolingLayer( const CTensorCache& tensors, const CDimCache& dims, CNeoMLLinkCache& neoMLLinks, CDnn& dnn, int pooledDims )
{
	CPtr<CMeanPoolingLayer> poolingLayer = new CMeanPoolingLayer( dnn.GetMathEngine() );
	poolingLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	// Making it global.
	for( int axisIndex = 0; axisIndex < axes.Size(); ++axisIndex ) {
		TBlobDim dim = ( dims[Input[0]] )[axes[axisIndex]];
		const bool isDimPooled = ( ( ( 1 << static_cast<int>( dim ) ) & pooledDims ) != 0 );
		switch( dim ) {
			case BD_Height:
				poolingLayer->SetFilterHeight( isDimPooled ? tensors[Input[0]].Shape[axes[axisIndex]] : 1 );
				poolingLayer->SetStrideHeight( 1 );
				break;
			case BD_Width:
				poolingLayer->SetFilterWidth( isDimPooled ? tensors[Input[0]].Shape[axes[axisIndex]] : 1 );
				poolingLayer->SetStrideWidth( 1 );
				break;
			default:
				CheckNeoOnnxInternal( false, CString( "dimension " ) + Str( dim ) + " can not be pooled",
					OnnxNode );
		}
	}

	poolingLayer->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *poolingLayer );

	neoMLLinks[Output[0]] = CNeoMLLink( poolingLayer, 0 );
}

} // namespace NeoOnnx
