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

#include "common.h"
#pragma hdrstop

#include "GlobalAveragePoolNode.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGlobalAveragePoolNode::CGlobalAveragePoolNode( int nodeIndex, const onnx::NodeProto& globalAveragePool, int opsetVersion ) :
	COpNode( nodeIndex, globalAveragePool, opsetVersion )
{
	// This operator doesn't have multiple versions
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", globalAveragePool );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", globalAveragePool );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", globalAveragePool );
}

void CGlobalAveragePoolNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	const CTensorShape& inputShape = tensors[Input[0]].Shape;
	CheckOnnxProtocol( inputShape.Size() >= 2, "node's input must have at least 2 dimensions", OnnxNode );
	tensors[Output[0]].Shape.Add( 1, inputShape.Size() );
	tensors[Output[0]].Shape[0] = inputShape[0];
	tensors[Output[0]].Shape[1] = inputShape[1];

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
	// The tensors[Output[0]].Data was already set to nullptr in default constructor.
}

void CGlobalAveragePoolNode::MarkTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( !dims[Input[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, dims[Input[0]], dims[Output[0]] ),
			"marking output dimensions failed", OnnxNode );
	}

	if( !dims[Output[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, dims[Output[0]], dims[Input[0]] ),
			"marking input dimensions failed", OnnxNode );
	}
}

static const int pool2dDims = ( 1 << static_cast<int>( BD_Height ) ) | ( 1 << static_cast<int>( BD_Width ) );

void CGlobalAveragePoolNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	int pooledDims = 0;
	const CTensorDim& inputDim = dims[Input[0]];

	for( int dimIndex = 2; dimIndex < inputDim.Size(); ++dimIndex ) {
		pooledDims |= ( 1 << static_cast<int>( inputDim[dimIndex] ) );
	}

	CheckNeoOnnxSupport( ( pooledDims | pool2dDims ) == pool2dDims,
		"reduce over dimensions other than BD_Height and BD_Width", OnnxNode );

	add2dPoolingLayer( tensors, dims, neoMLLinks, dnn, pooledDims );
}

// Adds 2-dimensional mean pooling layer to dnn
void CGlobalAveragePoolNode::add2dPoolingLayer( const CTensorCache& tensors, const CDimCache& dims, CNeoMLLinkCache& neoMLLinks,
	CDnn& dnn, int pooledDims )
{
	CPtr<CMeanPoolingLayer> poolingLayer = new CMeanPoolingLayer( dnn.GetMathEngine() );
	poolingLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );
	const CTensorDim& inputDim = dims[Input[0]];

	// Making it global.
	for( int dimIndex = 2; dimIndex < inputDim.Size(); ++dimIndex ) {
		TBlobDim dim = inputDim[dimIndex];
		const bool isDimPooled = ( ( ( 1 << static_cast<int>( dim ) ) & pooledDims ) != 0 );
		switch( dim ) {
			case BD_Height:
				poolingLayer->SetFilterHeight( isDimPooled ? tensors[Input[0]].Shape[dimIndex] : 1 );
				poolingLayer->SetStrideHeight( 1 );
				break;
			case BD_Width:
				poolingLayer->SetFilterWidth( isDimPooled ? tensors[Input[0]].Shape[dimIndex] : 1 );
				poolingLayer->SetStrideWidth( 1 );
				break;
			default:
				CheckNeoOnnxInternal( false, "dimension " + Str( dim ) + " can not be pooled",
					OnnxNode );
		}
	}

	poolingLayer->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *poolingLayer );

	neoMLLinks[Output[0]] = CNeoMLLink( poolingLayer, 0 );
}

} // namespace NeoOnnx
