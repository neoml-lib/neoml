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

CGlobalAveragePoolNode::CGlobalAveragePoolNode( const onnx::NodeProto& globalAveragePool, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( globalAveragePool, nodeOutputs )
{
	CheckOnnxProtocol( input.Size() == 1, "node must have 1 input", globalAveragePool );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", globalAveragePool );
}

void CGlobalAveragePoolNode::OnnxReshape()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).GetType() == TT_DataTensor, "constant input", onnxNode );
	const CTensorShape& inputShape = InputTensor( 0 ).GetShape();
	CheckOnnxProtocol( inputShape.Size() >= 2, "node's input must have at least 2 dimensions", onnxNode );

	CTensorShape outputShape( { inputShape[0], inputShape[1] } );
	outputData.Add( CTensor( TT_DataTensor, outputShape ) );
}

void CGlobalAveragePoolNode::MarkTensorDims()
{
	const CTensorDim& inputDim = InputTensor( 0 ).GetTensorDim();
	CheckNeoOnnxInternal( inputDim.Size() == InputTensor( 0 ).GetShape().Size(),
		"input's dimensions must be marked", onnxNode );

	CheckNeoOnnxInternal( outputData[0].SetTensorDim( { inputDim[0], inputDim[1] } ),
		"marking output dimensions failed", onnxNode );
}

static const int pool2dDims = ( 1 << static_cast<int>( BD_Height ) ) | ( 1 << static_cast<int>( BD_Width ) );

void CGlobalAveragePoolNode::AddLayers( CDnn& dnn )
{
	int pooledDims = 0;
	const CTensorDim& inputDim = InputTensor( 0 ).GetTensorDim();

	for( int dimIndex = 2; dimIndex < inputDim.Size(); ++dimIndex ) {
		pooledDims |= ( 1 << static_cast<int>( inputDim[dimIndex] ) );
	}

	CheckNeoOnnxSupport( ( pooledDims | pool2dDims ) == pool2dDims,
		"reduce over dimensions other than BD_Height and BD_Width", onnxNode );

	add2dPoolingLayer( dnn, pooledDims );
}

void CGlobalAveragePoolNode::add2dPoolingLayer( CDnn& dnn, int pooledDims )
{
	CPtr<CMeanPoolingLayer> poolingLayer = new CMeanPoolingLayer( dnn.GetMathEngine() );
	poolingLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );
	const CTensorDim& inputDim = InputTensor( 0 ).GetTensorDim();

	// Making it global.
	for( int dimIndex = 2; dimIndex < inputDim.Size(); ++dimIndex ) {
		TBlobDim dim = inputDim[dimIndex];
		const bool isDimPooled = ( ( ( 1 << static_cast<int>( dim ) ) & pooledDims ) != 0 );
		switch( dim ) {
			case BD_Height:
				poolingLayer->SetFilterHeight( isDimPooled ? InputTensor( 0 ).GetShape()[dimIndex] : 1 );
				poolingLayer->SetStrideHeight( 1 );
				break;
			case BD_Width:
				poolingLayer->SetFilterWidth( isDimPooled ? InputTensor( 0 ).GetShape()[dimIndex] : 1 );
				poolingLayer->SetStrideWidth( 1 );
				break;
			default:
				CheckNeoOnnxInternal( false, "dimension " + Str( dim ) + " cannot be pooled",
					onnxNode );
		}
	}

	poolingLayer->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	dnn.AddLayer( *poolingLayer );

	outputInfo.Add( COutputInfo( poolingLayer, 0 ) );
}

} // namespace NeoOnnx
