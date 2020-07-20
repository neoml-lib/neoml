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

#include "GemmNode.h"
#include "FlattenNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGemmNode::CGemmNode( int nodeIndex, const onnx::NodeProto& gemm, int opsetVersion ) :
	COpNode( nodeIndex, gemm, opsetVersion ),
	alpha( attributes.GetOptionalFloat( "alpha", 1.f ) ),
	beta( attributes.GetOptionalFloat( "beta", 1.f ) ),
	transA( attributes.GetOptionalInt( "transA", 0 ) ),
	transB( attributes.GetOptionalInt( "transB", 0 ) )
{
	// Older versions have broadcast support
	CheckNeoOnnxSupport( opsetVersion >= 7 && opsetVersion <= MaxOpsetVersion, "opset version", gemm );

	CheckOnnxProtocol( InputCount() == 2 || InputCount() == 3, "node must have 2 or 3 inputs", gemm );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", gemm );

	CheckNeoOnnxSupport( alpha == 1.0f, "alpha != 1", gemm ); // TODO: add "alpha != 1.0" support.
	CheckNeoOnnxSupport( beta == 1.0f, "beta != 1", gemm ); // TODO: add "beta != 1.0" support.
	CheckNeoOnnxSupport( transA == 0, "transA != 0", gemm ); // TODO: add "TransA != 0" support.
	CheckNeoOnnxSupport( transB != 0, "transB == 0", gemm ); // TODO: add "TransB != 0" support.
}

void CGemmNode::CalcOutputTensors( CGraphTensors& tensors, IMathEngine& mathEngine )
{
	CheckNeoOnnxSupport( InputTensor( tensors, 0 ).Data == nullptr, "constant input", onnxNode );
	const CTensorShape& inputShape = InputTensor( tensors, 0 ).Shape;
	CheckOnnxProtocol( inputShape.Size() == 2, "input must be 2-dimensional", onnxNode );
	const int batchSize = inputShape[transA == 0 ? 0 : 1];
	const int inputObjectSize = inputShape[transA == 0 ? 1 : 0];

	CheckNeoOnnxSupport( InputTensor( tensors, 1 ).Data != nullptr, "non-constant weights", onnxNode );
	const CTensorShape& matrixShape = InputTensor( tensors, 1 ).Shape;
	CheckOnnxProtocol( matrixShape.Size() == 2, "weights must be 2-dimensional", onnxNode );
	CheckOnnxProtocol( matrixShape[transB == 0 ? 0 : 1] == inputObjectSize, "wrong weight size", onnxNode );
	const int numberOfElements = matrixShape[transB == 0 ? 1 : 0];

	if( InputCount() == 3 ) {
		CheckNeoOnnxSupport( InputTensor( tensors, 2 ).Data != nullptr, "non-constant bias", onnxNode );
		const CTensorShape& biasShape = InputTensor( tensors, 2 ).Shape;
		CheckOnnxProtocol( biasShape.Size() == 1, "bias must be 1-dimensional", onnxNode );
		CheckOnnxProtocol( biasShape[0] == numberOfElements, "wrong bias size", onnxNode );
	}

	OutputTensor( tensors, 0 ).Shape = { batchSize, numberOfElements };

	CheckNeoOnnxSupport( InputTensor( tensors, 0 ).Data == nullptr, "output pre-calculation", onnxNode );
	// The OutputTensor( tensors, 0 ).Data was already set to nullptr in default constructor.
}

void CGemmNode::MarkTensorDims( const CGraphTensors& tensors, CGraphDims& dims )
{
	// Gemm operator in onnx always works with 2-dimensional tensors.
	CheckNeoOnnxInternal( SetTensorDim( OutputTensor( tensors, 0 ).Shape, { BD_BatchWidth, BD_Channels }, OutputDim( dims, 0 ) ),
		"marking output dimensions failed", onnxNode );
	CheckNeoOnnxInternal( SetTensorDim( InputTensor( tensors, 0 ).Shape, { BD_BatchWidth, BD_Channels }, InputDim( dims, 0 ) ),
		"marking input dimensions failed", onnxNode );
}

void CGemmNode::AddLayers( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn )
{
	CPtr<CFullyConnectedLayer> fc = new CFullyConnectedLayer( dnn.GetMathEngine() );
	fc->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	const CTensorShape& matrixShape = InputTensor( tensors, 1 ).Shape;
	const int numberOfElements = matrixShape[transB == 0 ? 1 : 0];

	fc->SetNumberOfElements( numberOfElements );

	CPtr<CDnnBlob> weight = InputTensor( tensors, 1 ).Data->GetCopy();
	CBlobDesc weightDesc( CT_Float );
	weightDesc.SetDimSize( BD_BatchWidth, weight->GetDesc().DimSize( 0 ) );
	weightDesc.SetDimSize( BD_Channels, weight->GetDesc().DimSize( 1 ) );
	weight->ReinterpretDimensions( weightDesc );

	// If there is a 'Flatten' node before this, we need to reorder weights.
	weight = reorderWeightAfterFlatten( graph, tensors, dims, weight );

	fc->SetWeightsData( weight );

	if( InputCount() > 2 ) {
		fc->SetFreeTermData( InputTensor( tensors, 2 ).Data );
	} else {
		fc->SetZeroFreeTerm( true );
	}

	fc->Connect( 0, *InputMapping( mappings, 0 ).Layer, InputMapping( mappings, 0 ).OutputIndex );
	dnn.AddLayer( *fc );

	OutputMapping( mappings, 0 ) = CNeoMLMapping( fc, 0 );
}

// Reorders weight matrix if this 'Gemm' is located after 'Flatten'.
CPtr<CDnnBlob> CGemmNode::reorderWeightAfterFlatten( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CDnnBlob* weight ) const
{
	const CNode* flatten = graph[GetInput( 0 ).NodeIndex];
	
	if( dynamic_cast<const CFlattenNode*>( flatten ) == nullptr ) {
		return weight;
	}

	const CTensorShape& flattenInputShape = flatten->InputTensor( tensors, 0 ).Shape;
	const CTensorDim& flattenInputDim = flatten->InputDim( dims, 0 );

	CBlobDesc newWeightDesc( CT_Float );
	for( int dimIndex = 0; dimIndex < flattenInputShape.Size(); ++dimIndex ) {
		newWeightDesc.SetDimSize( flattenInputDim[dimIndex], flattenInputShape[dimIndex] );
	}

	if( ( newWeightDesc.Height() == 1 && newWeightDesc.Width() == 1 )
		|| ( newWeightDesc.Channels() == 1 && newWeightDesc.Depth() == 1 ) )
	{
		return weight;
	}

	// Weights needs conversion from CHW to HWC
	IMathEngine& mathEngine = weight->GetMathEngine();
	CPtr<CDnnBlob> newWeight = weight->GetClone();
	mathEngine.TransposeMatrix( weight->GetObjectCount(), weight->GetData(), newWeightDesc.Channels(), newWeightDesc.Depth(),
		newWeightDesc.Height() * newWeightDesc.Width(), 1, newWeight->GetData(), newWeight->GetDataSize() );
	return newWeight;
}

} // namespace NeoOnnx
