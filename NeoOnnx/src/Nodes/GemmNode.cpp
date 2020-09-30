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
#include "GraphCache.h"
#include "FlattenNode.h"
#include "NeoOnnxCheck.h"
#include "NodeUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGemmNode::CGemmNode( int nodeIndex, const onnx::NodeProto& gemm, int opsetVersion ) :
	COpNode( nodeIndex, gemm, opsetVersion ),
	alpha( Attributes.GetOptionalFloat( "alpha", 1.f ) ),
	beta( Attributes.GetOptionalFloat( "beta", 1.f ) ),
	transA( Attributes.GetOptionalInt( "transA", 0 ) ),
	transB( Attributes.GetOptionalInt( "transB", 0 ) )
{
	// Older versions have broadcast support
	CheckNeoOnnxSupport( OpsetVersion >= 7 && OpsetVersion <= MaxOpsetVersion, "opset version", gemm );

	CheckOnnxProtocol( InputCount() == 2 || InputCount() == 3, "node must have 2 or 3 inputs", gemm );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", gemm );

	CheckNeoOnnxSupport( alpha == 1.0f, "alpha != 1", gemm );
	CheckNeoOnnxSupport( beta == 1.0f, "beta != 1", gemm );
	CheckNeoOnnxSupport( transA == 0, "transA != 0", gemm );
	CheckNeoOnnxSupport( transB != 0, "transB == 0", gemm );
}

void CGemmNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "constant input", OnnxNode );
	const CTensorShape& inputShape = tensors[Input[0]].Shape;
	CheckOnnxProtocol( inputShape.Size() == 2, "input must be 2-dimensional", OnnxNode );
	const int batchSize = inputShape[transA == 0 ? 0 : 1];
	const int inputObjectSize = inputShape[transA == 0 ? 1 : 0];

	CheckNeoOnnxSupport( tensors[Input[1]].Data != nullptr, "non-constant weights", OnnxNode );
	const CTensorShape& matrixShape = tensors[Input[1]].Shape;
	CheckOnnxProtocol( matrixShape.Size() == 2, "weights must be 2-dimensional", OnnxNode );
	CheckOnnxProtocol( matrixShape[transB == 0 ? 0 : 1] == inputObjectSize, "wrong weight size", OnnxNode );
	const int numberOfElements = matrixShape[transB == 0 ? 1 : 0];

	if( InputCount() == 3 ) {
		CheckNeoOnnxSupport( tensors[Input[2]].Data != nullptr, "non-constant bias", OnnxNode );
		const CTensorShape& biasShape = tensors[Input[2]].Shape;
		CheckOnnxProtocol( biasShape.Size() == 1, "bias must be 1-dimensional", OnnxNode );
		CheckOnnxProtocol( biasShape[0] == numberOfElements, "wrong bias size", OnnxNode );
	}

	tensors[Output[0]].Shape = { batchSize, numberOfElements };

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
}

void CGemmNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	// Gemm operator in onnx always works with 2-dimensional tensors
	CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, { BD_BatchWidth, BD_Channels }, dims[Output[0]] ),
		"labeling output dimensions failed", OnnxNode );
	CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, { BD_BatchWidth, BD_Channels }, dims[Input[0]] ),
		"labeling input dimensions failed", OnnxNode );
}

void CGemmNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CFullyConnectedLayer> fc = new CFullyConnectedLayer( dnn.GetMathEngine() );
	fc->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	const CTensorShape& matrixShape = tensors[Input[1]].Shape;
	const int numberOfElements = matrixShape[transB == 0 ? 1 : 0];

	fc->SetNumberOfElements( numberOfElements );

	CPtr<CDnnBlob> weight = tensors[Input[1]].Data->GetCopy();
	CBlobDesc weightDesc( CT_Float );
	weightDesc.SetDimSize( BD_BatchWidth, weight->GetDesc().DimSize( 0 ) );
	weightDesc.SetDimSize( BD_Channels, weight->GetDesc().DimSize( 1 ) );
	weight->ReinterpretDimensions( weightDesc );

	// If there is a 'Flatten' node before this we have to reorder weights
	weight = RepackWeightIfFlattened( graph[Input[0]], tensors, dims, weight );

	fc->SetWeightsData( weight );

	if( InputCount() > 2 ) {
		fc->SetFreeTermData( tensors[Input[2]].Data );
	} else {
		fc->SetZeroFreeTerm( true );
	}

	fc->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *fc );

	neoMLLinks[Output[0]] = CNeoMLLink( fc, 0 );
}

} // namespace NeoOnnx
