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

#include "MatMulNode.h"
#include "GraphCache.h"
#include "NodeUtils.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CMatMulNode::CMatMulNode( int nodeIndex, const onnx::NodeProto& matMul, int opsetVersion ) :
	COpNode( nodeIndex, matMul, opsetVersion )
{
	// The differences between versions are in legacy optimization flags
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", matMul );

	CheckOnnxProtocol( InputCount() == 2, "node must have 2 inputs", matMul );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", matMul );
}

void CMatMulNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	// The only scenario we support is this:
	//     first input - user-provided data, single matrix
	//     second input - pre-calculated data, single matrix
	// In this case we can emulate this node by CFullyConnectedLayer
	const CTensor& firstInput = tensors[Input[0]];
	const CTensor& secondInput = tensors[Input[1]];
	CheckNeoOnnxSupport( firstInput.Data == nullptr, "pre-calculated first input", OnnxNode );
	CheckNeoOnnxSupport( firstInput.Shape.Size() == 2, "non-2D first input", OnnxNode );
	CheckNeoOnnxSupport( secondInput.Data != nullptr, "user-provided second input", OnnxNode );
	CheckNeoOnnxSupport( secondInput.Shape.Size() == 2, "non-2D first input", OnnxNode );

	CTensorShape& outputShape = tensors[Output[0]].Shape;
	outputShape.Add( firstInput.Shape[0] );
	outputShape.Add( secondInput.Shape[1] );
}

void CMatMulNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( !dims[Input[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, dims[Input[0]], dims[Output[0]] ),
			"labeling output dimensions failed", OnnxNode );
	}

	if( !dims[Output[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, dims[Output[0]], dims[Input[0]] ),
			"labeling input dimensions failed", OnnxNode );
	}
}

void CMatMulNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CFullyConnectedLayer> fc = new CFullyConnectedLayer( dnn.GetMathEngine() );
	fc->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );
	fc->SetNumberOfElements( tensors[Output[0]].Shape[1] );
	
	const int inputElems = tensors[Input[1]].Shape[0];
	const int outputElems = tensors[Input[1]].Shape[1];

	CBlobDesc weightDesc = tensors[Input[1]].Data->GetDesc();
	weightDesc.SetDimSize( 0, outputElems );
	weightDesc.SetDimSize( 1, inputElems );
	CPtr<CDnnBlob> weight = CDnnBlob::CreateBlob( dnn.GetMathEngine(), weightDesc );
	weight->TransposeFrom( tensors[Input[1]].Data, 0, 1 );

	weightDesc = CBlobDesc( CT_Float );
	weightDesc.SetDimSize( BD_BatchWidth, outputElems );
	weightDesc.SetDimSize( BD_Channels, inputElems );
	weight->ReinterpretDimensions( weightDesc );
	weight = RepackWeightIfFlattened( graph[Input[0]], tensors, dims, weight );
	fc->SetWeightsData( weight );
	fc->SetZeroFreeTerm( true );

	fc->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *fc );
	neoMLLinks[Output[0]] = CNeoMLLink( fc, 0 );
}

} // namespace NeoOnnx
