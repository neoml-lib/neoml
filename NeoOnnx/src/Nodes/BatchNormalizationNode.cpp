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

#include "BatchNormalizationNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CBatchNormalizationNode::CBatchNormalizationNode( int nodeIndex, const onnx::NodeProto& batchNormalization, int opsetVersion ) :
	COpNode( nodeIndex, batchNormalization, opsetVersion ),
	eps( Attributes.GetOptionalFloat( "epsilon", 1e-5f ) )
{
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", batchNormalization );

	CheckNeoOnnxSupport( OpsetVersion > 6 || Attributes.GetOptionalInt( "is_test", 0 ) != 0,
		"training batch normalization is not supported", batchNormalization );

	CheckOnnxProtocol( InputCount() == 5 || InputCount() == 6, "node must have 5 or 6 inputs", OnnxNode );
	CheckNeoOnnxSupport( OutputCount() == 1, "node must have 1 output", OnnxNode );
}

void CBatchNormalizationNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	tensors[Input[0]].Shape.CopyTo( tensors[Output[0]].Shape );

	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
}

void CBatchNormalizationNode::MarkTensorDims( const CTensorCache& tensors, CDimCache& dims )
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

void CBatchNormalizationNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CheckNeoOnnxInternal( ( dims[Input[0]] )[1] == BD_Channels,
		"operation must be performed along input's BD_Channels", OnnxNode );
	CheckNeoOnnxInternal( ( dims[Output[0]] )[1] == BD_Channels,
		"operation must be performed along output's BD_Channels", OnnxNode );

	CPtr<CBatchNormalizationLayer> bnLayer = new CBatchNormalizationLayer( dnn.GetMathEngine() );
	bnLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	// Since v9 batch normalization is always spatial
	// Before that spatial was set by a flag
	if( OpsetVersion >= 9 || Attributes.GetOptionalInt( "spatial", 1 ) != 0 ) {
		bnLayer->SetChannelBased( true );
	}

	bnLayer->SetFinalParams( calculateFinalParams( tensors ) );

	bnLayer->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *bnLayer );
	
	neoMLLinks[Output[0]] = CNeoMLLink( bnLayer, 0 );
}

// Calculate final params blob based on onnx node's inputs
// This format can is compatible with NeoML::CBatchNormalizationLayer
CPtr<CDnnBlob> CBatchNormalizationNode::calculateFinalParams( const CTensorCache& tensors )
{
	const int channels = tensors[Input[0]].Shape[1];

	for( int inputIndex = 1; inputIndex < 5; ++inputIndex ) {
		CheckNeoOnnxSupport( tensors[Input[inputIndex]].Data != nullptr,
			"non-constant weights", OnnxNode );
		CheckOnnxProtocol( tensors[Input[inputIndex]].Shape.Size() == 1,
			"weights must be 1-dimensional", OnnxNode );
		CheckOnnxProtocol( tensors[Input[inputIndex]].Shape[0] == channels,
			"weights must have 'channels' length", OnnxNode );
	}

	const CDnnBlob* scale = tensors[Input[1]].Data;
	const CDnnBlob* bias = tensors[Input[2]].Data;
	const CDnnBlob* mean = tensors[Input[3]].Data;
	const CDnnBlob* var = tensors[Input[4]].Data;

	IMathEngine& mathEngine = scale->GetMathEngine();

	// Calculating final params
	CPtr<CDnnBlob> finalParams = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, 2, channels );
	CFloatHandle gamma = finalParams->GetObjectData( 0 );

	mathEngine.VectorFill( gamma, eps, channels );
	mathEngine.VectorAdd( var->GetData(), gamma, gamma, channels );
	mathEngine.VectorSqrt( gamma, gamma, channels );
	mathEngine.VectorInv( gamma, gamma, channels );
	mathEngine.VectorEltwiseMultiply( scale->GetData(), gamma, gamma, channels );

	CFloatHandle beta = finalParams->GetObjectData( 1 );
	mathEngine.VectorEltwiseMultiply( mean->GetData(), gamma, beta, channels );
	mathEngine.VectorSub( bias->GetData(), beta, beta, channels );

	return finalParams;
}

} // namespace NeoOnnx
