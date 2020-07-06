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

CBatchNormalizationNode::CBatchNormalizationNode( const onnx::NodeProto& batchNormalization, int opsetVersion,
		IMathEngine& /*mathEngine*/ ) :
	CNode( batchNormalization, opsetVersion ),
	eps( attributes.GetOptionalFloat( "epsilon", 1e-5f ) )
{
	// Older versions of this operator have spatial flag which can lead to wrong calculation
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", batchNormalization );

	CheckNeoOnnxSupport( opsetVersion > 7 || attributes.GetOptionalInt( "spatial", 1 ) == 1,
		"non-spatial batch normalization is not supported", batchNormalization );

	CheckNeoOnnxSupport( opsetVersion > 6 || attributes.GetOptionalInt( "is_test", 0 ) != 0,
		"training batch normalization is not supported", batchNormalization );

	CheckOnnxProtocol( input.Size() == 5 || input.Size() == 6, "node must have 5 or 6 inputs", onnxNode );
	CheckNeoOnnxSupport( OutputCount() == 1, "node must have 1 output", onnxNode );
}

void CBatchNormalizationNode::CalcOutputShape()
{
	InputTensor( 0 ).Shape.CopyTo( output[0].Shape );
}

void CBatchNormalizationNode::CalcOutputData()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).Data == nullptr, "output pre-calculation", onnxNode );
	// The output[0].Data was already set to nullptr in default constructor.
}

void CBatchNormalizationNode::MarkTensorDims()
{
	if( !InputTensor( 0 ).Dim.IsEmpty() ) {
		CheckNeoOnnxInternal( output[0].SetTensorDim( InputTensor( 0 ).Dim ),
			"marking output dimensions failed", onnxNode );
	}

	if( !output[0].Dim.IsEmpty() ) {
		CheckNeoOnnxInternal( InputTensor( 0 ).SetTensorDim( output[0].Dim ),
			"marking input dimensions failed", onnxNode );
	}
}

void CBatchNormalizationNode::AddLayers( CDnn& dnn )
{
	CheckNeoOnnxInternal( InputTensor( 0 ).Dim[1] == BD_Channels,
		"operation must be performed along input's BD_Channels", onnxNode );
	CheckNeoOnnxInternal( output[0].Dim[1] == BD_Channels,
		"operation must be performed along output's BD_Channels", onnxNode );

	CPtr<CBatchNormalizationLayer> bnLayer = new CBatchNormalizationLayer( dnn.GetMathEngine() );
	bnLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	bnLayer->SetChannelBased( true );
	bnLayer->SetFinalParams( calculateFinalParams() );

	bnLayer->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	dnn.AddLayer( *bnLayer );
	
	neoMLInputInfo.Add( CNeoMLInputInfo( bnLayer, 0 ) );
}

CPtr<CDnnBlob> CBatchNormalizationNode::calculateFinalParams()
{
	const int channels = InputTensor( 0 ).Shape[1];

	for( int inputIndex = 1; inputIndex < 5; ++inputIndex ) {
		CheckNeoOnnxSupport( InputTensor( inputIndex ).Data != nullptr,
			"non-constant weights", onnxNode );
		CheckOnnxProtocol( InputTensor( inputIndex ).Shape.Size() == 1,
			"weights must be 1-dimensional", onnxNode );
		CheckOnnxProtocol( InputTensor( inputIndex ).Shape[0] == channels,
			"weights must have 'channels' length", onnxNode );
	}

	const CDnnBlob* scale = InputTensor( 1 ).Data;
	const CDnnBlob* bias = InputTensor( 2 ).Data;
	const CDnnBlob* mean = InputTensor( 3 ).Data;
	const CDnnBlob* var = InputTensor( 4 ).Data;

	IMathEngine& mathEngine = scale->GetMathEngine();

	// Calculating final params.
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
