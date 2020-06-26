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

CBatchNormalizationNode::CBatchNormalizationNode( const onnx::NodeProto& batchNormalization,
		CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( batchNormalization, nodeOutputs ),
	eps( attributes.GetOptionalFloat( "epsilon", 1e-5f ) )
{
	CheckOnnxProtocol( input.Size() == 5 || input.Size() == 6, "node must have 5 or 6 inputs", onnxNode );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", onnxNode );
}

void CBatchNormalizationNode::OnnxReshape()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).GetType() == TT_DataTensor, "constant input data", onnxNode );
	outputData.Add( InputTensor( 0 ) );
}

void CBatchNormalizationNode::MarkTensorDims()
{
	if( !InputTensor( 0 ).GetTensorDim().IsEmpty() ) {
		CheckNeoOnnxInternal( outputData[0].SetTensorDim( InputTensor( 0 ).GetTensorDim() ),
			"marking output dimensions failed", onnxNode );
	}

	if( !outputData[0].GetTensorDim().IsEmpty() ) {
		CheckNeoOnnxInternal( InputTensor( 0 ).SetTensorDim( outputData[0].GetTensorDim() ),
			"marking input dimensions failed", onnxNode );
	}
}

void CBatchNormalizationNode::AddLayers( CDnn& dnn )
{
	CheckNeoOnnxInternal( InputTensor( 0 ).GetTensorDim()[1] == BD_Channels,
		"operation must be performed along input's BD_Channels", onnxNode );
	CheckNeoOnnxInternal( outputData[0].GetTensorDim()[1] == BD_Channels,
		"operation must be performed along output's BD_Channels", onnxNode );

	CPtr<CBatchNormalizationLayer> bnLayer = new CBatchNormalizationLayer( dnn.GetMathEngine() );
	bnLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	bnLayer->SetChannelBased( true );
	bnLayer->SetFinalParams( calculateFinalParams() );

	bnLayer->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	dnn.AddLayer( *bnLayer );
	
	outputInfo.Add( COutputInfo( bnLayer, 0 ) );
}

CPtr<CDnnBlob> CBatchNormalizationNode::calculateFinalParams()
{
	const int channels = InputTensor( 0 ).GetShape()[1];

	for( int inputIndex = 1; inputIndex < 5; ++inputIndex ) {
		CheckNeoOnnxSupport( InputTensor( inputIndex ).GetType() == TT_ConstantTensor,
			"non-constant weights", onnxNode );
		CheckOnnxProtocol( InputTensor( inputIndex ).GetShape().Size() == 1,
			"weights must be 1-dimensional", onnxNode );
		CheckOnnxProtocol( InputTensor( inputIndex ).GetShape()[0] == channels,
			"weights must have 'channels' length", onnxNode );
	}

	const CDnnBlob* scale = InputTensor( 1 ).GetData();
	const CDnnBlob* bias = InputTensor( 2 ).GetData();
	const CDnnBlob* mean = InputTensor( 3 ).GetData();
	const CDnnBlob* var = InputTensor( 4 ).GetData();

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
