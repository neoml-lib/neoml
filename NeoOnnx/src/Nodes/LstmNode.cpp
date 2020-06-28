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

#include "LstmNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CLstmNode::CLstmNode( const onnx::NodeProto& lstm, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( lstm, nodeOutputs ),
	direction( attributes.GetOptionalString( "direction", "forward" ) ),
	hiddenSize( attributes.GetRequiredInt( "hidden_size" ) )
{
	CheckOnnxProtocol( input.Size() >= 3 && input.Size() <= 8, "node must have from 3 up to 8 inputs", lstm );
	CheckOnnxProtocol( OutputCount() >= 1 && OutputCount() <= 3, "node must have from 1 up to 3 outputs", lstm );

	CheckNeoOnnxSupport( direction != "bidirectional", "bidirectional LSTM", lstm );
}

void CLstmNode::OnnxReshape()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).GetType() == TT_DataTensor, "constant input", onnxNode );
	CheckNeoOnnxSupport( InputTensor( 1 ).GetType() == TT_ConstantTensor, "non-constant weight", onnxNode );
	CheckNeoOnnxSupport( InputTensor( 2 ).GetType() == TT_ConstantTensor, "non-constant recurrent weight", onnxNode );

	const CTensorShape& inputShape = InputTensor( 0 ).GetShape();

	CPtr<CDnnBlob> biasValue = nullptr;
	if( input.Size() > 3 && input[3].InputNode != nullptr ) {
		CTensor& bias = InputTensor( 3 );
		CheckNeoOnnxSupport( bias.GetType() == TT_ConstantTensor, "non-constant bias", onnxNode );
	}

	CheckNeoOnnxSupport( input.Size() <= 4 || input[4].InputNode == nullptr, "sequence lengths",
		onnxNode ); // NeoML doesn't support sequence lengths

	if( input.Size() > 5 && input[5].InputNode != nullptr ) {
		const CTensor& initialH = InputTensor( 5 );
		CheckNeoOnnxSupport( initialH.GetType() == TT_ConstantTensor, "non-constant initial history", onnxNode );
	}

	if( input.Size() > 6 && input[6].InputNode != nullptr ) {
		const CTensor& initialC = InputTensor( 6 );
		CheckNeoOnnxSupport( initialC.GetType() == TT_ConstantTensor, "non-constant initial state", onnxNode );
	}

	CheckNeoOnnxSupport( input.Size() < 8 || input[7].InputNode == nullptr, "peepholes",
		onnxNode ); // NeoML doesn't support peepholes

	const int sequenceLength = inputShape[0];
	const int batchSize = inputShape[1];

	outputData.Add( CTensor( TT_DataTensor, { sequenceLength, 1, batchSize, hiddenSize } ) );
	for( int outputIndex = 1; outputIndex < OutputCount(); ++outputIndex ) {
		outputData.Add( CTensor() );
	}
}

void CLstmNode::MarkTensorDims()
{
	CheckNeoOnnxInternal( outputData[0].SetTensorDim( { BD_BatchLength, BD_ListSize, BD_BatchWidth, BD_Channels } ),
		"marking output dimensions failed", onnxNode );
	CheckNeoOnnxInternal( InputTensor( 0 ).SetTensorDim( { BD_BatchLength, BD_BatchWidth, BD_Channels } ),
		"marking input dimensions failed", onnxNode );
}

void CLstmNode::AddLayers( CDnn& dnn )
{
	IMathEngine& mathEngine = dnn.GetMathEngine();
	CPtr<CLstmLayer> lstmLayer = new CLstmLayer( mathEngine );
	lstmLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	CPtr<CDnnBlob> weightMatrix = InputTensor( 1 ).GetData()->GetCopy();
	CPtr<CDnnBlob> recurWeightMatrix = InputTensor( 2 ).GetData()->GetCopy();

	const int inputObjectSize = weightMatrix->DimSize( 1 );

	CBlobDesc blobDesc( CT_Float );
	blobDesc.SetDimSize( BD_BatchWidth, 4 * hiddenSize );
	blobDesc.SetDimSize( BD_Channels, inputObjectSize );
	weightMatrix->ReinterpretDimensions( blobDesc );

	blobDesc.SetDimSize( BD_Channels, hiddenSize );
	recurWeightMatrix->ReinterpretDimensions( blobDesc );

	lstmLayer->SetHiddenSize( hiddenSize );

	weightMatrix = reorderGates( weightMatrix, BD_BatchWidth );
	recurWeightMatrix = reorderGates( recurWeightMatrix, BD_BatchWidth );

	lstmLayer->SetInputWeightsData( weightMatrix );
	lstmLayer->SetRecurWeightsData( recurWeightMatrix );

	if( input.Size() > 3 && input[3].InputNode != nullptr ) {
		CPtr<CDnnBlob> neoMLBias = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, 1, 4 * hiddenSize );
		mathEngine.VectorCopy( neoMLBias->GetData(), InputTensor( 3 ).GetData()->GetData(), 4 * hiddenSize );
		neoMLBias = reorderGates( neoMLBias, BD_Channels );
		lstmLayer->SetInputFreeTermData( neoMLBias );
		lstmLayer->SetRecurFreeTermData( nullptr );
	}

	lstmLayer->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	dnn.AddLayer( *lstmLayer );

	outputInfo.Add( COutputInfo( lstmLayer, 0 ) );
	for( int outputIndex = 1; outputIndex < OutputCount(); ++outputIndex ) {
		outputInfo.Add( COutputInfo() );
	}
}

CPtr<CDnnBlob> CLstmNode::reorderGates( CPtr<CDnnBlob> blob, TBlobDim dim )
{
	IMathEngine& mathEngine = blob->GetMathEngine();

	CObjectArray<CDnnBlob> onnxGateWeight;

	CBlobDesc gateWeightDesc = blob->GetDesc();
	gateWeightDesc.SetDimSize( dim, gateWeightDesc.DimSize( dim ) / 4 );
	
	for( int gateIndex = 0; gateIndex < 4; ++gateIndex ) {
		onnxGateWeight.Add( CDnnBlob::CreateBlob( mathEngine, gateWeightDesc ) );
	}

	CDnnBlob::SplitByDim( mathEngine, dim, blob, onnxGateWeight );

	CObjectArray<CDnnBlob> neoMLGateWeight;
	neoMLGateWeight.Add( onnxGateWeight[3] ); // Main gate.
	neoMLGateWeight.Add( onnxGateWeight[2] ); // Forget gate.
	neoMLGateWeight.Add( onnxGateWeight[0] ); // Input gate.
	neoMLGateWeight.Add( onnxGateWeight[1] ); // Reset gate.

	CPtr<CDnnBlob> result = blob->GetClone();
	CDnnBlob::MergeByDim( mathEngine, dim, neoMLGateWeight, result );
	return result;
}

} // namespace NeoOnnx
