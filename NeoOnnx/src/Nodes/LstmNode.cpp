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

CLstmNode::CLstmNode( int nodeIndex, const onnx::NodeProto& lstm, int opsetVersion ) :
	COpNode( nodeIndex, lstm, opsetVersion ),
	direction( attributes.GetOptionalString( "direction", "forward" ) ),
	hiddenSize( attributes.GetRequiredInt( "hidden_size" ) )
{
	// The differences between versions are in some flags and support (i.e. output_sequence)
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", lstm );

	CheckOnnxProtocol( InputCount() >= 3 && InputCount() <= 8, "node must have from 3 upto 8 inputs", lstm );
	CheckOnnxProtocol( OutputCount() >= 1 && OutputCount() <= 3, "node must have from 1 upto 3 outputs", lstm );

	CheckNeoOnnxSupport( direction != "bidirectional", "bidirectional LSTM", lstm ); // TODO: add support of bidirectional LSTM
}

void CLstmNode::CalcOutputTensors( CGraphTensors& tensors, IMathEngine& mathEngine )
{
	CheckNeoOnnxSupport( InputTensor( tensors, 0 ).Data == nullptr, "constant input", onnxNode );
	CheckNeoOnnxSupport( InputTensor( tensors, 1 ).Data != nullptr, "non-constant weight", onnxNode );
	CheckNeoOnnxSupport( InputTensor( tensors, 2 ).Data != nullptr, "non-constant recurrent weight", onnxNode );

	const CTensorShape& inputShape = InputTensor( tensors, 0 ).Shape;

	CPtr<CDnnBlob> biasValue = nullptr;
	if( InputCount() > 3 && GetInput( 3 ).NodeIndex != NotFound ) {
		const CTensor& bias = InputTensor( tensors, 3 );
		CheckNeoOnnxSupport( bias.Data != nullptr, "non-constant bias", onnxNode );
	}

	// NeoML doesn't support sequence lengths
	CheckNeoOnnxSupport( InputCount() <= 4 || GetInput( 4 ).NodeIndex == NotFound, "sequence lengths", onnxNode );

	if( InputCount() > 5 && GetInput( 5 ).NodeIndex != NotFound ) {
		const CTensor& initialH = InputTensor( tensors, 5 );
		CheckNeoOnnxSupport( initialH.Data != nullptr, "non-constant initial history", onnxNode );
	}

	if( InputCount() > 6 && GetInput( 6 ).NodeIndex != NotFound ) {
		const CTensor& initialC = InputTensor( tensors, 6 );
		CheckNeoOnnxSupport( initialC.Data != nullptr, "non-constant initial state", onnxNode );
	}

	CheckNeoOnnxSupport( InputCount() < 8 || GetInput( 7 ).NodeIndex == NotFound, "peepholes",
		onnxNode ); // NeoML doesn't support peepholes

	const int sequenceLength = inputShape[0];
	const int batchSize = inputShape[1];

	OutputTensor( tensors, 0 ).Shape = { sequenceLength, 1, batchSize, hiddenSize };

	CheckNeoOnnxSupport( InputTensor( tensors, 0 ).Data == nullptr, "output pre-calculation", onnxNode );
	// The OutputTensor( tensors, 0 ).Data was already set to nullptr in default constructor.
}

void CLstmNode::MarkTensorDims( const CGraphTensors& tensors, CGraphDims& dims )
{
	CheckNeoOnnxInternal( SetTensorDim( OutputTensor( tensors, 0 ).Shape, { BD_BatchLength, BD_ListSize, BD_BatchWidth, BD_Channels },
		OutputDim( dims, 0 ) ), "marking output dimensions failed", onnxNode );
	CheckNeoOnnxInternal( SetTensorDim( InputTensor( tensors, 0 ).Shape, { BD_BatchLength, BD_BatchWidth, BD_Channels },
		InputDim( dims, 0 ) ), "marking input dimensions failed", onnxNode );
}

void CLstmNode::AddLayers( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn )
{
	IMathEngine& mathEngine = dnn.GetMathEngine();
	CPtr<CLstmLayer> lstmLayer = new CLstmLayer( mathEngine );
	lstmLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	CPtr<CDnnBlob> weightMatrix = InputTensor( tensors, 1 ).Data->GetCopy();
	CPtr<CDnnBlob> recurWeightMatrix = InputTensor( tensors, 2 ).Data->GetCopy();

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

	if( InputCount() > 3 && GetInput( 3 ).NodeIndex != NotFound ) {
		CPtr<CDnnBlob> neoMLBias = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, 1, 4 * hiddenSize );
		mathEngine.VectorCopy( neoMLBias->GetData(), InputTensor( tensors, 3 ).Data->GetData(), 4 * hiddenSize );
		neoMLBias = reorderGates( neoMLBias, BD_Channels );
		lstmLayer->SetInputFreeTermData( neoMLBias );
		lstmLayer->SetRecurFreeTermData( nullptr );
	}

	// TODO: Add support for other inputs.
	lstmLayer->Connect( 0, *InputMapping( mappings, 0 ).Layer, InputMapping( mappings, 0 ).OutputIndex );
	dnn.AddLayer( *lstmLayer );

	OutputMapping( mappings, 0 ) = CNeoMLMapping( lstmLayer, 0 );
	// TODO: add support of other outputs
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
