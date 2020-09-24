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
#include "NodeUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CLstmNode::CLstmNode( int nodeIndex, const onnx::NodeProto& lstm, int opsetVersion ) :
	COpNode( nodeIndex, lstm, opsetVersion ),
	direction( Attributes.GetOptionalString( "direction", "forward" ) ),
	hiddenSize( Attributes.GetRequiredInt( "hidden_size" ) )
{
	// The differences between versions are in some flags and support (i.e. output_sequence)
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", lstm );

	CheckOnnxProtocol( InputCount() >= 3 && InputCount() <= 8, "node must have from 3 upto 8 inputs", lstm );
	CheckOnnxProtocol( OutputCount() >= 1 && OutputCount() <= 3, "node must have from 1 upto 3 outputs", lstm );

	CheckNeoOnnxSupport( direction != "bidirectional", "bidirectional LSTM", lstm );
}

void CLstmNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "constant input", OnnxNode );
	CheckNeoOnnxSupport( tensors[Input[1]].Data != nullptr, "non-constant weight", OnnxNode );
	CheckNeoOnnxSupport( tensors[Input[2]].Data != nullptr, "non-constant recurrent weight", OnnxNode );

	const CTensorShape& inputShape = tensors[Input[0]].Shape;

	CPtr<CDnnBlob> biasValue = nullptr;
	if( InputCount() > 3 && Input[3].NodeIndex != NotFound ) {
		const CTensor& bias = tensors[Input[3]];
		CheckNeoOnnxSupport( bias.Data != nullptr, "non-constant bias", OnnxNode );
	}

	// NeoML doesn't support sequence lengths
	CheckNeoOnnxSupport( InputCount() <= 4 || Input[4].NodeIndex == NotFound, "sequence lengths", OnnxNode );
	if( InputCount() > 5 && Input[5].NodeIndex != NotFound ) {
		const CTensor& initialH = tensors[Input[5]];
		CheckNeoOnnxSupport( initialH.Data != nullptr, "non-constant initial history", OnnxNode );
	}
	if( InputCount() > 6 && Input[6].NodeIndex != NotFound ) {
		const CTensor& initialC = tensors[Input[6]];
		CheckNeoOnnxSupport( initialC.Data != nullptr, "non-constant initial state", OnnxNode );
	}

	CheckNeoOnnxSupport( InputCount() < 8 || Input[7].NodeIndex == NotFound, "peepholes", OnnxNode );
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );

	const int sequenceLength = inputShape[0];
	const int batchSize = inputShape[1];
	tensors[Output[0]].Shape = { sequenceLength, 1, batchSize, hiddenSize };
}

void CLstmNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, { BD_BatchLength, BD_ListSize, BD_BatchWidth, BD_Channels },
		dims[Output[0]] ), "labeling output dimensions failed", OnnxNode );
	CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, { BD_BatchLength, BD_BatchWidth, BD_Channels },
		dims[Input[0]] ), "labeling input dimensions failed", OnnxNode );
}

void CLstmNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	IMathEngine& mathEngine = dnn.GetMathEngine();
	CPtr<CLstmLayer> lstmLayer = new CLstmLayer( mathEngine );
	lstmLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	CPtr<CDnnBlob> weightMatrix = tensors[Input[1]].Data->GetCopy();
	CPtr<CDnnBlob> recurWeightMatrix = tensors[Input[2]].Data->GetCopy();

	const int inputObjectSize = weightMatrix->DimSize( 1 );

	CBlobDesc blobDesc( CT_Float );
	blobDesc.SetDimSize( BD_BatchWidth, 4 * hiddenSize );
	blobDesc.SetDimSize( BD_Channels, inputObjectSize );
	weightMatrix->ReinterpretDimensions( blobDesc );
	blobDesc.SetDimSize( BD_Channels, hiddenSize );
	recurWeightMatrix->ReinterpretDimensions( blobDesc );

	lstmLayer->SetHiddenSize( hiddenSize );
	weightMatrix = reorderGates( weightMatrix, BD_BatchWidth );
	weightMatrix = RepackWeightIfFlattened( graph[Input[0]], tensors, dims, weightMatrix );
	recurWeightMatrix = reorderGates( recurWeightMatrix, BD_BatchWidth );
	recurWeightMatrix = RepackWeightIfFlattened( graph[Input[0]], tensors, dims, recurWeightMatrix );

	lstmLayer->SetInputWeightsData( weightMatrix );
	lstmLayer->SetRecurWeightsData( recurWeightMatrix );

	if( InputCount() > 3 && Input[3].NodeIndex != NotFound ) {
		CPtr<CDnnBlob> neoMLBias = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, 1, 4 * hiddenSize );
		mathEngine.VectorCopy( neoMLBias->GetData(), tensors[Input[3]].Data->GetData(), 4 * hiddenSize );
		neoMLBias = reorderGates( neoMLBias, BD_Channels );
		neoMLBias = RepackWeightIfFlattened( graph[Input[0]], tensors, dims, neoMLBias );
		lstmLayer->SetInputFreeTermData( neoMLBias );
		lstmLayer->SetRecurFreeTermData( nullptr );
	}

	lstmLayer->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *lstmLayer );
	neoMLLinks[Output[0]] = CNeoMLLink( lstmLayer, 0 );
}

// Converts lstm weights from onnx format to NeoML
// because gate weights in onnx and NeoML are oredered differently
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
	neoMLGateWeight.Add( onnxGateWeight[3] ); // Main gate
	neoMLGateWeight.Add( onnxGateWeight[2] ); // Forget gate
	neoMLGateWeight.Add( onnxGateWeight[0] ); // Input gate
	neoMLGateWeight.Add( onnxGateWeight[1] ); // Reset gate

	CPtr<CDnnBlob> result = blob->GetClone();
	CDnnBlob::MergeByDim( mathEngine, dim, neoMLGateWeight, result );
	return result;
}

} // namespace NeoOnnx
