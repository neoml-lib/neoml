/* Copyright © 2017-2024 ABBYY

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

#include "LstmOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"
#include <NeoML/Dnn/Layers/ConcatLayer.h>
#include <NeoML/Dnn/Layers/SplitLayer.h>

#include "onnx.pb.h"

#include <algorithm>

using namespace NeoML;

namespace NeoOnnx {

CLstmOperator::CLstmOperator( const onnx::NodeProto& lstm, int opsetVersion ) :
	CLayerOperator( lstm, opsetVersion ),
	direction( "forward" ),
	hiddenSize( -1 )
{
	// v1 - original
	// v7 - added initial state and peephole weight inputs
	// v14 - layout attribute added
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() >= 3 && InputCount() <= 8, "operator must have from 3 upto 8 inputs", *this );
	CheckOnnxProtocol( OutputCount() >= 1 && OutputCount() <= 3, "operator must have from 1 upto 3 outputs", *this );

	GetAttribute( "direction", direction );
	CheckOnnxProtocol( GetAttribute( "hidden_size", hiddenSize ), "'hidden_size' attribute is missing", *this );
}

//----------------------------------------------------------------------------------------------

class CLstmOperator::CParseState final {
public:
	using TConstBlobPtr = const CPtr<const CDnnBlob>;
	using TConstTensorPtr = const CPtr<const CUserTensor>;
	using TLstmPtrs = const CPtr<CLstmLayer>[2];

	CParseState( const CString& direction, int hiddenSize, CDnn& dnn, const CLstmOperator& owner );

	void ReorderWeights( TConstBlobPtr weights, TConstBlobPtr recurWeights );
	void ReorderBias( TConstBlobPtr bias );
	void CreateLstms( CPtr<CLstmLayer> lstmLayers[2], TConstTensorPtr input, const CString& nameLstm ) const;
	void ConnectInitial( TLstmPtrs lstmLayers, TConstTensorPtr init, CString nameSplit, int inputNumber ) const;
	CBaseLayer* GetOutputLayer( TLstmPtrs lstmLayers, const CString& nameLstm ) const;

private:
	// Converts lstm weights from onnx format to NeoML, because gate weights in onnx and NeoML are ordered differently
	CPtr<CDnnBlob> reorderGates( TConstBlobPtr weights, TBlobDim dim ) const;
	void reorderWeights( TConstBlobPtr weights, CObjectArray<CDnnBlob>& weightsSplitted );
	void reorderBias( TConstBlobPtr bias, CObjectArray<CDnnBlob>& biasSplitted, CBlobDesc biasDesc, int offset );

	CDnn& dnn;
	IMathEngine& mathEngine;
	const CLstmOperator& owner;

	const CString& direction;
	const int hiddenSize;

	const bool bidirectional;
	const int numDirections;

	CObjectArray<CDnnBlob> weightsSplitted{};
	CObjectArray<CDnnBlob> recursiveWeightsSplitted{};

	CObjectArray<CDnnBlob> weightBiasSplitted{};
	CObjectArray<CDnnBlob> recursBiasSplitted{};
};

CLstmOperator::CParseState::CParseState( const CString& direction, int hiddenSize, CDnn& dnn, const CLstmOperator& owner ) :
	dnn( dnn ),
	mathEngine( dnn.GetMathEngine() ),
	owner( owner ),
	direction( direction ),
	hiddenSize( hiddenSize ),
	bidirectional( direction == "bidirectional" ),
	numDirections( bidirectional ? 2 : 1 )
{}

void CLstmOperator::CParseState::ReorderWeights( TConstBlobPtr weights, TConstBlobPtr recurWeights )
{
	reorderWeights( weights, weightsSplitted );
	reorderWeights( recurWeights, recursiveWeightsSplitted );
}

void CLstmOperator::CParseState::ReorderBias( TConstBlobPtr bias )
{
	if( bias == nullptr ) {
		return;
	}
	CBlobDesc biasDesc( CT_Float );
	biasDesc.SetDimSize( BD_Channels, 4 * hiddenSize );
	for( int dir = 0; dir < numDirections; ++dir ) {
		reorderBias( bias, weightBiasSplitted, biasDesc, dir * numDirections );
		reorderBias( bias, recursBiasSplitted, biasDesc, dir * numDirections + 1 );
	}
}

void CLstmOperator::CParseState::CreateLstms( CPtr<CLstmLayer> lstmLayers[2], TConstTensorPtr input, const CString& nameLstm ) const
{
	for( int dir = 0; dir < numDirections; ++dir ) {
		const bool reversed = ( direction == "backward" || dir != 0 );

		lstmLayers[dir] = new CLstmLayer( mathEngine );
		lstmLayers[dir]->SetName( nameLstm + ( reversed ? "_Reversed" : "" ) );
		lstmLayers[dir]->SetHiddenSize( hiddenSize );
		lstmLayers[dir]->Connect( /*inputNumber*/0, *input->Layer(), input->OutputIndex() );
		lstmLayers[dir]->SetReverseSequence( reversed );

		lstmLayers[dir]->SetInputWeightsData( weightsSplitted[dir] );
		lstmLayers[dir]->SetRecurWeightsData( recursiveWeightsSplitted[dir] );
		if( dir < weightBiasSplitted.Size() ) {
			lstmLayers[dir]->SetInputFreeTermData( weightBiasSplitted[dir] );
			lstmLayers[dir]->SetRecurFreeTermData( recursBiasSplitted[dir] );
		}
		dnn.AddLayer( *lstmLayers[dir] );
	}
}

void CLstmOperator::CParseState::ConnectInitial( TLstmPtrs lstmLayers, TConstTensorPtr init, CString nameSplit, int inputNumber ) const
{
	if( bidirectional ) {
		auto* splitLayer = new CSplitBatchLengthLayer( mathEngine );
		splitLayer->SetName( nameSplit );
		splitLayer->Connect( /*inputNumber*/0, *init->Layer(), init->OutputIndex() );
		splitLayer->SetOutputCounts2( 1 );
		dnn.AddLayer( *splitLayer );

		lstmLayers[0]->Connect( inputNumber, *splitLayer, /*outputNumber*/0 );
		lstmLayers[1]->Connect( inputNumber, *splitLayer, /*outputNumber*/1 );
	} else {
		lstmLayers[0]->Connect( inputNumber, *init->Layer(), init->OutputIndex() );
	}
}

CBaseLayer* CLstmOperator::CParseState::GetOutputLayer( TLstmPtrs lstmLayers, const CString& nameLstm ) const
{
	if( bidirectional ) {
		auto* concatLayer = new CConcatListSizeLayer( mathEngine );
		concatLayer->SetName( nameLstm + "_Concat" );
		concatLayer->Connect( /*inputNumber*/0, *( lstmLayers[0] ), /*outputNumber*/0 );
		concatLayer->Connect( /*inputNumber*/1, *( lstmLayers[1] ), /*outputNumber*/0 );
		dnn.AddLayer( *concatLayer );

		return concatLayer;
	} else {
		return lstmLayers[0];
	}
}

CPtr<CDnnBlob> CLstmOperator::CParseState::reorderGates( TConstBlobPtr blob, TBlobDim dim ) const
{
	const int gatesIndexNum = 4;
	const int gatesNumerate[gatesIndexNum]{ /*Main gate*/3, /*Forget gate*/2, /*Input gate*/0, /*Reset gate*/1 };

	CBlobDesc gateWeightDesc = blob->GetDesc();
	gateWeightDesc.SetDimSize( dim, gateWeightDesc.DimSize( dim ) / gatesIndexNum );

	CObjectArray<CDnnBlob> onnxGateWeight{};
	for( int gateIndex = 0; gateIndex < gatesIndexNum; ++gateIndex ) {
		onnxGateWeight.Add( CDnnBlob::CreateBlob( mathEngine, gateWeightDesc ) );
	}
	CDnnBlob::SplitByDim( mathEngine, dim, blob, onnxGateWeight );

	CObjectArray<CDnnBlob> neoMLGateWeight{};
	for( int gateIndex = 0; gateIndex < gatesIndexNum; ++gateIndex ) {
		neoMLGateWeight.Add( onnxGateWeight[gatesNumerate[gateIndex]] );
	}

	CPtr<CDnnBlob> result = blob->GetClone();
	CDnnBlob::MergeByDim( mathEngine, dim, neoMLGateWeight, result );
	return result;
}

void CLstmOperator::CParseState::reorderWeights( TConstBlobPtr weights, CObjectArray<CDnnBlob>& weightsSplitted )
{
	CheckOnnxProtocol( weights->GetDesc().BatchLength() == numDirections, "invalid number directions for weights", owner );
	if( bidirectional ) {
		CBlobDesc wDesc = weights->GetDesc();
		wDesc.SetDimSize( BD_BatchLength, wDesc.BatchLength() / numDirections );

		for( int dir = 0; dir < numDirections; ++dir ) {
			weightsSplitted.Add( CDnnBlob::CreateBlob( mathEngine, wDesc ) );
		}
		CDnnBlob::SplitByDim( mathEngine, BD_BatchLength, weights, weightsSplitted );
	} else {
		weightsSplitted.Add( weights->GetCopy() );
	}

	CBlobDesc blobDesc( CT_Float );
	blobDesc.SetDimSize( BD_BatchWidth, 4 * hiddenSize );
	for( int dir = 0; dir < numDirections; ++dir ) {
		blobDesc.SetDimSize( BD_Channels, weightsSplitted[dir]->GetDataSize() / blobDesc.BatchWidth() );
		weightsSplitted[dir]->ReinterpretDimensions( blobDesc );
		weightsSplitted[dir] = reorderGates( weightsSplitted[dir], BD_BatchWidth );
	}
}

void CLstmOperator::CParseState::reorderBias( TConstBlobPtr bias, CObjectArray<CDnnBlob>& biasSplitted, CBlobDesc biasDesc, int offset )
{
	CheckOnnxProtocol( bias->GetDesc().BatchWidth() == numDirections, "invalid number directions for bias", owner );
	const int biasChannels = biasDesc.Channels();
	biasSplitted.Add( CDnnBlob::CreateBlob( mathEngine, biasDesc ) );
	mathEngine.VectorCopy( biasSplitted.Last()->GetData(), bias->GetData() + offset * biasChannels, biasChannels );
	biasSplitted.Last() = reorderGates( biasSplitted.Last(), BD_Channels );
}

//----------------------------------------------------------------------------------------------

void CLstmOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoShapeInputs( inputs );
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	// NeoML doesn't support sequence lengths
	CheckNeoOnnxSupport( InputCount() <= 4 || inputs[4] == nullptr, "sequence lengths", *this );
	// NeoML doesn't support peepholes
	CheckNeoOnnxSupport( InputCount() <= 7 || inputs[7] == nullptr, "peepholes", *this );

	int layoutAttr = 0;
	GetAttribute( "layout", layoutAttr );
	// ONNX: The shape format of inputs X, initial_h, initial_c and outputs Y, Y_h, Y_c.
	const CTensorLayout neomlLayout = ( layoutAttr != 0 )
		? CTensorLayout{ BD_BatchWidth, BD_BatchLength, BD_Channels }
		: CTensorLayout{ BD_BatchLength, BD_BatchWidth, BD_Channels };
	CParseState::TConstTensorPtr inputData = AsUserTensor( *ConvertTensor( *inputs[0], neomlLayout ), Name() + "_Source", dnn );

	const CTensorLayout dataLayout{ BD_BatchLength, BD_BatchWidth, BD_Channels };
	CheckNeoOnnxSupport( inputs[1] != nullptr && inputs[1]->Type() == TTensorType::Data,
		"User-provided weight", *this );
	CParseState::TConstBlobPtr weights = dynamic_cast<const CDataTensor*>( ConvertTensor( *inputs[1], dataLayout ).Ptr() )->Data()->GetCopy();

	CheckNeoOnnxSupport( inputs[2] != nullptr && inputs[2]->Type() == TTensorType::Data,
		"User-provided recurrent weight", *this );
	CParseState::TConstBlobPtr recurWeights = dynamic_cast<const CDataTensor*>( ConvertTensor( *inputs[2], dataLayout ).Ptr() )->Data()->GetCopy();

	CPtr<const CDnnBlob> bias = nullptr;
	if( InputCount() > 3 && inputs[3] != nullptr ) {
		const CTensorLayout biasLayout{ BD_BatchWidth, BD_Channels };
		CheckNeoOnnxSupport( inputs[3]->Type() == TTensorType::Data, "User-provided bias", *this );
		bias = dynamic_cast<const CDataTensor*>( ConvertTensor( *inputs[3], biasLayout ).Ptr() )->Data()->GetCopy();
	}

	CParseState ps( direction, hiddenSize, dnn, *this );
	ps.ReorderWeights( weights, recurWeights );
	ps.ReorderBias( bias );

	CPtr<CLstmLayer> lstmLayers[2]{}; // lstmLayer("forward" or "backward") [+ reversedLstmLayer, if bidirectional]
	ps.CreateLstms( lstmLayers, inputData, Name() );

	if( inputs.Size() > 5 && inputs[5] != nullptr ) {
		CParseState::TConstTensorPtr initH = AsUserTensor( *ConvertTensor( *inputs[5], neomlLayout ), Name() + "_InitH", dnn );
		ps.ConnectInitial( lstmLayers, initH, Name() + "_SplitH", /*inputNumber*/2 );
	}

	if( inputs.Size() > 6 && inputs[6] != nullptr ) {
		CParseState::TConstTensorPtr initC = AsUserTensor( *ConvertTensor( *inputs[6], neomlLayout ), Name() + "_InitC", dnn );
		ps.ConnectInitial( lstmLayers, initC, Name() + "_SplitC", /*inputNumber*/1 );
	}

	CBaseLayer* outputLayer = ps.GetOutputLayer( lstmLayers, Name() );
	const CTensorLayout outputLayout = ( layoutAttr != 0 )
		? CTensorLayout{ BD_BatchWidth, BD_BatchLength, BD_ListSize, BD_Channels }
		: CTensorLayout{ BD_BatchLength, BD_ListSize, BD_BatchWidth, BD_Channels };
	outputs.Add( new CUserTensor( outputLayout, CLayerOutput( outputLayer, /*outputIndex*/0 ) ) );
	outputs.Add( /*Tensor*/nullptr, OutputCount() - 1 ); // internal state
}

} // namespace NeoOnnx
