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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/AttentionLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/SoftmaxLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/TransposeLayer.h>
#include <NeoML/Dnn/Layers/ConcatLayer.h>
#include <NeoML/Dnn/Layers/SplitLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoML/Dnn/Layers/SubSequenceLayer.h>

namespace NeoML {

CAttentionWeightedSumLayer::CAttentionWeightedSumLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnAttentionWeightedSumLayer", false )
{
}

static const int AttentionWeightedSumLayerVersion = 2000;

void CAttentionWeightedSumLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( AttentionWeightedSumLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

void CAttentionWeightedSumLayer::Reshape()
{
	CheckInputs();

	CheckArchitecture( inputDescs.Size() == 2, GetName(), "Weighted sum layer must have 2 inputs (objects, coeffs)" );
	CheckArchitecture( inputDescs[0].BatchWidth() == inputDescs[1].BatchWidth(), GetName(), "Batch width mismatch" );
	CheckArchitecture( inputDescs[0].ListSize() == inputDescs[1].ListSize(), GetName(), "List size mismatch" );
	CheckArchitecture( inputDescs[1].BatchLength() == 1 || GetDnn()->IsRecurrentMode(), GetName(),
		"Layer must be used inside of recurrent decoder or inputDescs[1].BatchLength must be equal to 1" );

	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize(BD_BatchLength, inputDescs[1].BatchLength());
	outputDescs[0].SetDimSize(BD_ListSize, 1);
}

void CAttentionWeightedSumLayer::RunOnce()
{
	MathEngine().MultiplyMatrixByMatrix(inputBlobs[0]->GetBatchWidth(), inputBlobs[1]->GetData(), 1,  inputBlobs[1]->GetListSize(),
		inputBlobs[0]->GetData(), inputBlobs[0]->GetObjectSize(), outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize());
}

void CAttentionWeightedSumLayer::BackwardOnce()
{
	MathEngine().MultiplyMatrixByMatrix(inputBlobs[0]->GetBatchWidth(), inputBlobs[1]->GetData(),
		inputBlobs[1]->GetListSize(), 1, outputDiffBlobs[0]->GetData(),
		outputDiffBlobs[0]->GetObjectSize(), inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize());
	MathEngine().MultiplyMatrixByMatrix(inputBlobs[0]->GetBatchWidth(), inputBlobs[0]->GetData(),
		inputBlobs[0]->GetListSize(), inputBlobs[0]->GetObjectSize(),
		outputDiffBlobs[0]->GetData(), 1, inputDiffBlobs[1]->GetData(), inputDiffBlobs[1]->GetDataSize());
}

//---------------------------------------------------------------------------------------------------------------------

CAttentionDotProductLayer::CAttentionDotProductLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnAttentionDotProductLayer", false )
{
}

static const int AttentionDotProductLayerVersion = 2000;

void CAttentionDotProductLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( AttentionDotProductLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

void CAttentionDotProductLayer::Reshape()
{
	CheckInputs();

	CheckArchitecture( inputDescs.Size() == 2, GetName(), "Weighted sum layer must have 2 inputs (objects, coeffs)" );
	CheckArchitecture( inputDescs[0].BatchWidth() == inputDescs[1].BatchWidth(), GetName(), "Batch width mismatch" );
	CheckArchitecture( inputDescs[0].ObjectSize() == inputDescs[1].ObjectSize(), GetName(), "Object size mismatch" );
	CheckArchitecture( inputDescs[1].BatchLength() == 1 || GetDnn()->IsRecurrentMode(), GetName(),
		"Layer must be used inside of recurrent decoder or inputDescs[1].BatchLength must be equal to 1" );

	outputDescs[0] = inputDescs[1];
	outputDescs[0].SetDimSize( BD_ListSize, inputDescs[0].ListSize() );
	outputDescs[0].SetDimSize( BD_Height, 1 );
	outputDescs[0].SetDimSize( BD_Width, 1 );
	outputDescs[0].SetDimSize( BD_Depth, 1 );
	outputDescs[0].SetDimSize( BD_Channels, 1 );
}

void CAttentionDotProductLayer::RunOnce()
{
	MathEngine().MultiplyMatrixByMatrix(inputBlobs[0]->GetBatchWidth(), inputBlobs[0]->GetData(), inputBlobs[0]->GetListSize(),
		inputBlobs[0]->GetObjectSize(),	inputBlobs[1]->GetData(), 1, outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize());
}

void CAttentionDotProductLayer::BackwardOnce()
{
	MathEngine().MultiplyMatrixByMatrix(inputBlobs[0]->GetBatchWidth(), outputDiffBlobs[0]->GetData(), outputDiffBlobs[0]->GetListSize(), 1,
		inputBlobs[1]->GetData(), inputBlobs[1]->GetObjectSize(), inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize());
	MathEngine().MultiplyMatrixByMatrix(inputBlobs[0]->GetBatchWidth(), outputDiffBlobs[0]->GetData(), 1, outputDiffBlobs[0]->GetListSize(),
		inputBlobs[0]->GetData(), inputBlobs[0]->GetObjectSize(), inputDiffBlobs[1]->GetData(), inputDiffBlobs[1]->GetDataSize());
}

//---------------------------------------------------------------------------------------------------------------------

CAttentionSumLayer::CAttentionSumLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnAttentionSumLayer", false )
{
}

static const int AttentionSumLayer = 2000;

void CAttentionSumLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( AttentionSumLayer );
	CBaseLayer::Serialize( archive );
}

void CAttentionSumLayer::Reshape()
{
	CheckInputs();

	CheckArchitecture( inputDescs.Size() == 2, GetName(), "Weighted sum layer must have 2 inputs (objects, coeffs)" );
	CheckArchitecture( inputDescs[0].BatchWidth() == inputDescs[1].BatchWidth(), GetName(), "Batch width mismatch" );
	CheckArchitecture( inputDescs[0].ObjectSize() == inputDescs[1].ObjectSize(), GetName(), "Object size mismatch" );
	CheckArchitecture( inputDescs[1].BatchLength() == 1 || GetDnn()->IsRecurrentMode(), GetName(),
		"Layer must be used inside of recurrent decoder or inputDescs[1].BatchLength must be equal to 1" );

	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize(BD_BatchLength, inputDescs[1].BatchLength());
}

void CAttentionSumLayer::RunOnce()
{
	MathEngine().AddVectorToMatrixRows(inputBlobs[0]->GetBatchWidth(), inputBlobs[0]->GetData(), outputBlobs[0]->GetData(), inputBlobs[0]->GetListSize(),
		inputBlobs[0]->GetObjectSize(), inputBlobs[1]->GetData());
}

void CAttentionSumLayer::BackwardOnce()
{
	inputDiffBlobs[0]->CopyFrom(outputDiffBlobs[0]);
	MathEngine().SumMatrixRows(inputDiffBlobs[1]->GetBatchWidth(), inputDiffBlobs[1]->GetData(), outputDiffBlobs[0]->GetData(), 
		outputDiffBlobs[0]->GetListSize(), outputDiffBlobs[0]->GetObjectSize());
}

//---------------------------------------------------------------------------------------------------------------------

CAttentionDecoderLayer::CAttentionDecoderLayer( IMathEngine& mathEngine ) :
	CCompositeLayer( mathEngine, "CCnnAttentionDecoderLayer" ),
	score(AS_Additive)
{
	buildLayer();
}

void CAttentionDecoderLayer::SetAttentionScore( TAttentionScore newScore )
{
	if( score == newScore ) {
		return;
	}
	score = newScore;
	buildLayer();
}

int CAttentionDecoderLayer::GetOutputSequenceLen() const
{
	return recurrentLayer->GetRepeatCount();
}

void CAttentionDecoderLayer::SetOutputSequenceLen(int outSeqLen)
{
	if( outSeqLen != recurrentLayer->GetRepeatCount() ) {
		ForceReshape();
	}
	recurrentLayer->SetRepeatCount(outSeqLen);
}

int CAttentionDecoderLayer::GetOutputObjectSize() const
{
	return recurrentLayer->GetOutputObjectSize();
}

void CAttentionDecoderLayer::SetOutputObjectSize(int outObjectSize)
{
	recurrentLayer->SetOutputObjectSize(outObjectSize);
}

int CAttentionDecoderLayer::GetHiddenLayerSize() const
{
	return hiddenLayer != 0 ? hiddenLayer->GetNumberOfElements() : 0;
}

void CAttentionDecoderLayer::SetHiddenLayerSize(int size)
{
	hiddenLayer->SetNumberOfElements(size);
	initLayer->SetNumberOfElements(size);
	recurrentLayer->SetHiddenLayerSize(size);
}

static const int AttentionDecoderLayerVersion = 2000;

void CAttentionDecoderLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( AttentionDecoderLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CCompositeLayer::Serialize( archive );
	
	archive.SerializeEnum(score);
	if( archive.IsLoading() ) {
		hiddenLayer = CheckCast<CFullyConnectedLayer>(GetLayer(hiddenLayer->GetName()));
		initLayer = CheckCast<CFullyConnectedLayer>(GetLayer(initLayer->GetName()));
		recurrentLayer = CheckCast<CAttentionRecurrentLayer>(GetLayer(recurrentLayer->GetName()));
	}
}

void CAttentionDecoderLayer::buildLayer()
{
	DeleteAllLayers();

	auto transposeLayer = FINE_DEBUG_NEW CTransposeLayer( MathEngine() );
	AddLayer(*transposeLayer);
	SetInputMapping(I_InputSequence, *transposeLayer);
	transposeLayer->SetTransposedDimensions(BD_BatchLength, BD_ListSize);
	hiddenLayer = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	AddLayer(*hiddenLayer);
	hiddenLayer->Connect(*transposeLayer);

	recurrentLayer = FINE_DEBUG_NEW CAttentionRecurrentLayer( MathEngine() );
	AddLayer(*recurrentLayer);
	recurrentLayer->Connect(CAttentionRecurrentLayer::I_InputList, *transposeLayer);
	recurrentLayer->Connect(CAttentionRecurrentLayer::I_ProcessedInputList, *hiddenLayer);
	recurrentLayer->SetAttentionScore(score);

	// The input sequence for "teacher forcing" mode
	SetInputMapping(I_OutputInitializer, *recurrentLayer, CAttentionRecurrentLayer::I_OutputInitializer);

	// Calculate the initial state of the decoder
	auto getFirstItem = FINE_DEBUG_NEW CSubSequenceLayer( MathEngine() );
	AddLayer(*getFirstItem);
	SetInputMapping(I_InputSequence, *getFirstItem);
	getFirstItem->SetStartPos(0);
	getFirstItem->SetLength(1);

	initLayer = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	CString initLayerName = initLayer->GetName() + CString( ".init" );
	initLayer->SetName( initLayerName );
	initLayer->SetZeroFreeTerm( true );
	AddLayer(*initLayer);
	initLayer->Connect(*getFirstItem);

	auto initTanh = FINE_DEBUG_NEW CTanhLayer( MathEngine() );
	CString initTanhName = initTanh->GetName() + CString( ".init" );
	initTanh->SetName( initTanhName );
	AddLayer( *initTanh );
	initTanh->Connect( *initLayer );

	recurrentLayer->Connect(CAttentionRecurrentLayer::I_InitialState, *initTanh);

	SetOutputMapping(*recurrentLayer);
}

//---------------------------------------------------------------------------------------------------------------------

const CString CAttentionRecurrentLayer::hiddenLayerName = "hiddenFullyConnectedLayer";

CAttentionRecurrentLayer::CAttentionRecurrentLayer( IMathEngine& mathEngine ) :
	CRecurrentLayer( mathEngine, "CCnnAttentionRecurrentLayer" ),
	score( AS_Additive )
{
	buildLayer();
}

void CAttentionRecurrentLayer::SetAttentionScore( TAttentionScore newScore )
{
	if( score == newScore ) {
		return;
	}
	score = newScore;
	buildLayer();
}

int CAttentionRecurrentLayer::GetOutputObjectSize() const
{
	return outputLayer->GetNumberOfElements();
}

void CAttentionRecurrentLayer::SetOutputObjectSize(int outObjectSize)
{
	mainBackLink->SetDimSize(BD_Channels, outObjectSize);
	outputLayer->SetNumberOfElements(outObjectSize);
}

void CAttentionRecurrentLayer::SetHiddenLayerSize(int size)
{
	if(hiddenLayer != 0) {
		hiddenLayer->SetNumberOfElements(size);
	}
	stateBackLink->SetDimSize(BD_Channels, size);
	mainLayer->SetNumberOfElements(size);
	gateLayer->SetNumberOfElements(size * 2);
	splitLayer->SetOutputCounts2(size);
}

static const int AttentionRecurrentLayerVersion = 2000;

void CAttentionRecurrentLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( AttentionRecurrentLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CRecurrentLayer::Serialize( archive );

	archive.SerializeEnum(score);
	if( archive.IsLoading() ) {
		if( HasLayer(hiddenLayerName) ) {
			hiddenLayer = CheckCast<CFullyConnectedLayer>(GetLayer(hiddenLayerName));
		} else {
			hiddenLayer = 0;
		}
		attentionLayer = CheckCast<CAttentionLayer>(GetLayer(attentionLayer->GetName()));
		stateBackLink = CheckCast<CBackLinkLayer>(GetLayer(stateBackLink->GetName()));
		mainBackLink = CheckCast<CBackLinkLayer>(GetLayer(mainBackLink->GetName()));

		mainLayer = CheckCast<CFullyConnectedLayer>(GetLayer(mainLayer->GetName()));
		gateLayer = CheckCast<CFullyConnectedLayer>(GetLayer(gateLayer->GetName()));
		outputLayer = CheckCast<CFullyConnectedLayer>(GetLayer(outputLayer->GetName()));
		splitLayer = CheckCast<CSplitChannelsLayer>(GetLayer(splitLayer->GetName()));
	}
}

void CAttentionRecurrentLayer::buildLayer()
{
	DeleteAllLayersAndBackLinks();

	stateBackLink = FINE_DEBUG_NEW CBackLinkLayer( MathEngine() );
	CString stateBackLinkName = stateBackLink->GetName() + CString( ".state" );
	stateBackLink->SetName( stateBackLinkName );
	AddBackLink(*stateBackLink);
	if(score == AS_Additive) {
		hiddenLayer = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
		hiddenLayer->SetName(hiddenLayerName);
		hiddenLayer->SetZeroFreeTerm( true );
		AddLayer(*hiddenLayer);
		hiddenLayer->Connect(*stateBackLink);
	} else {
		hiddenLayer = 0;
	}
	attentionLayer = FINE_DEBUG_NEW CAttentionLayer( MathEngine() );
	AddLayer(*attentionLayer);
	// The transposed input sequence
	SetInputMapping(I_InputList, *attentionLayer, CAttentionLayer::I_InputList);
	// The transposed input sequence after a fully-connected layer
	SetInputMapping(I_ProcessedInputList, *attentionLayer, CAttentionLayer::I_ProcessedInputList);
	attentionLayer->SetAttentionScore(score);

	if(hiddenLayer != 0) { // an output sequence element
		attentionLayer->Connect(CAttentionLayer::I_DecoderState, *hiddenLayer);
	} else {
		attentionLayer->Connect(CAttentionLayer::I_DecoderState, *stateBackLink);
	}

	mainBackLink = FINE_DEBUG_NEW CBackLinkLayer( MathEngine() );
	CString mainBackLinkName = mainBackLink->GetName() + CString(".main");
	mainBackLink->SetName(mainBackLinkName);
	AddBackLink(*mainBackLink);

	auto gateConcat = FINE_DEBUG_NEW CConcatObjectLayer( MathEngine() );
	CString gateConcatName = gateConcat->GetName() + CString(".gates");
	gateConcat->SetName(gateConcatName);
	gateConcat->Connect(0, *attentionLayer);
	gateConcat->Connect(1, *mainBackLink);
	gateConcat->Connect(2, *stateBackLink);
	AddLayer(*gateConcat);

	gateLayer = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	CString gateLayerName = gateLayer->GetName() + CString(".gates");
	gateLayer->SetName(gateLayerName);
	gateLayer->Connect(*gateConcat);
	AddLayer(*gateLayer);

	splitLayer = FINE_DEBUG_NEW CSplitChannelsLayer( MathEngine() );
	splitLayer->SetOutputCounts2(0);
	splitLayer->Connect(*gateLayer);
	AddLayer(*splitLayer);

	auto resetSigmoid = FINE_DEBUG_NEW CSigmoidLayer( MathEngine() );
	CString resetSigmoidName = resetSigmoid->GetName() + CString(".reset");
	resetSigmoid->SetName(resetSigmoidName);
	resetSigmoid->Connect(0, *splitLayer, 0);
	AddLayer(*resetSigmoid);

	auto resetGate = FINE_DEBUG_NEW CEltwiseMulLayer( MathEngine() );
	CString resetGateName = resetGate->GetName() + CString(".reset");
	resetGate->SetName(resetGateName);
	resetGate->Connect(0, *resetSigmoid);
	resetGate->Connect(1, *stateBackLink);
	AddLayer(*resetGate);

	auto mainConcat = FINE_DEBUG_NEW CConcatChannelsLayer( MathEngine() );
	CString mainConcatName = mainConcat->GetName() + CString(".main");
	mainConcat->SetName(mainConcatName);
	mainConcat->Connect(0, *attentionLayer);
	mainConcat->Connect(1, *mainBackLink);
	mainConcat->Connect(2, *resetGate);
	AddLayer(*mainConcat);

	mainLayer = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	CString mainLayerName = mainLayer->GetName() + CString(".main");
	mainLayer->SetName(mainLayerName);
	mainLayer->Connect(*mainConcat);
	AddLayer(*mainLayer);

	auto mainTanh = FINE_DEBUG_NEW CTanhLayer( MathEngine() );
	mainTanh->Connect(*mainLayer);
	AddLayer(*mainTanh);

	auto updateSigmoid = FINE_DEBUG_NEW CSigmoidLayer( MathEngine() );
	CString updateSigmoidName = updateSigmoid->GetName() + CString(".update");
	updateSigmoid->SetName(updateSigmoidName);
	updateSigmoid->Connect(0, *splitLayer, 1);
	AddLayer(*updateSigmoid);

	auto updateGate = FINE_DEBUG_NEW CEltwiseMulLayer( MathEngine() );
	CString updateGateName = updateGate->GetName() + CString(".update");
	updateGate->SetName(updateGateName);
	updateGate->Connect(0, *updateSigmoid);
	updateGate->Connect(1, *mainTanh);
	AddLayer(*updateGate);

	auto forgetGate = FINE_DEBUG_NEW CEltwiseNegMulLayer( MathEngine() );
	CString forgetGateName = forgetGate->GetName() + CString(".forget");
	forgetGate->SetName(forgetGateName);
	forgetGate->Connect(0, *updateSigmoid);
	forgetGate->Connect(1, *stateBackLink);
	AddLayer(*forgetGate);

	auto newState = FINE_DEBUG_NEW CEltwiseSumLayer( MathEngine() );
	newState->Connect(0, *updateGate);
	newState->Connect(1, *forgetGate);
	AddLayer(*newState);

	// Connect the back link
	stateBackLink->Connect(*newState);

	// The initial decoder state
	SetInputMapping( I_InitialState, *stateBackLink, 1 );

	outputLayer = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	CString outputLayerName = outputLayer->GetName() + CString(".output");
	outputLayer->SetName(outputLayerName);
	AddLayer(*outputLayer);
	outputLayer->Connect(*gateConcat);
	auto outputSoftmax = FINE_DEBUG_NEW CSoftmaxLayer( MathEngine() );
	outputSoftmax->Connect(*outputLayer);
	AddLayer(*outputSoftmax);
	
	mainBackLink->Connect(*outputSoftmax);
	// a) initialize the output sequence on a forward pass
	// b) the output sequence when learning in "teacher forcing" mode
	SetInputMapping( I_OutputInitializer, *mainBackLink, 1 );

	SetOutputMapping(*outputSoftmax);
}

//---------------------------------------------------------------------------------------------------------------------

CAttentionLayer::CAttentionLayer( IMathEngine& mathEngine ) :
	CCompositeLayer( mathEngine, "CCnnAttentionLayer" ),
	score(AS_Additive),
	tanhFc(0)
{
	buildLayer();
}

void CAttentionLayer::SetAttentionScore( TAttentionScore newScore )
{
	if( score == newScore ) {
		return;
	}
	score = newScore;
	buildLayer();
}

CPtr<CDnnBlob> CAttentionLayer::GetFcWeightsData() const
{
	NeoAssert( tanhFc != 0 );
	return tanhFc->GetWeightsData();
}

void CAttentionLayer::SetFcWeightsData( const CPtr<CDnnBlob>& newWeights )
{
	NeoAssert( tanhFc != 0 );
	tanhFc->SetWeightsData( newWeights );
}

CPtr<CDnnBlob> CAttentionLayer::GetFcFreeTermData() const
{
	NeoAssert( tanhFc != 0 );
	return tanhFc->GetFreeTermData();
}

void CAttentionLayer::SetFcFreeTermData( const CPtr<CDnnBlob>& newFreeTerms )
{
	NeoAssert( tanhFc != 0 );
	tanhFc->SetFreeTermData( newFreeTerms );
}

void CAttentionLayer::buildLayer()
{
	DeleteAllLayers();

	CPtr<CBaseLayer> weights;
	if(score == AS_Additive) {
		auto seqSumLayer = FINE_DEBUG_NEW CAttentionSumLayer( MathEngine() );
		AddLayer(*seqSumLayer);
		SetInputMapping(I_ProcessedInputList, *seqSumLayer, 0); // the input sequence after a fully-connected layer
		SetInputMapping(I_DecoderState, *seqSumLayer, 1); // the output sequence after a fully-connected layer
		auto tanhLayer = FINE_DEBUG_NEW CTanhLayer( MathEngine() );
		tanhLayer->Connect(*seqSumLayer);
		AddLayer(*tanhLayer);
		tanhFc = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
		tanhFc->SetNumberOfElements(1);
		tanhFc->SetZeroFreeTerm( true );
		AddLayer(*tanhFc);
		tanhFc->Connect(*tanhLayer);
		weights = tanhFc;
	} else {
		auto seqDotLayer = FINE_DEBUG_NEW CAttentionDotProductLayer( MathEngine() );
		AddLayer(*seqDotLayer);
		SetInputMapping(I_ProcessedInputList, *seqDotLayer, 0); // the input sequence after a fully-connected layer
		SetInputMapping(I_DecoderState, *seqDotLayer, 1); // the output sequence
		weights = seqDotLayer;
	}
	auto softmax = FINE_DEBUG_NEW CSoftmaxLayer( MathEngine() );
	softmax->SetNormalizationArea( CSoftmaxLayer::NA_ListSize );
	AddLayer(*softmax);
	softmax->Connect(*weights);
	
	auto outputCalculationLayer = FINE_DEBUG_NEW CAttentionWeightedSumLayer( MathEngine() );
	AddLayer( *outputCalculationLayer );
	SetInputMapping(I_InputList, *outputCalculationLayer);  // the input sequence
	outputCalculationLayer->Connect(1, *softmax ); // the input sequence elements' weights

	SetOutputMapping( *outputCalculationLayer );
}

static const CString tanhFcLayerName = "CCnnFullyConnectedLayer";

static const int AttentionLayerVersion = 2000;

void CAttentionLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( AttentionLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CCompositeLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		archive << static_cast<int>(score);
	} else if( archive.IsLoading() ) {
		int intBuffer;
		archive >> intBuffer;
		score = static_cast<TAttentionScore>(intBuffer);
		if( score == AS_Additive ) {
			NeoAssert( HasLayer( tanhFcLayerName ) );
			tanhFc = CheckCast<CFullyConnectedLayer>( GetLayer( tanhFcLayerName ) );
		} else {
			tanhFc = 0;
		}
	} else {
		NeoAssert(false);
	}
}

} // namespace NeoML
