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

#include <NeoML/Dnn/Layers/LstmLayer.h>
#include <NeoML/Dnn/Layers/ConcatLayer.h>
#include <NeoML/Dnn/Layers/SplitLayer.h>

namespace NeoML {

// Names used in versions < 2001.
// There was a single dropout and a single fully-connected layer for both input and recurrent.
static const char* hiddenLayerName = "CCnnFullyConnectedLayer";
static const char* dropoutName = "Dropout";

// Names used in versions >= 2001.
static const char* inputDropoutName = "InputDropout";
static const char* recurDropoutName = "RecurDropout";
static const char* inputHiddenLayerName = "InputHidden";
static const char* recurHiddenLayerName = "RecurHidden";

CLstmLayer::CLstmLayer( IMathEngine& mathEngine ) :
	CRecurrentLayer( mathEngine, "CCnnLstmLayer" ),
	recurrentActivation( AF_Sigmoid ),
	isInCompatibilityMode( false )
{
	buildLayer(0);
}

// Builds the layer
void CLstmLayer::buildLayer(float dropout)
{
	// Initialize a back link
	if( mainBackLink == nullptr ) {
		mainBackLink = FINE_DEBUG_NEW CBackLinkLayer( MathEngine() );
		CString mainBackLinkName = mainBackLink->GetName() + CString(".main");
		mainBackLink->SetName(mainBackLinkName);
	}
	AddBackLink(*mainBackLink);

	if( stateBackLink == nullptr ) {
		stateBackLink = FINE_DEBUG_NEW CBackLinkLayer( MathEngine() );
		CString stateBackLinkName = stateBackLink->GetName() + CString(".state");
		stateBackLink->SetName(stateBackLinkName);
	}
	AddBackLink(*stateBackLink);

	if( dropout > 0 ) {
		inputDropoutLayer = FINE_DEBUG_NEW CDropoutLayer( MathEngine() );
		inputDropoutLayer->SetName(inputDropoutName);
		inputDropoutLayer->SetDropoutRate(dropout);
		SetInputMapping(*inputDropoutLayer);
		AddLayer(*inputDropoutLayer);

		recurDropoutLayer = FINE_DEBUG_NEW CDropoutLayer( MathEngine() );
		recurDropoutLayer->SetName(recurDropoutName);
		recurDropoutLayer->SetDropoutRate(dropout);
		recurDropoutLayer->Connect(*mainBackLink);
		AddLayer(*recurDropoutLayer);
	} else {
		inputDropoutLayer = nullptr;
		recurDropoutLayer = nullptr;
	}
	
	if( inputHiddenLayer == nullptr ) {
		inputHiddenLayer = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
		inputHiddenLayer->SetName( inputHiddenLayerName );
	}

	if( inputDropoutLayer == nullptr ) {
		SetInputMapping( *inputHiddenLayer );
	} else {
		inputHiddenLayer->Connect( *inputDropoutLayer );
	}
	AddLayer( *inputHiddenLayer );

	if( recurHiddenLayer == nullptr ) {
		recurHiddenLayer = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
		recurHiddenLayer->SetName( recurHiddenLayerName );
	}

	if( recurDropoutLayer == nullptr ) {
		recurHiddenLayer->Connect( *mainBackLink );
	} else {
		recurHiddenLayer->Connect( *recurDropoutLayer );
	}
	AddLayer( *recurHiddenLayer );

	CPtr<CEltwiseSumLayer> hiddenLayerSum = FINE_DEBUG_NEW CEltwiseSumLayer( MathEngine() );
	hiddenLayerSum->SetName( "HiddenLayerSum" );
	hiddenLayerSum->Connect( *inputHiddenLayer );
	hiddenLayerSum->Connect( 1, *recurHiddenLayer, 0 );
	AddLayer( *hiddenLayerSum );

	if( splitLayer == 0 ) {
		splitLayer = FINE_DEBUG_NEW CSplitChannelsLayer( MathEngine() );
		splitLayer->SetOutputCounts4(0, 0, 0);
	}
	splitLayer->Connect(*hiddenLayerSum);
	AddLayer(*splitLayer);

	CPtr<CTanhLayer> mainTanh = FINE_DEBUG_NEW CTanhLayer( MathEngine() );
	CString mainTanhName = mainTanh->GetName() + CString(".main");
	mainTanh->SetName(mainTanhName);
	mainTanh->Connect(0, *splitLayer, G_Main);
	AddLayer(*mainTanh);

	CPtr<CBaseLayer> inputSigmoid = CreateActivationLayer( MathEngine(), recurrentActivation );
	CString inputSigmoidName = inputSigmoid->GetName() + CString(".input");
	inputSigmoid->SetName(inputSigmoidName);
	inputSigmoid->Connect(0, *splitLayer, G_Input);
	AddLayer(*inputSigmoid);

	CPtr<CBaseLayer> forgetSigmoid = CreateActivationLayer( MathEngine(), recurrentActivation );
	CString forgetSigmoidName = forgetSigmoid->GetName() + CString(".forget");
	forgetSigmoid->SetName(forgetSigmoidName);
	forgetSigmoid->Connect(0, *splitLayer, G_Forget);
	AddLayer(*forgetSigmoid);

	CPtr<CBaseLayer> resetSigmoid = CreateActivationLayer( MathEngine(), recurrentActivation );
	CString resetSigmoidName = resetSigmoid->GetName() + CString(".reset");
	resetSigmoid->SetName(resetSigmoidName);
	resetSigmoid->Connect(0, *splitLayer, G_Reset);
	AddLayer(*resetSigmoid);

	CPtr<CEltwiseMulLayer> inputGate = FINE_DEBUG_NEW CEltwiseMulLayer( MathEngine() );
	CString inputGateName = inputGate->GetName() + CString(".input");
	inputGate->SetName(inputGateName);
	inputGate->Connect(0, *inputSigmoid);
	inputGate->Connect(1, *mainTanh);
	AddLayer(*inputGate);

	CPtr<CEltwiseMulLayer> forgetGate = FINE_DEBUG_NEW CEltwiseMulLayer( MathEngine() );
	CString forgetGateName = forgetGate->GetName() + CString(".forget");
	forgetGate->SetName(forgetGateName);
	forgetGate->Connect(0, *forgetSigmoid);
	forgetGate->Connect(1, *stateBackLink);
	AddLayer(*forgetGate);

	CPtr<CEltwiseSumLayer> newState = FINE_DEBUG_NEW CEltwiseSumLayer( MathEngine() );
	newState->Connect(0, *inputGate);
	newState->Connect(1, *forgetGate);
	AddLayer(*newState);

	outputTanh = FINE_DEBUG_NEW CTanhLayer( MathEngine() );
	CString outputTanhName = outputTanh->GetName() + CString(".output");
	outputTanh->SetName(outputTanhName);
	outputTanh->Connect(*newState);
	AddLayer(*outputTanh);

	resetGate = FINE_DEBUG_NEW CEltwiseMulLayer( MathEngine() );
	CString resetGateName = resetGate->GetName() + CString(".reset");
	resetGate->SetName(resetGateName);
	resetGate->Connect(0, *resetSigmoid);
	resetGate->Connect(1, *outputTanh);
	AddLayer(*resetGate);

	// Connect the back links
	mainBackLink->Connect(*resetGate);
	stateBackLink->Connect(*newState);

	// Initial state
	SetInputMapping( 1, *stateBackLink, 1 );

	// Initial history
	SetInputMapping( 2, *mainBackLink, 1 );

	// The output
	if( isInCompatibilityMode ) {
		SetOutputMapping( *outputTanh );
	} else {
		SetOutputMapping( *resetGate );
	}

	// Output the hidden state
	SetOutputMapping(1, *newState);
}

void CLstmLayer::SetDropoutRate(float newDropoutRate)
{
	if( ( newDropoutRate > 0 && inputDropoutLayer == 0 ) || ( newDropoutRate <= 0 && inputDropoutLayer != 0 ) ) {
		DeleteAllLayersAndBackLinks();
		buildLayer(newDropoutRate);
	} else if( inputDropoutLayer != 0 ) {
		inputDropoutLayer->SetDropoutRate(newDropoutRate);
		recurDropoutLayer->SetDropoutRate(newDropoutRate);
	}
}

void CLstmLayer::SetHiddenSize(int size)
{
	inputHiddenLayer->SetNumberOfElements(size * G_Count);
	recurHiddenLayer->SetNumberOfElements(size * G_Count);
	splitLayer->SetOutputCounts4(size, size, size);
	mainBackLink->SetDimSize(BD_Channels, size);
	stateBackLink->SetDimSize(BD_Channels, size);
}

void CLstmLayer::SetRecurrentActivation( TActivationFunction newActivation )
{
	if( recurrentActivation == newActivation ) {
		return;
	}

	recurrentActivation = newActivation;
	float dropoutRate = GetDropoutRate();
	DeleteAllLayersAndBackLinks();
	buildLayer( dropoutRate );
}

void CLstmLayer::SetCompatibilityMode( bool compatibilityMode )
{
	if( isInCompatibilityMode == compatibilityMode ) {
		return;
	}

	isInCompatibilityMode = compatibilityMode;
	if( isInCompatibilityMode ) {
		SetOutputMapping( *outputTanh );
	} else {
		SetOutputMapping( *resetGate );
	}

	ForceReshape();
}

static const int LstmLayerVersion = 2001;

void CLstmLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( LstmLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CRecurrentLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		int recurrentActivationInt = static_cast<TActivationFunction>( recurrentActivation );
		archive << recurrentActivationInt;
	} else if( archive.IsLoading() ) {
		int recurrentActivationInt = 0;
		archive >> recurrentActivationInt;
		recurrentActivation = static_cast<TActivationFunction>( recurrentActivationInt );

		if( version < 2001 ) {
			inputHiddenLayer = nullptr;
			recurHiddenLayer = nullptr;
		} else {
			inputHiddenLayer = CheckCast<CFullyConnectedLayer>( GetLayer( inputHiddenLayerName ) );
			recurHiddenLayer = CheckCast<CFullyConnectedLayer>( GetLayer( recurHiddenLayerName ) );
		}

		if( HasLayer( inputDropoutName ) ) {
			inputDropoutLayer = CheckCast<CDropoutLayer>( GetLayer( inputDropoutName ) );
			recurDropoutLayer = CheckCast<CDropoutLayer>( GetLayer( recurDropoutName ) );
		} else {
			inputDropoutLayer = nullptr;
			recurDropoutLayer = nullptr;
		}
		splitLayer = CheckCast<CSplitChannelsLayer>(GetLayer(splitLayer->GetName()));
		mainBackLink = CheckCast<CBackLinkLayer>(GetLayer(mainBackLink->GetName()));
		stateBackLink = CheckCast<CBackLinkLayer>(GetLayer(stateBackLink->GetName()));
		outputTanh = CheckCast<CTanhLayer>( GetLayer( outputTanh->GetName() ) );
		resetGate = CheckCast<CEltwiseMulLayer>( GetLayer( resetGate->GetName() ) );
		isInCompatibilityMode = GetOutputMapping( 0 ).InternalLayerName == outputTanh->GetName();

		if( version < 2001 ) {
			// Old version, when LSTM had 1 fully connected layer for both input and recurrent.

			// Getting old weights.
			CPtr<CDnnBlob> weights;
			CPtr<CDnnBlob> freeTerm;
			{
				CPtr<CFullyConnectedLayer> oldFc = CheckCast<CFullyConnectedLayer>( GetLayer( hiddenLayerName ) );
				weights = oldFc->GetWeightsData();
				freeTerm = oldFc->GetFreeTermData();
			}

			// Getting old dropout rate.
			float dropoutRate = 0.f;
			if( HasLayer( dropoutName ) ) {
				dropoutRate = CheckCast<CDropoutLayer>( GetLayer( dropoutName ) )->GetDropoutRate();
			}

			// Rebuilding layer.
			DeleteAllLayersAndBackLinks();
			buildLayer( dropoutRate );

			// Setting old weights.
			// Inside this method weight will be distributed between inputHiddenLayer and recurHiddenLayer.
			setWeightsData( weights );

			// Setting free terms.
			SetInputFreeTermData( freeTerm );
			// In order to reproduce old behavior, free terms must be added only once (in inputHiddenLayer).
			// Thats why recurHiddenLayer's free terms must be zero.
			SetRecurFreeTermData( nullptr );
		}
	} else {
		NeoAssert( false );
	}
}

void CLstmLayer::RunOnce() {
	if( !IsInCompatibilityMode() ) {
		fastLstm();
	} else {
		CRecurrentLayer::RunOnce();
	}
}

// Sets weights to input and reccurrent fully connected layers.
// Used for loading previous versions of LSTM, where single fully connected layer was used.
void CLstmLayer::setWeightsData( const CPtr<CDnnBlob>& newWeights )
{
	if( newWeights == nullptr ) {
		SetInputWeightsData( newWeights );
		SetRecurWeightsData( newWeights );
		return;
	}

	// In this case we need to split weights into two:
	// 1. inputSize x (4 * hiddenSize) for inputHiddenLayer.
	// 2. hiddenSize x (4 * hiddenSize) for recurHiddenLayer.
	NeoAssert( newWeights->GetObjectCount() > 0 );
	NeoAssert( newWeights->GetObjectCount() % 4 == 0 );

	const int newHiddenSize = newWeights->GetObjectCount() / 4;
	NeoAssert( newWeights->GetObjectSize() > newHiddenSize );
	CObjectArray<CDnnBlob> splitWeights;
	CBlobDesc weightDesc = newWeights->GetDesc();

	// We can't be sure which of the dimensions were used in weights.
	// Thats why we reinterpret all blobs like BatchWidth x Channels.

	// Blob for input hidden layer.
	weightDesc.SetDimSize( BD_BatchLength, 1 );
	weightDesc.SetDimSize( BD_BatchWidth, 4 * newHiddenSize );
	weightDesc.SetDimSize( BD_ListSize, 1 );
	weightDesc.SetDimSize( BD_Height, 1 );
	weightDesc.SetDimSize( BD_Width, 1 );
	weightDesc.SetDimSize( BD_Depth, 1 );
	weightDesc.SetDimSize( BD_Channels, newWeights->GetObjectSize() - newHiddenSize );
	splitWeights.Add( CDnnBlob::CreateBlob( MathEngine(), weightDesc ) );

	// Blob for recurrent hidden layer.
	weightDesc.SetDimSize( BD_Channels, newHiddenSize );
	splitWeights.Add( CDnnBlob::CreateBlob( MathEngine(), weightDesc ) );

	// Setting weightDesc to the (.
	weightDesc.SetDimSize( BD_Channels, newWeights->GetObjectSize() );

	CArray<CBlobDesc> splitDesc;
	splitDesc.Add( splitWeights[0]->GetDesc() );
	splitDesc.Add( splitWeights[1]->GetDesc() );

	CArray<CFloatHandle> splitData;
	splitData.Add( splitWeights[0]->GetData() );
	splitData.Add( splitWeights[1]->GetData() );

	MathEngine().BlobSplitByDim( BD_Channels, weightDesc, newWeights->GetData(), splitDesc.GetPtr(), splitData.GetPtr(), 2 );

	SetInputWeightsData( splitWeights[0] );
	SetRecurWeightsData( splitWeights[1] );
}

void CLstmLayer::fastLstm() 
{
	auto initRecurentBlob = [&]( CPtr<CDnnBlob>& backlinkBlob, int num ) {
		if( inputBlobs.Size() > num && inputBlobs[num] != nullptr ) {
			CPtr<CDnnBlob> windowBlob = CDnnBlob::CreateWindowBlob( backlinkBlob );
			windowBlob->SetParentPos( IsReverseSequence() ? backlinkBlob->GetBatchLength() - 1 : 0 );
			NeoAssert( windowBlob->GetDataSize() == inputBlobs[num]->GetDataSize() );
			MathEngine().VectorCopy( windowBlob->GetData(), inputBlobs[num]->GetData(), windowBlob->GetDataSize() );
		} else {
			backlinkBlob->Clear();
		}
	};

	const size_t hiddenSize = GetHiddenSize();

	// Write state data directly to output or create temporary blob for recurent 
	CPtr<CDnnBlob> stateBacklinkBlob;
	if( outputDescs.Size() == 2 ) {
		stateBacklinkBlob = outputBlobs[1];
	} else {
		stateBacklinkBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, outputDescs[0].BatchWidth(), outputDescs[0].ObjectSize() );
	}

	CPtr<CDnnBlob>& inputWeights = inputHiddenLayer->GetWeightsData();
	CPtr<CDnnBlob>& inputFreeTerm = inputHiddenLayer->GetFreeTermData();
	CPtr<CDnnBlob>& recurrentWeights = recurHiddenLayer->GetWeightsData();
	CPtr<CDnnBlob>& recurrentFreeTerm = recurHiddenLayer->GetFreeTermData();

	// Emulate working of LSTM recurrent implementation
	CPtr<CDnnBlob>& mainBacklink = outputBlobs[0];
	CPtr<CDnnBlob>& stateBacklink = stateBacklinkBlob;

	// Create temporary blobs for result of fully connected layers
	CPtr<CDnnBlob> inputFullyConnectedResult = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, inputDescs[0].BatchWidth(), G_Count * hiddenSize );
	CPtr<CDnnBlob> reccurentFullyConnectedResult = CDnnBlob::CreateBlob( MathEngine(), CT_Float, inputFullyConnectedResult->GetDesc() );

	CLstmDesc* lstmDesc = MathEngine().InitLstm(
		inputWeights->GetData(), inputFreeTerm.Ptr() ? &( inputFreeTerm->GetData() ) : nullptr,
		recurrentWeights->GetData(), recurrentFreeTerm.Ptr() ? &( recurrentFreeTerm->GetData() ) : nullptr,
		inputFullyConnectedResult->GetData(), reccurentFullyConnectedResult->GetData(),
		hiddenSize, inputDescs[0].BatchWidth(), inputDescs[0].ObjectSize() );

	CPtr<CDnnBlob> mainBacklinkInput = CDnnBlob::CreateWindowBlob( mainBacklink );
	CPtr<CDnnBlob> mainBacklinkOutput = CDnnBlob::CreateWindowBlob( mainBacklink );
	CPtr<CDnnBlob> input = CDnnBlob::CreateWindowBlob( inputBlobs[0] );
	CPtr<CDnnBlob> stateBacklinkInput = CDnnBlob::CreateWindowBlob( stateBacklink );
	CPtr<CDnnBlob> stateBacklinkOutput = CDnnBlob::CreateWindowBlob( stateBacklink );

	// Init state and main backlink blobs
	initRecurentBlob( stateBacklink, 1 );
	initRecurentBlob( mainBacklink, 2 );

	// Create termporary blobs for result of dropout (if it is applied)
	CPtr<CDnnBlob> tempMainBackLink;
	CPtr<CDnnBlob> tempInput;

	if( GetDropoutRate() != 0. ) {
		// Create temp vector in order to apply dropout
		tempMainBackLink = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, mainBacklink->GetBatchWidth(), mainBacklink->GetObjectSize() );
		tempInput = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, input->GetBatchWidth(), input->GetObjectSize() );
	}

	// Iterate recurent net step by step
	for( int i = 0; i < inputBlobs[0]->GetBatchLength(); i++ ) {
		int inputPos, outputPos;
		if( IsReverseSequence() ) {
			const int LastIdx = inputBlobs[0]->GetBatchLength() - 1;
			int iRev = LastIdx - i;
			inputPos = min( LastIdx, iRev + 1 );
			outputPos = iRev;
		} else {
			inputPos = max( 0, i - 1 );
			outputPos = i;
		}
		// Set current step
		mainBacklinkInput->SetParentPos( inputPos );
		input->SetParentPos( outputPos );

		// Apply dropout
		if( GetDropoutRate() != 0. ) {
			dropoutRunOnce( mainBacklinkInput, tempMainBackLink );
			dropoutRunOnce( input, tempInput );
		} else {
			// Just create alias if dropout isn't set
			tempMainBackLink = mainBacklinkInput;
			tempInput = input;
		}

		if( outputDescs.Size() == 2 ) {
			// if ( outputDescs.Size() == 1 ) we could preserve only one step of state ( and we do it )
			stateBacklinkInput->SetParentPos( inputPos );
			stateBacklinkOutput->SetParentPos( outputPos );
		}
		mainBacklinkOutput->SetParentPos( outputPos );

		MathEngine().Lstm( *lstmDesc, stateBacklinkInput->GetData(), tempMainBackLink->GetData(),
			tempInput->GetData(), stateBacklinkOutput->GetData(), mainBacklinkOutput->GetData() );
	}

	delete lstmDesc;
}

void CLstmLayer::dropoutRunOnce( const CPtr<CDnnBlob>& src, CPtr<CDnnBlob>& dst )
{
	if( GetDropoutRate() == 0. ) {
		// Dropout isn't applied
		return;
	}

	if( !IsBackwardPerformed() ) {
		MathEngine().VectorCopy( dst->GetData(), src->GetData(), src->GetDataSize() );
		return;
	}

	CDropoutDesc* dropoutDesc = MathEngine().InitDropout( GetDropoutRate(), false, false, src->GetDesc(), dst->GetDesc(),
			GetDnn()->Random().Next() );
	MathEngine().Dropout( *dropoutDesc, src->GetData(), dst->GetData() );
	delete dropoutDesc;
}

CLayerWrapper<CLstmLayer> Lstm(
	int hiddenSize, float dropoutRate, bool isInCompatibilityMode )
{
	return CLayerWrapper<CLstmLayer>( "", [=]( CLstmLayer* result ) {
		result->SetHiddenSize( hiddenSize );
		result->SetDropoutRate( dropoutRate );
		result->SetCompatibilityMode( isInCompatibilityMode );
	} );
}

} // namespace NeoML
