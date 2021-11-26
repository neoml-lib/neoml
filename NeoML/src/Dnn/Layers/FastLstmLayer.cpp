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

#include <NeoML/Dnn/Layers/FastLstmLayer.h>

namespace NeoML {

CFastLstmLayer::CFastLstmLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnFastLstmLayer", true ),
	useDropout( false ),
	dropoutRate( 0. ),
	dropoutDesc( nullptr ),
	recurrentActivation( AF_Sigmoid ),
	isInCompatibilityMode( false ),
	isReverseSequence( false ),
	hiddenSize( 0 )
{
}

void CFastLstmLayer::Serialize( CArchive& archive )
{
	// FIXME:
}

void CFastLstmLayer::SetHiddenSize( int size )
{
	hiddenSize = size;
	ForceReshape();
}

void CFastLstmLayer::SetDropoutRate( float newDropoutRate )
{
	dropoutRate = newDropoutRate;
	useDropout = true;
	if( dropoutDesc != 0 ) {
		delete dropoutDesc;
		dropoutDesc = nullptr;
	}
}

void CFastLstmLayer::SetReverseSequence( bool _isReverseSequense )
{
	// FIXME:
}

void CFastLstmLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( inputWeights->GetObjectSize() == G_Count * hiddenSize * inputBlobs[0]->GetObjectSize(),
		GetName(), "wrong input weight size" );
	CheckArchitecture( recurrentWeights->GetObjectSize() == G_Count * hiddenSize * hiddenSize,
		GetName(), "wrong recurrent weight size" );
	CheckArchitecture( inputFreeTerm->GetObjectSize() == G_Count * hiddenSize,
		GetName(), "wrong input freeTerm size" );
	CheckArchitecture( recurrentFreeTerm->GetObjectSize() == G_Count * hiddenSize,
		GetName(), "wrong reccurent freeTerm size" );

	CPtr<CDnnBlob> mainBacklink = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float,
		inputBlobs[0]->GetBatchLength(), inputBlobs[0]->GetBatchWidth(), hiddenSize );
	CPtr<CDnnBlob> stateBacklink = CDnnBlob::CreateBlob( MathEngine(), mainBacklink->GetDesc() );
}

void CFastLstmLayer::RunOnce()
{
	CPtr<CDnnBlob> mainBacklinkWindow = CDnnBlob::CreateWindowBlob( mainBacklink );
	CPtr<CDnnBlob> inputWindow = CDnnBlob::CreateWindowBlob( inputBlobs[0] );

	// Init state and main backlink blobs
	initBacklinkBlobs();

	// Create termporary blobs for result of dropout (if it is applied)
	CPtr<CDnnBlob> tempMainBackLink;
	CPtr<CDnnBlob> tempInput;
	if( useDropout ) {
		// Create temp vector in order to apply dropout
		tempMainBackLink = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, mainBacklink->GetBatchWidth(), mainBacklink->GetObjectSize() );
		tempInput = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, inputWindow->GetBatchWidth(), inputWindow->GetObjectSize() );
	}

	// Create temporary blobs for result of fully connected layers
	CPtr<CDnnBlob> inputFullyConnectedResult = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, inputWindow->GetBatchWidth(), G_Count * hiddenSize );
	CPtr<CDnnBlob> reccurentFullyConnectedResult = CDnnBlob::CreateBlob( MathEngine(), CT_Float, inputFullyConnectedResult->GetDesc() );

	// Iterate recurent net step by step
	for( int i = 0; i < inputBlobs[0]->GetBatchLength(); i++ ) {
		const int InputPos = max( 0, i - 1 );
		const int OutputPos = i;
		// Set current step
		mainBacklinkWindow->SetParentPos( i );
		inputWindow->SetParentPos( i );

		// Apply dropout
		if( useDropout ) {
			dropoutRunOnce( mainBacklinkWindow, tempMainBackLink );
			dropoutRunOnce( inputWindow, tempInput );
		} else {
			// Just create alias if dropout isn't set
			tempMainBackLink = mainBacklinkWindow;
			tempInput = inputWindow;
		}

		// Apply fully connected layers
		fullyconnectedRunOnce( tempInput, inputWeights, inputFullyConnectedResult, inputFreeTerm );
		fullyconnectedRunOnce( tempMainBackLink, recurrentWeights, reccurentFullyConnectedResult, recurrentFreeTerm );

		// Perform remaining calculation of LSTM
		processRestOfLstm( inputFullyConnectedResult, reccurentFullyConnectedResult, InputPos, OutputPos );
	}
}

void CFastLstmLayer::BackwardOnce()
{
	// FIXME:
}

void CFastLstmLayer::setWeightsData( const CPtr<CDnnBlob>& src, CPtr<CDnnBlob>& dst )
{
	if( src == nullptr ) {
		NeoAssert( dst == nullptr || GetDnn() == nullptr );
		dst = 0;
	} else if( dst != nullptr && GetDnn() != nullptr ) {
		NeoAssert( dst->GetObjectCount() == src->GetObjectCount() );
		NeoAssert( dst->GetObjectSize() == src->GetObjectSize() );
		dst->CopyFrom( src );
	} else {
		dst = src->GetCopy();
	}
}

void CFastLstmLayer::setFreeTermData( const CPtr<CDnnBlob>& src, CPtr<CDnnBlob>& dst )
{
	if( src == nullptr ) {
		NeoAssert( dst == nullptr || GetDnn() == nullptr );
		dst = 0;
	} else {
		if( dst != nullptr && GetDnn() != nullptr ) {
			NeoAssert( dst->GetDataSize() == src->GetDataSize() );

			dst->CopyFrom( src );
		} else {
			dst = src->GetCopy();
		}
	}
}

void CFastLstmLayer::dropoutRunOnce( const CPtr<CDnnBlob>& src, CPtr<CDnnBlob>& dst )
{
	if( !useDropout ) {
		// Dropout isn't applied
		return;
	}

	if( !IsBackwardPerformed() ) {
		MathEngine().VectorCopy( dst->GetData(), src->GetData(), src->GetDataSize() );
		return;
	}

	if( dropoutDesc == nullptr ) {
		dropoutDesc = MathEngine().InitDropout( dropoutRate, false, false, src->GetDesc(), dst->GetDesc(),
			GetDnn()->Random().Next() );
	}
	MathEngine().Dropout( *dropoutDesc, src->GetData(), dst->GetData() );
}

void CFastLstmLayer::dropoutBackwardOnce( const CPtr<CDnnBlob>& src, CPtr<CDnnBlob>& dst )
{
	if( !useDropout ) {
		// Dropout isn't applied
		return;
	}

	// Backward pass is only possible when learning
	NeoAssert( dropoutDesc != nullptr );

	MathEngine().Dropout( *dropoutDesc, src->GetData(), dst->GetData() );

	if( !GetDnn()->IsRecurrentMode() || GetDnn()->IsFirstSequencePos() ) {
		// Clear the memory after the whole sequence is processed
		delete dropoutDesc;
		dropoutDesc = nullptr;
	}
}

void CFastLstmLayer::initBacklinkBlobs()
{
	auto initRecurentBlob = [&]( CPtr<CDnnBlob>& backlinkBlob, int num ) {
		if( inputBlobs.Size() > num && inputBlobs[num] != nullptr ) {
			NeoAssert( backlinkBlob->GetObjectSize() == inputBlobs[num]->GetObjectSize() );
			MathEngine().VectorCopy( backlinkBlob->GetObjectData( 0 ), inputBlobs[num]->GetData(), backlinkBlob->GetObjectSize() );
		} else {
			backlinkBlob->ClearObject( 0 );
		}
	};

	initRecurentBlob( stateBacklink, 1 );
	initRecurentBlob( mainBacklink, 2 );
}

void CFastLstmLayer::fullyconnectedRunOnce( const CDnnBlob* input, const CDnnBlob* weights, CDnnBlob* output, CDnnBlob* freeTerm )
{
	CConstFloatHandle inputData = input->GetData();
	CFloatHandle outputData = output->GetData();
	CConstFloatHandle weightData = weights->GetData();

	MathEngine().MultiplyMatrixByTransposedMatrix( inputData, input->GetObjectCount(),
		input->GetObjectSize(), input->GetObjectSize(),
		weightData, G_Count * hiddenSize, weights->GetObjectSize(),
		outputData, output->GetObjectSize(), output->GetObjectSize() * input->GetObjectCount() );

	if( freeTerm != nullptr ) {
		MathEngine().AddVectorToMatrixRows( 1, outputData, outputData, input->GetObjectCount(),
			output->GetObjectSize(), freeTerm->GetData() );
	}
}

void CFastLstmLayer::processRestOfLstm( CDnnBlob* inputFullyConnectedResult, CDnnBlob* reccurentFullyConnectedResult,
	int inputPos, int outputPos )
{
	CPtr<CDnnBlob> stateBacklinkInput = CDnnBlob::CreateWindowBlob( stateBacklink );
	CPtr<CDnnBlob> stateBacklinkOutput = CDnnBlob::CreateWindowBlob( stateBacklink );
	CPtr<CDnnBlob> mainBacklinkOutput = CDnnBlob::CreateWindowBlob( mainBacklink );

	stateBacklinkInput->SetParentPos( inputPos );
	stateBacklinkOutput->SetParentPos( outputPos );
	mainBacklinkOutput->SetParentPos( outputPos );

	// Elementwise summ of fully connected layers' results (inplace)
	CDnnBlob* hiddenLayerSum = inputFullyConnectedResult;
	MathEngine().VectorAdd( inputFullyConnectedResult->GetData(), reccurentFullyConnectedResult->GetData(), hiddenLayerSum->GetData(), hiddenLayerSum->GetDataSize() );
	const int DataSize = mainBacklinkOutput->GetDataSize();
	CFloatHandle forgetData = hiddenLayerSum->GetData();
	CFloatHandle inputData = forgetData + DataSize;
	CFloatHandle inputTanhData = inputData + DataSize;
	CFloatHandle outputData = inputTanhData + DataSize;

	// Apply activations
	MathEngine().VectorSigmoid( forgetData, forgetData, DataSize );
	MathEngine().VectorSigmoid( inputData, inputData, DataSize );
	MathEngine().VectorTanh( inputTanhData, inputTanhData, DataSize );
	MathEngine().VectorSigmoid( outputData, outputData, DataSize );

	// Multiply input gates
	MathEngine().VectorEltwiseMultiply( inputData, inputTanhData, inputData, DataSize );

	// Multiply state backlink with forget gate
	MathEngine().VectorEltwiseMultiply( forgetData, stateBacklinkInput->GetData(), forgetData, DataSize );

	// Append input gate to state backlink
	MathEngine().VectorAdd( stateBacklinkInput->GetData(), inputData, stateBacklinkOutput->GetData(), DataSize );

	// Apply tanh to state baclink
	MathEngine().VectorTanh( stateBacklinkOutput->GetData(), inputData, DataSize );

	// Multiply output gate with result of previous operation
	MathEngine().VectorEltwiseMultiply( outputData, inputData, mainBacklinkOutput->GetData(), DataSize );
}

} // namespace NeoML
