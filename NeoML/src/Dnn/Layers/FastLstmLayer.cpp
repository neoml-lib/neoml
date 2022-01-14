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
	hiddenSize( 0 ),
	useDropout( false ),
	dropoutRate( 0. ),
	dropoutDesc( nullptr ),
	isReverseSequence( false ),
	lstmDesc( nullptr )
{
	paramBlobs.SetSize( 4 );
}

CFastLstmLayer::~CFastLstmLayer()
{
	if( dropoutDesc != nullptr ) {
		delete dropoutDesc;
	}

	if( lstmDesc != nullptr ) {
		delete lstmDesc;
	}
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
	if( dropoutDesc != nullptr ) {
		delete dropoutDesc;
		dropoutDesc = nullptr;
	}
}

void CFastLstmLayer::Reshape()
{
	CheckInputs();
	CheckOutputs();
	CheckArchitecture( outputDescs.Size() == 1 || outputDescs.Size() == 2,
		GetName(), "Lstm layer has only 1 or 2 outputs" );

	initWeightAndFreeTerm( inputWeights(), inputFreeTerm(), 0, inputDescs[0].ObjectSize() );
	initWeightAndFreeTerm( recurrentWeights(), recurrentFreeTerm(), 1, hiddenSize );


	for( int i = 0; i < outputDescs.Size(); i++ ) {
		outputDescs[i] = inputDescs[0];
		outputDescs[i].SetDimSize( BD_Height, 1 );
		outputDescs[i].SetDimSize( BD_Width, 1 );
		outputDescs[i].SetDimSize( BD_Depth, 1 );
		outputDescs[i].SetDimSize( BD_Channels, hiddenSize );
	}

	if( outputDescs.Size() == 2 ) {
		// We will write state directly to output ( we bind them on RunOnce step )
		stateBacklinkBlob = 0;
	} else {
		// Create temporary state blob
		stateBacklinkBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float,
			1, outputDescs[0].BatchWidth(), outputDescs[0].ObjectSize() );
	}

	// Create temporary blobs for result of fully connected layers
	inputFullyConnectedResult = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, inputDescs[0].BatchWidth(), G_Count * hiddenSize );
	reccurentFullyConnectedResult = CDnnBlob::CreateBlob( MathEngine(), CT_Float, inputFullyConnectedResult->GetDesc() );

	if( lstmDesc != nullptr ) {
		delete lstmDesc;
	}
}

void CFastLstmLayer::RunOnce()
{
	bool hasStateBacklinkOutput = outputDescs.Size() == 2;
	if( hasStateBacklinkOutput ) {
		stateBacklinkBlob = outputBlobs[1];
	}

	if( lstmDesc == nullptr ) {
		lstmDesc = MathEngine().InitLstm(
			inputFullyConnectedResult->GetData(), reccurentFullyConnectedResult->GetData(),
			hiddenSize, inputDescs[0].BatchWidth(), inputDescs[0].ObjectSize() );
	}

	CPtr<CDnnBlob> mainBacklinkInput = CDnnBlob::CreateWindowBlob( mainBacklink() );
	CPtr<CDnnBlob> mainBacklinkOutput = CDnnBlob::CreateWindowBlob( mainBacklink() );
	CPtr<CDnnBlob> input = CDnnBlob::CreateWindowBlob( inputBlobs[0] );
	CPtr<CDnnBlob> stateBacklinkInput = CDnnBlob::CreateWindowBlob( stateBacklink() );
	CPtr<CDnnBlob> stateBacklinkOutput = CDnnBlob::CreateWindowBlob( stateBacklink() );
	

	// Init state and main backlink blobs
	initBacklinkBlobs();

	// Create termporary blobs for result of dropout (if it is applied)
	CPtr<CDnnBlob> tempMainBackLink;
	CPtr<CDnnBlob> tempInput;
	if( useDropout ) {
		// Create temp vector in order to apply dropout
		tempMainBackLink = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, mainBacklink()->GetBatchWidth(), mainBacklink()->GetObjectSize() );
		tempInput = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, input->GetBatchWidth(), input->GetObjectSize() );
	}

	// Iterate recurent net step by step
	for( int i = 0; i < inputBlobs[0]->GetBatchLength(); i++ ) {
		int inputPos, outputPos;
		if( isReverseSequence ) {
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
		if( useDropout ) {
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

		MathEngine().Lstm( *lstmDesc, 
			inputWeights()->GetData(), inputFreeTerm().Ptr() ? &( inputFreeTerm()->GetData() ) : nullptr,
			recurrentWeights()->GetData(), recurrentFreeTerm().Ptr() ? &( recurrentFreeTerm()->GetData() ) : nullptr, 
			stateBacklinkInput->GetData(), tempMainBackLink->GetData(),
			tempInput->GetData(), stateBacklinkOutput->GetData(), mainBacklinkOutput->GetData() );
	}
}

void CFastLstmLayer::BackwardOnce()
{
	// FIXME:
}

void CFastLstmLayer::LearnOnce()
{
	// FIXME:
}

void CFastLstmLayer::initWeightAndFreeTerm( CDnnBlob* weight, CDnnBlob* freeTerm, int inputIndex, size_t inputObjectSize )
{
	const size_t MatrixWidth = G_Count * hiddenSize;
	const size_t WeightObjectSize = inputObjectSize * MatrixWidth;
	const size_t FreeTermObjectSize = 1 * MatrixWidth;

	if( weight == 0 ) {
		// Create a weights matrix
		weight = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 1, WeightObjectSize );
		// Initialize
		InitializeParamBlob( inputIndex, *weight );
	} else {
		CheckArchitecture( weight->GetDataSize() == WeightObjectSize,
			GetName(), "weights have wrong size" );
	}

	if( freeTerm == 0 ) {
		freeTerm = CDnnBlob::CreateVector( MathEngine(), CT_Float, FreeTermObjectSize );
		// Initialize
		freeTerm->Fill( 0 );
	} else {
		CheckArchitecture( freeTerm->GetDataSize() == FreeTermObjectSize,
			GetName(), "free terms have wrong size" );
	}
}

void CFastLstmLayer::setData( CPtr<CDnnBlob>& dst, CDnnBlob* src, bool makeCopy )
{
	if( !makeCopy ) {
		dst = src;
	} else {
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
}

CPtr<CDnnBlob> CFastLstmLayer::getData( const CPtr<CDnnBlob>& data ) const
{
	if( data == 0 ) {
		return 0;
	}

	return data->GetCopy();
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
			CPtr<CDnnBlob> windowBlob = CDnnBlob::CreateWindowBlob( backlinkBlob );
			windowBlob->SetParentPos( isReverseSequence ? backlinkBlob->GetBatchLength() - 1 : 0 );
			NeoAssert( windowBlob->GetDataSize() == inputBlobs[num]->GetDataSize() );
			MathEngine().VectorCopy( windowBlob->GetData(), inputBlobs[num]->GetData(), windowBlob->GetDataSize() );
		} else {
			backlinkBlob->Clear();
		}
	};

	initRecurentBlob( stateBacklink(), 1 );
	initRecurentBlob( mainBacklink(), 2 );
}
} // namespace NeoML
