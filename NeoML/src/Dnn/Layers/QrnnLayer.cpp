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

#include <NeoML/Dnn/Layers/QrnnLayer.h>
#include <NeoML/Dnn/Layers/ConcatLayer.h>

namespace NeoML {

// Layer names inside CQrnnLayer composite
static const char* timeConvName = "TimeConv";
static const char* splitName = "Split";
static const char* updateActivationName = "UpdateGateActivation";
static const char* forgetSigmoidName = "ForgetSigmoid";
static const char* preDropoutLinearName = "PreDropoutLinear";
static const char* dropoutName = "Dropout";
static const char* postDropoutLinearName = "PostDropoutLinear";
static const char* outputSigmoidName = "OutputSigmoid";
static const char* negForgetByUpdateByOutputName = "NegForgetByUpdateByOutput";
static const char* forgetByOutputName = "ForgetByOutput";
static const char* firstRecurrentName = "FirstRecurrent";
static const char* secondRecurrentName = "SecondRecurrent";
static const char* bidirectionalMergeName = "BidirectionalMergeName";
// Layer names inside CQrnnLayer recurrent
static const char* prevStateMultName = "PrevStateMult";
static const char* resultSumName = "ResultSum";
static const char* backLinkName = "BackLink";

static inline bool IsBidirectional( CQrnnLayer::TRecurrentMode mode )
{
	static_assert( CQrnnLayer::RM_Count == 4, "CQrnnLayer::RM_Count != 4" );
	return mode == CQrnnLayer::RM_BidirectionalConcat || mode == CQrnnLayer::RM_BidirectionalSum;
}

CQrnnLayer::CQrnnLayer( IMathEngine& mathEngine ) :
	CCompositeLayer( mathEngine ),
	activation( AF_Tanh ),
	recurrentMode( RM_Direct )
{
	buildLayer();
}

void CQrnnLayer::SetHiddenSize( int hiddenSize )
{
	if( GetHiddenSize() == hiddenSize ) {
		return;
	}

	timeConv->SetFilterCount( 3 * hiddenSize );
	split->SetOutputCounts3( hiddenSize, hiddenSize );
	firstBackLink->SetDimSize( BD_Channels, hiddenSize );
	if( secondBackLink != nullptr ) {
		secondBackLink->SetDimSize( BD_Channels, hiddenSize );
	}
	ForceReshape();
}

void CQrnnLayer::SetWindowSize( int windowSize )
{
	NeoAssert( windowSize > 0 );
	if( timeConv->GetFilterSize() == windowSize ) {
		return;
	}

	timeConv->SetFilterSize( windowSize );
	ForceReshape();
}

void CQrnnLayer::SetStride( int stride )
{
	NeoAssert( stride > 0 );
	if( timeConv->GetStride() == stride ) {
		return;
	}

	timeConv->SetStride( stride );
	ForceReshape();
}

void CQrnnLayer::SetPaddingFront( int padding )
{
	NeoAssert( padding >= 0 );
	if( GetPaddingFront() == padding ) {
		return;
	}

	timeConv->SetPaddingFront( padding );
	ForceReshape();
}

void CQrnnLayer::SetPaddingBack( int padding )
{
	NeoAssert( padding >= 0 );
	if( GetPaddingBack() == padding ) {
		return;
	}

	timeConv->SetPaddingBack( padding );
	ForceReshape();
}

void CQrnnLayer::SetActivation( TActivationFunction newActivation )
{
	if( activation == newActivation ) {
		return;
	}

	activation = newActivation;
	DeleteLayer( updateActivationName );
	CPtr<CBaseLayer> activationLayer = CreateActivationLayer( MathEngine(), activation );
	activationLayer->SetName( updateActivationName );
	activationLayer->Connect( 0, *split, G_Update );
	negForgetByUpdateByOutput->Connect( 1, *activationLayer );
	AddLayer( *activationLayer );

	ForceReshape();
}

void CQrnnLayer::SetDropout( float rate )
{
	NeoAssert( rate >= 0.f && rate <= 1.f );
	if( rate == 0.f && dropout != nullptr ) {
		deleteDropout();
	} else if( rate != 0.f && dropout == nullptr ) {
		addDropout( rate );
	} else if( rate != 0.f ) {
		dropout->SetDropoutRate( rate );
		postDropoutLinear->SetMultiplier( 1.f - rate );
	}

	if( rate == 0.f ) {
		NeoAssert( dropout == nullptr && postDropoutLinear == nullptr );
	} else {
		NeoAssert( dropout != nullptr && dropout->GetDropoutRate() == rate 
			&& postDropoutLinear != nullptr );
	}
}

void CQrnnLayer::SetRecurrentMode( CQrnnLayer::TRecurrentMode newMode )
{
	if( recurrentMode == newMode ) {
		return;
	}

	recurrentMode = newMode;

	if( IsBidirectional( recurrentMode ) ) {
		createBidirectionalLayers();
		firstRecurrent->SetReverseSequence( false );
	} else {
		deleteBidirectionalLayers();
		static_assert( RM_Count == 4, "RM_Count != 4" );
		firstRecurrent->SetReverseSequence( recurrentMode == RM_Reverse );
	}

	ForceReshape();

	static_assert( RM_Count == 4, "RM_Count != 4" );
	if( recurrentMode == RM_Direct ) {
		NeoAssert( !firstRecurrent->IsReverseSequence() && secondRecurrent == nullptr
			&& secondBackLink == nullptr && bidirectionalMerge == nullptr );
	} else if( recurrentMode == RM_Reverse ) {
		NeoAssert( firstRecurrent->IsReverseSequence() && secondRecurrent == nullptr
			&& secondBackLink == nullptr && bidirectionalMerge == nullptr );
	} else if( recurrentMode == RM_BidirectionalConcat ) {
		NeoAssert( !firstRecurrent->IsReverseSequence() && secondRecurrent != nullptr && secondRecurrent->IsReverseSequence()
			&& secondBackLink != nullptr && dynamic_cast<CConcatChannelsLayer*>( bidirectionalMerge.Ptr() ) != nullptr );
	} else if( recurrentMode == RM_BidirectionalSum ) {
		NeoAssert( !firstRecurrent->IsReverseSequence() && secondRecurrent != nullptr && secondRecurrent->IsReverseSequence()
			&& secondBackLink != nullptr && dynamic_cast<CEltwiseSumLayer*>( bidirectionalMerge.Ptr() ) != nullptr );
	} else {
		NeoAssert( false );
	}
}

static const int QrnnLayerVersion = 0;

void CQrnnLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( QrnnLayerVersion );
	CCompositeLayer::Serialize( archive );
	
	int activationInt = static_cast<int>( activation );
	archive.Serialize( activationInt );
	activation = static_cast<TActivationFunction>( activationInt );

	int recurrentModeInt = static_cast<int>( recurrentMode );
	archive.Serialize( recurrentModeInt );
	recurrentMode = static_cast<TRecurrentMode>( recurrentModeInt );

	if( archive.IsLoading() ) {
		timeConv = CheckCast<CTimeConvLayer>( GetLayer( timeConvName ) );
		split = CheckCast<CSplitChannelsLayer>( GetLayer( splitName ) );
		forgetSigmoid = CheckCast<CSigmoidLayer>( GetLayer( forgetSigmoidName ) );
		negForgetByUpdateByOutput = CheckCast<CEltwiseNegMulLayer>( GetLayer( negForgetByUpdateByOutputName ) );
		forgetByOutput = CheckCast<CEltwiseMulLayer>( GetLayer( forgetByOutputName ) );
		firstRecurrent = CheckCast<CRecurrentLayer>( GetLayer( firstRecurrentName ) );
		firstBackLink = CheckCast<CBackLinkLayer>( firstRecurrent->GetLayer( backLinkName ) );
		// Optional layers
		if( HasLayer( dropoutName ) ) {
			dropout = CheckCast<CDropoutLayer>( GetLayer( dropoutName ) );
			postDropoutLinear = CheckCast<CLinearLayer>( GetLayer( postDropoutLinearName ) );
		} else {
			dropout = nullptr;
			postDropoutLinear = nullptr;
		}
		if( IsBidirectional( recurrentMode ) ) {
			secondRecurrent = CheckCast<CRecurrentLayer>( GetLayer( secondRecurrentName ) );
			secondBackLink = CheckCast<CBackLinkLayer>( secondRecurrent->GetLayer( backLinkName ) );
			bidirectionalMerge = GetLayer( bidirectionalMergeName );
		}
	}
}

void CQrnnLayer::buildLayer()
{
	timeConv = new CTimeConvLayer( MathEngine() );
	timeConv->SetName( timeConvName );
	timeConv->SetFilterCount( 3 );
	timeConv->SetFilterSize( 1 );
	AddLayer( *timeConv );
	SetInputMapping( *timeConv );

	split = new CSplitChannelsLayer( MathEngine() );
	split->SetName( splitName );
	split->SetOutputCounts3( 1, 1 );
	split->Connect( *timeConv );
	AddLayer( *split );

	CPtr<CBaseLayer> activationLayer = CreateActivationLayer( MathEngine(), activation );
	activationLayer->SetName( updateActivationName );
	activationLayer->Connect( 0, *split, G_Update );
	AddLayer( *activationLayer );

	forgetSigmoid = new CSigmoidLayer( MathEngine() );
	forgetSigmoid->SetName( forgetSigmoidName );
	forgetSigmoid->Connect( 0, *split, G_Forget );
	AddLayer( *forgetSigmoid );

	CPtr<CSigmoidLayer> outputSigmoid = new CSigmoidLayer( MathEngine() );
	outputSigmoid->SetName( outputSigmoidName );
	outputSigmoid->Connect( 0, *split, G_Output );
	AddLayer( *outputSigmoid );

	negForgetByUpdateByOutput = new CEltwiseNegMulLayer( MathEngine() );
	negForgetByUpdateByOutput->SetName( negForgetByUpdateByOutputName );
	negForgetByUpdateByOutput->Connect( 0, *forgetSigmoid );
	negForgetByUpdateByOutput->Connect( 1, *activationLayer );
	negForgetByUpdateByOutput->Connect( 2, *outputSigmoid );
	AddLayer( *negForgetByUpdateByOutput );

	forgetByOutput = new CEltwiseMulLayer( MathEngine() );
	forgetByOutput->SetName( forgetByOutputName );
	forgetByOutput->Connect( 0, *forgetSigmoid );
	forgetByOutput->Connect( 1, *outputSigmoid );
	AddLayer( *forgetByOutput );

	NeoAssert( recurrentMode == RM_Direct );
	firstRecurrent = buildRecurrentPart( firstRecurrentName );
	firstRecurrent->Connect( 0, *forgetByOutput );
	firstRecurrent->Connect( 1, *negForgetByUpdateByOutput );
	AddLayer( *firstRecurrent );
	firstBackLink = CheckCast<CBackLinkLayer>( firstRecurrent->GetLayer( backLinkName ) );

	SetInputMapping( 1, *firstRecurrent, 2 );
	SetOutputMapping( *firstRecurrent );
}

CPtr<CRecurrentLayer> CQrnnLayer::buildRecurrentPart( const char* name )
{
	CPtr<CRecurrentLayer> result = new CRecurrentLayer( MathEngine() );
	result->SetName( name );

	CPtr<CBackLinkLayer> backLink = new CBackLinkLayer( MathEngine() );
	backLink->SetName( backLinkName );
	result->AddBackLink( *backLink );
	result->SetInputMapping( 2, *backLink, 1 );

	CPtr<CEltwiseMulLayer> prevStateMult = new CEltwiseMulLayer( MathEngine() );
	prevStateMult->SetName( prevStateMultName );
	result->AddLayer( *prevStateMult );
	result->SetInputMapping( 0, *prevStateMult, 0 );
	prevStateMult->Connect( 1, *backLink );

	CPtr<CEltwiseSumLayer> resultSum = new CEltwiseSumLayer( MathEngine() );
	resultSum->SetName( resultSumName );
	result->AddLayer( *resultSum );
	result->SetInputMapping( 1, *resultSum, 0 );
	resultSum->Connect( 1, *prevStateMult );

	result->SetOutputMapping( *resultSum );
	backLink->Connect( *resultSum );
	return result;
}

void CQrnnLayer::addDropout( float rate )
{
	// the article recommends new_F = 1 - dropout( 1 - F )
	// but in order to make less multiplications we use new_F = 1 + dropout( F - 1 )
	
	// subtracting 1 from F
	CPtr<CLinearLayer> preDropoutLinear = new CLinearLayer( MathEngine() );
	preDropoutLinear->SetName( preDropoutLinearName );
	preDropoutLinear->SetMultiplier( 1.f );
	preDropoutLinear->SetFreeTerm( -1.f );
	preDropoutLinear->Connect( *forgetSigmoid );
	AddLayer( *preDropoutLinear );

	// applying dropout to F - 1
	dropout = new CDropoutLayer( MathEngine() );
	dropout->SetName( dropoutName );
	dropout->SetDropoutRate( rate );
	dropout->Connect( *preDropoutLinear );
	AddLayer( *dropout );

	// adding 1 to the dropout result
	postDropoutLinear = new CLinearLayer( MathEngine() );
	postDropoutLinear->SetName( postDropoutLinearName );
	postDropoutLinear->SetMultiplier( 1.f - rate );
	postDropoutLinear->SetFreeTerm( 1.f );
	postDropoutLinear->Connect( *dropout );
	AddLayer( *postDropoutLinear );

	negForgetByUpdateByOutput->Connect( 0, *postDropoutLinear );
	forgetByOutput->Connect( 0, *postDropoutLinear );

	ForceReshape();
}

void CQrnnLayer::deleteDropout()
{
	DeleteLayer( preDropoutLinearName );
	DeleteLayer( *dropout );
	DeleteLayer( *postDropoutLinear );
	dropout = nullptr;
	postDropoutLinear = nullptr;
	negForgetByUpdateByOutput->Connect( 0, *forgetSigmoid );
	forgetByOutput->Connect( 0, *forgetSigmoid );
	ForceReshape();
}

void CQrnnLayer::createBidirectionalLayers()
{
	NeoAssert( IsBidirectional( recurrentMode ) );
	if( secondRecurrent == nullptr ) {
		secondRecurrent = buildRecurrentPart( secondRecurrentName );
		secondRecurrent->SetReverseSequence( true );
		secondRecurrent->Connect( 0, *forgetByOutput );
		secondRecurrent->Connect( 1, *negForgetByUpdateByOutput );
		AddLayer( *secondRecurrent );
		secondBackLink = CheckCast<CBackLinkLayer>( secondRecurrent->GetLayer( backLinkName ) );
		secondBackLink->SetDimSize( BD_Channels, firstBackLink->GetDimSize( BD_Channels ) );
		SetInputMapping( 1, *secondRecurrent, 2 );
	}

	if( bidirectionalMerge != nullptr ) {
		// We can be here only when the recurrentMode is changing
		// That means merge layer type was changed
		DeleteLayer( *bidirectionalMerge );
		bidirectionalMerge = nullptr;
	}

	static_assert( CQrnnLayer::RM_Count == 4, "CQrnnLayer::RM_Count != 4" );
	if( recurrentMode == RM_BidirectionalConcat ) {
		bidirectionalMerge = new CConcatChannelsLayer( MathEngine() );
	} else if( recurrentMode == RM_BidirectionalSum ) {
		bidirectionalMerge = new CEltwiseSumLayer( MathEngine() );
	} else {
		NeoAssert( false );
	}
	bidirectionalMerge->SetName( bidirectionalMergeName );
	bidirectionalMerge->Connect( 0, *firstRecurrent );
	bidirectionalMerge->Connect( 1, *secondRecurrent );
	AddLayer( *bidirectionalMerge );
	SetOutputMapping( *bidirectionalMerge );
}

void CQrnnLayer::deleteBidirectionalLayers()
{
	if( secondRecurrent != nullptr ) {
		NeoAssert( bidirectionalMerge != nullptr );
		DeleteLayer( *secondRecurrent );
		secondRecurrent = nullptr;
		secondBackLink = nullptr;
		DeleteLayer( *bidirectionalMerge );
		bidirectionalMerge = nullptr;
		SetOutputMapping( *firstRecurrent );
	}
}

} // namespace NeoML
