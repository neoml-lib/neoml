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

#include <NeoML/Dnn/Layers/QRNNLayer.h>

namespace NeoML {

// Layer names inside CQRNNLayer composite
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
static const char* recurrentPartName = "RecurrentPart";
static const char* prevStateMultName = "PrevStateMult";
static const char* resultSumName = "ResultSum";
static const char* backLinkName = "BackLink";

CQRNNLayer::CQRNNLayer( IMathEngine& mathEngine ) :
	CCompositeLayer( mathEngine ),
	activation( AF_Tanh )
{
	buildLayer();
}

void CQRNNLayer::SetHiddenSize( int hiddenSize )
{
	if( GetHiddenSize() == hiddenSize ) {
		return;
	}

	timeConv->SetFilterCount( 3 * hiddenSize );
	split->SetOutputCounts3( hiddenSize, hiddenSize );
	backLink->SetDimSize( BD_Channels, hiddenSize );
	ForceReshape();
}

void CQRNNLayer::SetWindowSize( int windowSize )
{
	NeoAssert( windowSize > 0 );
	if( timeConv->GetFilterSize() == windowSize ) {
		return;
	}

	timeConv->SetFilterSize( windowSize );
	ForceReshape();
}

void CQRNNLayer::SetStride( int stride )
{
	NeoAssert( stride > 0 );
	if( timeConv->GetStride() == stride ) {
		return;
	}

	timeConv->SetStride( stride );
	ForceReshape();
}

void CQRNNLayer::SetPaddingFront( int padding )
{
	NeoAssert( padding >= 0 );
	if( timeConv->GetPaddingFront() == padding ) {
		return;
	}

	timeConv->SetPaddingFront( padding );
	ForceReshape();
}

void CQRNNLayer::SetActivation( TActivationFunction newActivation )
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

void CQRNNLayer::SetDropout( float rate )
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
}

static const int QRNNLayerVersion = 0;

void CQRNNLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( QRNNLayerVersion );
	CCompositeLayer::Serialize( archive );
	
	int activationInt = static_cast<int>( activation );
	archive.Serialize( activationInt );
	activation = static_cast<TActivationFunction>( activationInt );

	if( archive.IsLoading() ) {
		timeConv = CheckCast<CTimeConvLayer>( GetLayer( timeConvName ) );
		split = CheckCast<CSplitChannelsLayer>( GetLayer( splitName ) );
		forgetSigmoid = CheckCast<CSigmoidLayer>( GetLayer( forgetSigmoidName ) );
		negForgetByUpdateByOutput = CheckCast<CEltwiseNegMulLayer>( GetLayer( negForgetByUpdateByOutputName ) );
		forgetByOutput = CheckCast<CEltwiseMulLayer>( GetLayer( forgetByOutputName ) );
		recurrentPart = CheckCast<CRecurrentLayer>( GetLayer( recurrentPartName ) );
		backLink = CheckCast<CBackLinkLayer>( recurrentPart->GetLayer( backLinkName ) );
		// Optional layers
		if( HasLayer( dropoutName ) ) {
			dropout = CheckCast<CDropoutLayer>( GetLayer( dropoutName ) );
			postDropoutLinear = CheckCast<CLinearLayer>( GetLayer( postDropoutLinearName ) );
		} else {
			dropout = nullptr;
			postDropoutLinear = nullptr;
		}
	}
}

void CQRNNLayer::buildLayer()
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

	forgetSigmoid = new CSigmoidLayer( MathEngine() );
	forgetSigmoid->SetName( forgetSigmoidName );
	forgetSigmoid->Connect( 0, *split, G_Forget );
	AddLayer( *forgetSigmoid );

	CPtr<CBaseLayer> activationLayer = CreateActivationLayer( MathEngine(), activation );
	activationLayer->SetName( updateActivationName );
	activationLayer->Connect( 0, *split, G_Update );
	AddLayer( *activationLayer );

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

	buildRecurrentPart();

	recurrentPart->Connect( 0, *forgetByOutput );
	recurrentPart->Connect( 1, *negForgetByUpdateByOutput );

	SetInputMapping( 1, *recurrentPart, 2 );
	SetOutputMapping( *recurrentPart );
}

void CQRNNLayer::buildRecurrentPart()
{
	recurrentPart = new CRecurrentLayer( MathEngine() );
	recurrentPart->SetName( recurrentPartName );

	backLink = new CBackLinkLayer( MathEngine() );
	backLink->SetName( backLinkName );
	recurrentPart->AddBackLink( *backLink );
	recurrentPart->SetInputMapping( 2, *backLink, 1 );

	CPtr<CEltwiseMulLayer> prevStateMult = new CEltwiseMulLayer( MathEngine() );
	prevStateMult->SetName( prevStateMultName );
	recurrentPart->AddLayer( *prevStateMult );
	recurrentPart->SetInputMapping( 0, *prevStateMult, 0 );
	prevStateMult->Connect( 1, *backLink );

	CPtr<CEltwiseSumLayer> resultSum = new CEltwiseSumLayer( MathEngine() );
	resultSum->SetName( resultSumName );
	recurrentPart->AddLayer( *resultSum );
	recurrentPart->SetInputMapping( 1, *resultSum, 0 );
	resultSum->Connect( 1, *prevStateMult );

	recurrentPart->SetOutputMapping( *resultSum );
	backLink->Connect( *resultSum );

	AddLayer( *recurrentPart );
}

void CQRNNLayer::addDropout( float rate )
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

void CQRNNLayer::deleteDropout()
{
	DeleteLayer( preDropoutLinearName );
	DeleteLayer( *dropout );
	DeleteLayer( *postDropoutLinear );
	dropout = nullptr;
	postDropoutLinear = nullptr;
	ForceReshape();
}

} // namespace NeoML
