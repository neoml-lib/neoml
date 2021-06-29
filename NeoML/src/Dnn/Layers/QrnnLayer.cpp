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
static const char* inputSigmoidName = "InputSigmoid";
static const char* firstPoolingName = "FirstPooling";
static const char* secondPoolingName = "SecondPooling";
static const char* firstOutputGateName = "FirstOutputGate";
static const char* secondOutputGateName = "SecondOutputGate";
static const char* bidirectionalMergeName = "BidirectionalMergeName";

CQrnnLayer::CQrnnLayer( IMathEngine& mathEngine ) :
	CCompositeLayer( mathEngine ),
	poolingType( PT_FPooling ),
	recurrentMode( RM_Direct ),
	activation( AF_Tanh )
{
	buildLayer( 0.f, 1, 1, 1, 0, 0 );
}

void CQrnnLayer::SetPoolingType( TPoolingType newPoolingType )
{
	if( poolingType == newPoolingType ) {
		return;
	}

	const int prevGateCount = gateCount();
	poolingType = newPoolingType;
	rebuildLayer( prevGateCount );
}

void CQrnnLayer::SetHiddenSize( int hiddenSize )
{
	NeoAssert( hiddenSize > 0 );
	if( GetHiddenSize() == hiddenSize ) {
		return;
	}

	timeConv->SetFilterCount( gateCount() * hiddenSize );
	CArray<int> outputCounts;
	outputCounts.Add( hiddenSize, gateCount() - 1 );
	split->SetOutputCounts( outputCounts );
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
	rebuildLayer( gateCount() );
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
	rebuildLayer( gateCount() );
}

static const int QrnnLayerVersion = 1;

void CQrnnLayer::Serialize( CArchive& archive )
{
	// During move from v0 to v1 the backward compatibility was broken
	archive.SerializeVersion( QrnnLayerVersion, 1 );
	CCompositeLayer::Serialize( archive );

	int poolingTypeInt = static_cast<int>( poolingType );
	archive.Serialize( poolingTypeInt );
	poolingType = static_cast<TPoolingType>( poolingTypeInt );

	int recurrentModeInt = static_cast<int>( recurrentMode );
	archive.Serialize( recurrentModeInt );
	recurrentMode = static_cast<TRecurrentMode>( recurrentModeInt );
	
	int activationInt = static_cast<int>( activation );
	archive.Serialize( activationInt );
	activation = static_cast<TActivationFunction>( activationInt );

	if( archive.IsLoading() ) {
		timeConv = CheckCast<CTimeConvLayer>( GetLayer( timeConvName ) );
		split = CheckCast<CSplitChannelsLayer>( GetLayer( splitName ) );
		forgetSigmoid = CheckCast<CSigmoidLayer>( GetLayer( forgetSigmoidName ) );
		// Optional layers
		if( HasLayer( dropoutName ) ) {
			dropout = CheckCast<CDropoutLayer>( GetLayer( dropoutName ) );
			postDropoutLinear = CheckCast<CLinearLayer>( GetLayer( postDropoutLinearName ) );
		} else {
			dropout = nullptr;
			postDropoutLinear = nullptr;
		}
		firstPooling = GetLayer( firstPoolingName );
		if( HasLayer( secondPoolingName ) ) {
			secondPooling = GetLayer( secondPoolingName );
		}
	}
}

void CQrnnLayer::buildLayer( float dropoutRate, int hiddenSize, int windowSize, int stride, int padFront, int padBack )
{
	// Add time conv
	timeConv = new CTimeConvLayer( MathEngine() );
	timeConv->SetName( timeConvName );
	timeConv->SetFilterCount( hiddenSize * gateCount() );
	timeConv->SetFilterSize( windowSize );
	timeConv->SetStride( stride );
	timeConv->SetPaddingFront( padFront );
	timeConv->SetPaddingBack( padBack );
	AddLayer( *timeConv );
	SetInputMapping( *timeConv );

	// Add split layers
	split = new CSplitChannelsLayer( MathEngine() );
	split->SetName( splitName );
	CArray<int> outputCounts;
	outputCounts.Add( hiddenSize, gateCount() - 1 );
	split->SetOutputCounts( outputCounts );
	split->Connect( *timeConv );
	AddLayer( *split );

	// Apply activation to every gate
	CPtr<CBaseLayer> activationLayer = CreateActivationLayer( MathEngine(), activation );
	activationLayer->SetName( updateActivationName );
	activationLayer->Connect( 0, *split, G_Update );
	AddLayer( *activationLayer );

	forgetSigmoid = addSigmoid( *split, G_Forget, forgetSigmoidName );

	CPtr<CSigmoidLayer> outputSigmoid;
	if( gateCount() > G_Output ) {
		outputSigmoid = addSigmoid( *split, G_Output, outputSigmoidName );
	}

	CPtr<CSigmoidLayer> inputSigmoid;
	if( gateCount() > G_Input ) {
		inputSigmoid = addSigmoid( *split, G_Input, inputSigmoidName );
	}

	static_assert( RM_Count == 4, "RM_Count != 4" );
	firstPooling = addPoolingLayer( firstPoolingName, recurrentMode == RM_Reverse );
	if( isBidirectional() ) {
		secondPooling = addPoolingLayer( secondPoolingName, true );
		if( gateCount() > G_Output ) {
			CPtr<CEltwiseMulLayer> firstOutputGate = addMulLayer( *firstPooling, *outputSigmoid, firstOutputGateName );
			CPtr<CEltwiseMulLayer> secondOutputGate = addMulLayer( *secondPooling, *outputSigmoid, secondOutputGateName );
			CPtr<CBaseLayer> mergeLayer = addBidirectionalMerge( *firstOutputGate, *secondOutputGate, bidirectionalMergeName );
			SetOutputMapping( *mergeLayer );
		} else {
			CPtr<CBaseLayer> mergeLayer = addBidirectionalMerge( *firstPooling, *secondPooling, bidirectionalMergeName );
			SetOutputMapping( *mergeLayer );
		}
	} else {
		secondPooling = nullptr;
		if( gateCount() > G_Output ) {
			CPtr<CEltwiseMulLayer> outputGate = addMulLayer( *firstPooling, *outputSigmoid, firstOutputGateName );
			SetOutputMapping( *outputGate );
		} else {
			SetOutputMapping( *firstPooling );
		}
	}

	addInitialStateInputMapping( *firstPooling, 1 );
	if( isBidirectional() ) {
		addInitialStateInputMapping( *secondPooling, 2 );
	}

	dropout = nullptr;
	postDropoutLinear = nullptr;
	if( dropoutRate > 0.f ) {
		addDropout( dropoutRate );
	}
}

// Adds sigmoid layer after outputIndex'th output of input layer
CPtr<CSigmoidLayer> CQrnnLayer::addSigmoid( CBaseLayer& input, int outputIndex, const char* sigmoidName )
{
	CPtr<CSigmoidLayer> sigmoid = new CSigmoidLayer( MathEngine() );
	sigmoid->SetName( sigmoidName );
	sigmoid->Connect( 0, input, outputIndex );
	AddLayer( *sigmoid );
	return sigmoid;
}

// Adds qrnn pooling layer
CPtr<CBaseLayer> CQrnnLayer::addPoolingLayer( const char* poolingName, bool reverse )
{
	CPtr<CBaseLayer> pooling;
	if( poolingType == PT_IfoPooling ) {
		CPtr<CQrnnIfPoolingLayer> ifPooling = new CQrnnIfPoolingLayer( MathEngine() );
		ifPooling->SetReverse( reverse );
		ifPooling->Connect( 2, inputSigmoidName );
		pooling = ifPooling.Ptr();
	} else {
		CPtr<CQrnnFPoolingLayer> fPooling = new CQrnnFPoolingLayer( MathEngine() );
		fPooling->SetReverse( reverse );
		pooling = fPooling.Ptr();
	}
	pooling->Connect( 0, updateActivationName );
	// there may be dropout after forget gate
	if( postDropoutLinear != nullptr ) {
		pooling->Connect( 1, *postDropoutLinear );
	} else {
		pooling->Connect( 1, *forgetSigmoid );
	}
	pooling->SetName( poolingName );
	AddLayer( *pooling );
	return pooling;
}

// Adds multiplication of 2 inputs
CPtr<CEltwiseMulLayer> CQrnnLayer::addMulLayer( CBaseLayer& first, CBaseLayer& second, const char* mulLayerName )
{
	CPtr<CEltwiseMulLayer> mulLayer = new CEltwiseMulLayer( MathEngine() );
	mulLayer->SetName( mulLayerName );
	mulLayer->Connect( 0, first );
	mulLayer->Connect( 1, second );
	AddLayer( *mulLayer );
	return mulLayer;
}

CPtr<CBaseLayer> CQrnnLayer::addBidirectionalMerge( CBaseLayer& first, CBaseLayer& second, const char* mergeName )
{
	static_assert( RM_Count == 4, "RM_Count != 4" );
	NeoAssert( recurrentMode == RM_BidirectionalConcat || recurrentMode == RM_BidirectionalSum );
	CPtr<CBaseLayer> mergeLayer;
	if( recurrentMode == RM_BidirectionalConcat ) {
		mergeLayer = new CConcatChannelsLayer( MathEngine() );
	} else {
		mergeLayer = new CEltwiseSumLayer( MathEngine() );
	}
	mergeLayer->SetName( mergeName );
	mergeLayer->Connect( 0, first );
	mergeLayer->Connect( 1, second );
	AddLayer( *mergeLayer );
	return mergeLayer;
}

// Adds input mapping for pooling's initial state
void CQrnnLayer::addInitialStateInputMapping( CBaseLayer& pooling, int inputMappingIndex )
{
	static_assert( PT_Count == 3, "PT_Count != 3" );
	const int initialStateInput = poolingType == PT_IfoPooling ? 3 : 2;
	SetInputMapping( inputMappingIndex, pooling, initialStateInput );
}

void CQrnnLayer::addDropout( float rate )
{
	NeoAssert( rate > 0.f );
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

	firstPooling->Connect( G_Forget, *postDropoutLinear );
	if( secondPooling != nullptr ) {
		secondPooling->Connect( G_Forget, *postDropoutLinear );
	}

	ForceReshape();
}

void CQrnnLayer::deleteDropout()
{
	DeleteLayer( preDropoutLinearName );
	DeleteLayer( *dropout );
	DeleteLayer( *postDropoutLinear );
	dropout = nullptr;
	postDropoutLinear = nullptr;
	firstPooling->Connect( G_Forget, *forgetSigmoid );
	if( secondPooling != nullptr ) {
		secondPooling->Connect( G_Forget, *forgetSigmoid );
	}
	ForceReshape();
}

void CQrnnLayer::rebuildLayer( int prevGateCount )
{
	const float dropoutRate = GetDropout();
	// Here the prevGateCount is used because current value of poolingType contains new value
	// And this new pooling may have different number of gates (if rebuildLayer was called from SetPoolingType)
	const int hiddenSize = timeConv->GetFilterCount() / prevGateCount;
	const int windowSize = timeConv->GetFilterSize();
	const int stride = timeConv->GetStride();
	const int padFront = timeConv->GetPaddingFront();
	const int padBack = timeConv->GetPaddingBack();

	DeleteAllLayers();

	buildLayer( dropoutRate, hiddenSize, windowSize, stride, padFront, padBack );
}

bool CQrnnLayer::isBidirectional() const
{
	static_assert( CQrnnLayer::RM_Count == 4, "CQrnnLayer::RM_Count != 4" );
	return recurrentMode == CQrnnLayer::RM_BidirectionalConcat || recurrentMode == CQrnnLayer::RM_BidirectionalSum;
}

// Returns number of gates used by this pooling
int CQrnnLayer::gateCount() const
{
	static_assert( PT_Count == 3, "PT_Count != 3" );
	switch( poolingType ) {
		case PT_FPooling:
			return 2;
		case PT_FoPooling:
			return 3;
		case PT_IfoPooling:
			return 4;
		case PT_Count:
		default:
			NeoAssert( false );
	}

	return -1;
}

CLayerWrapper<CQrnnLayer> Qrnn( CQrnnLayer::TPoolingType poolingType, CQrnnLayer::TRecurrentMode recurrentMode,
	int hiddenSize, int windowSize, int paddingFront, int paddingBack, float dropout, int stride,
	TActivationFunction activation )
{
	return CLayerWrapper<CQrnnLayer>( "", [=] ( CQrnnLayer* result ) {
		result->SetPoolingType( poolingType );
		result->SetRecurrentMode( recurrentMode );
		result->SetHiddenSize( hiddenSize );
		result->SetWindowSize( windowSize );
		result->SetStride( stride );
		result->SetPaddingFront( paddingFront );
		result->SetPaddingBack( paddingBack );
		result->SetDropout( dropout );
		result->SetActivation( activation );
	} );
}

// --------------------------------------------------------------------------------------------------------------------

static const int QrnnFPoolingLayerVersion = 0;

void CQrnnFPoolingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( QrnnFPoolingLayerVersion );
	CBaseLayer::Serialize( archive );
	archive.Serialize( reverse );
}

void CQrnnFPoolingLayer::Reshape()
{
	outputDescs[0] = inputDescs[0];
}

void CQrnnFPoolingLayer::RunOnce()
{
	const int sequenceLength = inputBlobs[0]->GetBatchLength();
	const int objectSize = inputBlobs[0]->GetDataSize() / sequenceLength;
	MathEngine().QrnnFPooling( reverse, sequenceLength, objectSize,
		inputBlobs[0]->GetData(), inputBlobs[1]->GetData(),
		inputBlobs.Size() == 2 ? CFloatHandle() : inputBlobs[2]->GetData(),
		outputBlobs[0]->GetData() );
}

void CQrnnFPoolingLayer::BackwardOnce()
{
	const int sequenceLength = inputBlobs[0]->GetBatchLength();
	const int objectSize = inputBlobs[0]->GetDataSize() / sequenceLength;
	MathEngine().QrnnFPoolingBackward( !reverse, sequenceLength, objectSize,
		inputBlobs[0]->GetData(), inputBlobs[1]->GetData(),
		inputBlobs.Size() == 2 ? CFloatHandle() : inputBlobs[2]->GetData(),
		outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[1]->GetData() );
}

// --------------------------------------------------------------------------------------------------------------------

static const int QrnnIfPoolingLayerVersion = 0;

void CQrnnIfPoolingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( QrnnIfPoolingLayerVersion );
	CBaseLayer::Serialize( archive );
	archive.Serialize( reverse );
}

void CQrnnIfPoolingLayer::Reshape()
{
	outputDescs[0] = inputDescs[0];
}

void CQrnnIfPoolingLayer::RunOnce()
{
	const int sequenceLength = inputBlobs[0]->GetBatchLength();
	const int objectSize = inputBlobs[0]->GetDataSize() / sequenceLength;
	MathEngine().QrnnIfPooling( reverse, sequenceLength, objectSize,
		inputBlobs[0]->GetData(), inputBlobs[1]->GetData(), inputBlobs[2]->GetData(),
		inputBlobs.Size() == 3 ? CFloatHandle() : inputBlobs[3]->GetData(),
		outputBlobs[0]->GetData() );
}

void CQrnnIfPoolingLayer::BackwardOnce()
{
	const int sequenceLength = inputBlobs[0]->GetBatchLength();
	const int objectSize = inputBlobs[0]->GetDataSize() / sequenceLength;
	MathEngine().QrnnIfPoolingBackward( !reverse, sequenceLength, objectSize,
		inputBlobs[0]->GetData(), inputBlobs[1]->GetData(), inputBlobs[2]->GetData(),
		inputBlobs.Size() == 3 ? CFloatHandle() : inputBlobs[3]->GetData(),
		outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[1]->GetData(), inputDiffBlobs[2]->GetData() );
}

} // namespace NeoML
