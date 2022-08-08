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

#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/GELULayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CPtr<CBaseLayer> CreateActivationLayer( IMathEngine& mathEngine, TActivationFunction type )
{
	static_assert( AF_Count == 15, "AF_Count != 15" );
	switch( type ) {
		case AF_Linear:
			return FINE_DEBUG_NEW CLinearLayer( mathEngine );
		case AF_ELU:
			return FINE_DEBUG_NEW CELULayer( mathEngine );
		case AF_ReLU:
			return FINE_DEBUG_NEW CReLULayer( mathEngine );
		case AF_LeakyReLU:
			return FINE_DEBUG_NEW CLeakyReLULayer( mathEngine );
		case AF_Abs:
			return FINE_DEBUG_NEW CAbsLayer( mathEngine );
		case AF_Sigmoid:
			return FINE_DEBUG_NEW CSigmoidLayer( mathEngine );
		case AF_Tanh:
			return FINE_DEBUG_NEW CTanhLayer( mathEngine );
		case AF_HardTanh:
			return FINE_DEBUG_NEW CHardTanhLayer( mathEngine );
		case AF_HardSigmoid:
			return FINE_DEBUG_NEW CHardSigmoidLayer( mathEngine );
		case AF_Power:
			return FINE_DEBUG_NEW CPowerLayer( mathEngine );
		case AF_HSwish:
			return FINE_DEBUG_NEW CHSwishLayer( mathEngine );
		case AF_GELU:
			return FINE_DEBUG_NEW CGELULayer( mathEngine );
		case AF_Exp:
			return FINE_DEBUG_NEW CExpLayer( mathEngine );
		case AF_Log:
			return FINE_DEBUG_NEW CLogLayer( mathEngine );
		case AF_Erf:
			return FINE_DEBUG_NEW CErfLayer( mathEngine );
		default:
			NeoAssert( false );
	}
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
CLinearLayer::CLinearLayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CCnnLinearLayer" )
{
	SetMultiplier(1.f);
	SetFreeTerm(0);
}

template<class T>
static void linearRunOnce( const CTypedMemoryHandle<const T>& input, T multiplier, T freeTerm, int dataSize,
	const CTypedMemoryHandle<T>& output )
{
	IMathEngine& mathEngine = *input.GetMathEngine();
	CTypedMemoryHandle<const T> currInput = input;

	if( multiplier != static_cast<T>( 1 ) ) {
		CMemoryHandleStackVar<T> multiplierVar( mathEngine );
		multiplierVar.SetValue( multiplier );
		mathEngine.VectorMultiply( currInput, output, dataSize, multiplierVar );
		currInput = output;
	}

	if( freeTerm != static_cast< T >( 0 ) ) {
		CMemoryHandleStackVar<T> freeTermVar( mathEngine );
		freeTermVar.SetValue( freeTerm );
		mathEngine.VectorAddValue( currInput, output, dataSize, freeTermVar );
		currInput = output;
	}

	if( currInput != output ) {
		mathEngine.VectorCopy( output, currInput, dataSize );
	}
}

void CLinearLayer::RunOnce()
{
	CheckInput1();

	const int dataSize = outputBlobs[0]->GetDataSize();

	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		linearRunOnce( inputBlobs[0]->GetData<const float>(), multiplier, freeTerm, dataSize, outputBlobs[0]->GetData() );
	} else {
		linearRunOnce( inputBlobs[0]->GetData<const int>(), static_cast<int>( multiplier ),
			static_cast<int>( freeTerm ), dataSize, outputBlobs[0]->GetData<int>() );
	}
}

void CLinearLayer::BackwardOnce()
{
	CConstFloatHandle outputDiffPtr = outputDiffBlobs[0]->GetData();
	CFloatHandle inputDiffPtr = inputDiffBlobs[0]->GetData();
	int dataSize = outputBlobs[0]->GetDataSize();

	if( multiplier != 1.f ) {
		CFloatHandleStackVar multiplierValue( MathEngine() );
		multiplierValue.SetValue( multiplier );
		MathEngine().VectorMultiply( outputDiffPtr, inputDiffPtr, dataSize, multiplierValue );
	} else if( outputDiffPtr != inputDiffPtr ) {
		MathEngine().VectorCopy( inputDiffPtr, outputDiffPtr, dataSize );
	}
}

static const int LinearLayerVersion = 2000;

void CLinearLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( LinearLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		archive << GetMultiplier() << GetFreeTerm();
	} else if( archive.IsLoading() ) {
		float temp, tempFreeTerm;
		archive >> temp >> tempFreeTerm;
		SetMultiplier(temp);
		SetFreeTerm(tempFreeTerm);
	} else {
		NeoAssert( false );
	}
}

CLayerWrapper<CLinearLayer> Linear( float multiplier, float freeTerm )
{
	return CLayerWrapper<CLinearLayer>( "Linear", [=]( CLinearLayer* result ) {
		result->SetMultiplier( multiplier );
		result->SetFreeTerm( freeTerm );
	} );
}

//---------------------------------------------------------------------------------------------------

CELULayer::CELULayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CCnnELULayer" )
{
	paramBlobs.Add( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) );
	SetAlpha( 0.01f );
}

static const int ELULayerVersion = 2000;

void CELULayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ELULayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );
}

float CELULayer::GetAlpha() const
{
	return paramBlobs[0]->GetData().GetValue();
}

void CELULayer::SetAlpha( float alpha )
{
	paramBlobs[0]->GetData().SetValue( alpha );
}

void CELULayer::RunOnce()
{
	CheckInput1();

	MathEngine().VectorELU( inputBlobs[0]->GetData(), outputBlobs[0]->GetData(),
		outputBlobs[0]->GetDataSize(), paramBlobs[0]->GetData() );
}

void CELULayer::BackwardOnce()
{
	MathEngine().VectorELUDiffOp( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize(), paramBlobs[0]->GetData() );
}

CLayerWrapper<CELULayer> Elu( float alpha )
{
	return CLayerWrapper<CELULayer>( "Elu", [=]( CELULayer* result ) {
		result->SetAlpha( alpha );
	} );
}

//---------------------------------------------------------------------------------------------------

static const int ReLULayerVersion = 2000;

void CReLULayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ReLULayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		archive << GetUpperThreshold();
	} else if( archive.IsLoading() ) {
		float threshold = 0;
		archive >> threshold;
		SetUpperThreshold(threshold);
	} else {
		NeoAssert( false );
	}
}

void CReLULayer::RunOnce()
{
	CheckInput1();

	CConstFloatHandle inputPtr = inputBlobs[0]->GetData();
	CFloatHandle outputPtr = outputBlobs[0]->GetData();
	int dataSize = outputBlobs[0]->GetDataSize();
	
	MathEngine().VectorReLU( inputPtr, outputPtr, dataSize, upperThreshold->GetData() );
}

void CReLULayer::BackwardOnce()
{
	MathEngine().VectorReLUDiffOp( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize(), upperThreshold->GetData() );
}

CLayerWrapper<CReLULayer> Relu( float threshold )
{
	return CLayerWrapper<CReLULayer>( "Relu", [=] ( CReLULayer* result ) {
		result->SetUpperThreshold( threshold );
	} );
}

//---------------------------------------------------------------------------------------------------

static const int LeakyReLULayerVersion = 2000;

void CLeakyReLULayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( LeakyReLULayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );
}

float CLeakyReLULayer::GetAlpha() const
{
	return paramBlobs[0]->GetData().GetValue();
}

void CLeakyReLULayer::SetAlpha( float alpha )
{
	paramBlobs[0]->GetData().SetValue( alpha );
}

CLeakyReLULayer::CLeakyReLULayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CCnnLeakyReLULayer" )
{
	paramBlobs.Add( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) );
	SetAlpha( 0.01f );
}

void CLeakyReLULayer::RunOnce()
{
	CheckInput1();
	CConstFloatHandle inputPtr = inputBlobs[0]->GetData();
	CConstFloatHandle alpha = paramBlobs[0]->GetData();
	CFloatHandle outputPtr = outputBlobs[0]->GetData();
	int dataSize = outputBlobs[0]->GetDataSize();

	MathEngine().VectorLeakyReLU( inputPtr, outputPtr, dataSize, alpha );
}

void CLeakyReLULayer::BackwardOnce()
{
	MathEngine().VectorLeakyReLUDiffOp( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize(), paramBlobs[0]->GetData() );
}

CLayerWrapper<CLeakyReLULayer> LeakyRelu( float alpha )
{
	return CLayerWrapper<CLeakyReLULayer>( "LeakyRelu", [=]( CLeakyReLULayer* result ) {
		result->SetAlpha( alpha );
	} );
}

//---------------------------------------------------------------------------------------------------

static const int HSwishLayerVersion = 2000;

void CHSwishLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( HSwishLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

void CHSwishLayer::Reshape()
{
	CheckInput1();
	CheckOutputs();
	outputDescs[0] = inputDescs[0];
}

void CHSwishLayer::RunOnce()
{
	CConstFloatHandle inputPtr = inputBlobs[0]->GetData();
	CFloatHandle outputPtr = outputBlobs[0]->GetData();
	const int dataSize = inputBlobs[0]->GetDataSize();

	MathEngine().VectorHSwish( inputPtr, outputPtr, dataSize );
}

void CHSwishLayer::BackwardOnce()
{
	MathEngine().VectorHSwishDiff( inputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize() );
}

CLayerWrapper<CHSwishLayer> HSwish()
{
	return CLayerWrapper<CHSwishLayer>( "HSwish" );
}

//---------------------------------------------------------------------------------------------------

static const int AbsLayerVersion = 2000;

void CAbsLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( AbsLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

void CAbsLayer::Reshape()
{
	CheckInput1();
	CheckOutputs();
	outputDescs[0] = inputDescs[0];
}

void CAbsLayer::RunOnce()
{
	CConstFloatHandle inputPtr = inputBlobs[0]->GetData();
	CFloatHandle outputPtr = outputBlobs[0]->GetData();
	int dataSize = inputBlobs[0]->GetDataSize();

	MathEngine().VectorAbs(inputPtr, outputPtr, dataSize);
}

void CAbsLayer::BackwardOnce()
{
	MathEngine().VectorAbsDiff(inputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize());
}

CLayerWrapper<CAbsLayer> Abs()
{
	return CLayerWrapper<CAbsLayer>( "Abs" );
}

//---------------------------------------------------------------------------------------------------

static const int SigmoidLayerVersion = 2000;

void CSigmoidLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( SigmoidLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );
}

void CSigmoidLayer::RunOnce()
{
	CheckInput1();

	MathEngine().VectorSigmoid(inputBlobs[0]->GetData(), outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize());
}

void CSigmoidLayer::BackwardOnce()
{
	MathEngine().VectorSigmoidDiffOp(outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize());
}

CLayerWrapper<CSigmoidLayer> Sigmoid()
{
	return CLayerWrapper<CSigmoidLayer>( "Sigmoid" );
}

//---------------------------------------------------------------------------------------------------

static const int TanhLayerVersion = 2000;

void CTanhLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( TanhLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );
}

void CTanhLayer::RunOnce()
{
	CheckInput1();

	MathEngine().VectorTanh(inputBlobs[0]->GetData(), outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize());
}

void CTanhLayer::BackwardOnce()
{
	MathEngine().VectorTanhDiffOp(outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize());
}

CLayerWrapper<CTanhLayer> Tanh()
{
	return CLayerWrapper<CTanhLayer>( "Tanh" );
}

//---------------------------------------------------------------------------------------------------

static const int HardTanhLayerVersion = 2000;

void CHardTanhLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( HardTanhLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );
}

void CHardTanhLayer::RunOnce()
{
	CheckInput1();

	MathEngine().VectorHardTanh(inputBlobs[0]->GetData(), outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize());
}

void CHardTanhLayer::BackwardOnce()
{
	MathEngine().VectorHardTanhDiffOp(outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize());
}

CLayerWrapper<CHardTanhLayer> HardTanh()
{
	return CLayerWrapper<CHardTanhLayer>( "HardTanh" );
}

//---------------------------------------------------------------------------------------------------

static const int HardSigmoidLayerVersion = 2001;

void CHardSigmoidLayer::setDefaultParamBlobs( IMathEngine& mathEngine )
{
	paramBlobs.Add( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) );
	SetSlope( 0.5f );
	paramBlobs.Add( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) );
	SetBias( 0.5f );
}

void CHardSigmoidLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( HardSigmoidLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );

	if( version <= 2000 && archive.IsLoading() ) {
		setDefaultParamBlobs( MathEngine() );
	}
}

CHardSigmoidLayer::CHardSigmoidLayer( IMathEngine& mathEngine ) : CBaseInPlaceLayer( mathEngine, "CCnnHardSigmoidLayer" )
{
	setDefaultParamBlobs( mathEngine );
}

void CHardSigmoidLayer::RunOnce()
{
	CheckInput1();

	MathEngine().VectorHardSigmoid( inputBlobs[0]->GetData(), outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize(),
		paramBlobs[0]->GetData(), paramBlobs[1]->GetData() );
}

void CHardSigmoidLayer::BackwardOnce()
{
	MathEngine().VectorHardSigmoidDiffOp( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize(), paramBlobs[0]->GetData(), paramBlobs[1]->GetData() );
}

CLayerWrapper<CHardSigmoidLayer> HardSigmoid( float slope, float bias )
{
	return CLayerWrapper<CHardSigmoidLayer>( "HardSigmoid", [=]( CHardSigmoidLayer* result ) {
		result->SetSlope( slope );
		result->SetBias( bias );
	} );
}

//---------------------------------------------------------------------------------------------------

static const int PowerLayerVersion = 2000;

void CPowerLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( PowerLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );
	archive.Serialize( exponent );
}

void CPowerLayer::RunOnce()
{
	CheckInput1();

	MathEngine().VectorPower(exponent, inputBlobs[0]->GetData(), outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize());
}

void CPowerLayer::BackwardOnce()
{
	MathEngine().VectorPowerDiffOp(exponent, outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize());
}

CLayerWrapper<CPowerLayer> Power( float exponent )
{
	return CLayerWrapper<CPowerLayer>( "Power", [=]( CPowerLayer* result ) {
		result->SetExponent( exponent );
	} );
}

//---------------------------------------------------------------------------------------------------

static const int ExpLayerVersion = 0;

void CExpLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ExpLayerVersion );
	CBaseInPlaceLayer::Serialize( archive );
}

void CExpLayer::RunOnce()
{
	MathEngine().VectorExp( inputBlobs[0]->GetData(), outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize() );
}

void CExpLayer::BackwardOnce()
{
	MathEngine().VectorEltwiseMultiply( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), outputBlobs[0]->GetDataSize() );
}

CLayerWrapper<CExpLayer> Exp()
{
	return CLayerWrapper<CExpLayer>( "Exp" );
}

//---------------------------------------------------------------------------------------------------

static const int LogLayerVersion = 0;

void CLogLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( LogLayerVersion );
	CBaseInPlaceLayer::Serialize( archive );
}

void CLogLayer::RunOnce()
{
	MathEngine().VectorLog( inputBlobs[0]->GetData(), outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize() );
}

void CLogLayer::BackwardOnce()
{
	if( inputBlobs[0].Ptr() == outputBlobs[0].Ptr() ) {
		MathEngine().VectorExp( outputBlobs[0]->GetData(), inputDiffBlobs[0]->GetData(), outputBlobs[0]->GetDataSize());
		MathEngine().VectorEltwiseDivide( outputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetData(),
			inputDiffBlobs[0]->GetData(), outputBlobs[0]->GetDataSize() );
	} else {
		MathEngine().VectorEltwiseDivide( outputDiffBlobs[0]->GetData(), inputBlobs[0]->GetData(),
			inputDiffBlobs[0]->GetData(), inputBlobs[0]->GetDataSize() );
	}
}

CLayerWrapper<CLogLayer> Log()
{
	return CLayerWrapper<CLogLayer>( "Log" );
}

//---------------------------------------------------------------------------------------------------

static const int ErfLayerVersion = 0;

void CErfLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ErfLayerVersion );
	CBaseLayer::Serialize( archive );
}

void CErfLayer::Reshape()
{
	CheckInput1();
	CheckOutputs();
	CheckArchitecture( inputDescs[0].GetDataType() == CT_Float, GetName(), "Layer works only with float data" );
	outputDescs[0] = inputDescs[0];
}

void CErfLayer::RunOnce()
{
	MathEngine().VectorErf( inputBlobs[0]->GetData(), outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize() );
}

void CErfLayer::BackwardOnce()
{
	const int dataSize = inputBlobs[0]->GetDataSize();
	CFloatHandle inputDiff = inputDiffBlobs[0]->GetData();
	MathEngine().VectorNegMultiply( inputBlobs[0]->GetData(), inputBlobs[0]->GetData(), dataSize, inputDiff );
	MathEngine().VectorExp( inputDiff, inputDiff, dataSize );
	CFloatHandleStackVar mult( MathEngine() );
	mult.SetValue( 1.1283791671f ); // 2 / sqrt( pi )
	MathEngine().VectorMultiply( inputDiff, inputDiff, dataSize, mult );
	MathEngine().VectorEltwiseMultiply( inputDiff, outputDiffBlobs[0]->GetData(), inputDiff, dataSize );
}

CLayerWrapper<CErfLayer> Erf()
{
	return CLayerWrapper<CErfLayer>( "Erf" );
}

} // namespace NeoML
