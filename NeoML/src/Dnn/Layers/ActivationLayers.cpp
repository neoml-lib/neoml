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
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

template<class CActivationLayer>
static CPtr<CBaseLayer> createActivationWithParam( IMathEngine& mathEngine, const CActivationDesc& desc )
{
	CPtr<CActivationLayer> result = FINE_DEBUG_NEW CActivationLayer( mathEngine );
	if( desc.HasParam() ) {
		result->ApplyParam( desc.GetParam<typename CActivationLayer::CParam>() );
	}
	return result.Ptr();
}

CPtr<CBaseLayer> CreateActivationLayer( IMathEngine& mathEngine, const CActivationDesc& desc )
{
	static_assert( AF_Count == 15, "AF_Count != 15" );
	switch( desc.GetType() ) {
		case AF_Linear:
			return createActivationWithParam<CLinearLayer>( mathEngine, desc );
		case AF_ELU:
			return createActivationWithParam<CELULayer>( mathEngine, desc );
		case AF_ReLU:
			return createActivationWithParam<CReLULayer>( mathEngine, desc );
		case AF_LeakyReLU:
			return createActivationWithParam<CLeakyReLULayer>( mathEngine, desc );
		case AF_Abs:
			return FINE_DEBUG_NEW CAbsLayer( mathEngine );
		case AF_Sigmoid:
			return FINE_DEBUG_NEW CSigmoidLayer( mathEngine );
		case AF_Tanh:
			return FINE_DEBUG_NEW CTanhLayer( mathEngine );
		case AF_HardTanh:
			return FINE_DEBUG_NEW CHardTanhLayer( mathEngine );
		case AF_HardSigmoid:
			return createActivationWithParam<CHardSigmoidLayer>( mathEngine, desc );
		case AF_Power:
			return createActivationWithParam<CPowerLayer>( mathEngine, desc );
		case AF_HSwish:
			return FINE_DEBUG_NEW CHSwishLayer( mathEngine );
		case AF_GELU:
			return createActivationWithParam<CGELULayer>( mathEngine, desc );
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

void StoreActivationDesc( const CActivationDesc& desc, CArchive& archive )
{
	TActivationFunction type = desc.GetType();

	archive.SerializeEnum( type );
	switch( type ) {
		case AF_Linear:
			archive << desc.GetParam<CLinearLayer::CParam>().Multiplier
				<< desc.GetParam<CLinearLayer::CParam>().FreeTerm;
			break;
		case AF_ELU:
			archive << desc.GetParam<CELULayer::CParam>().Alpha;
			break;
		case AF_ReLU:
			archive << desc.GetParam<CReLULayer::CParam>().UpperThreshold;
			break;
		case AF_LeakyReLU:
			archive << desc.GetParam<CLeakyReLULayer::CParam>().Alpha;
			break;
		case AF_HardSigmoid:
			archive << desc.GetParam<CHardSigmoidLayer::CParam>().Slope
				<< desc.GetParam<CHardSigmoidLayer::CParam>().Bias;
			break;
		case AF_Power:
			archive << desc.GetParam<CPowerLayer::CParam>().Exponent;
			break;
		case AF_GELU:
			archive << static_cast<int>( desc.GetParam<CGELULayer::CParam>().Mode );
			break;
		case AF_Abs:
		case AF_Sigmoid:
		case AF_Tanh:
		case AF_HardTanh:
		case AF_HSwish:
		case AF_Exp:
		case AF_Log:
		case AF_Erf:
			break;
		default:
			NeoAssert( false );
	}
}

CActivationDesc LoadActivationDesc( CArchive& archive )
{
	TActivationFunction type = AF_Count;
	archive.SerializeEnum( type );
	CActivationDesc result = CActivationDesc( type );

	switch( type ) {
		case AF_Linear:
		{
			CLinearLayer::CParam param;
			archive >> param.Multiplier >> param.FreeTerm;
			result.SetParam( param );
			break;
		}
		case AF_ELU:
		{
			CELULayer::CParam param;
			archive >> param.Alpha;
			result.SetParam( param );
			break;
		}
		case AF_ReLU:
		{
			CReLULayer::CParam param;
			archive >> param.UpperThreshold;
			result.SetParam( param );
			break;
		}
		case AF_LeakyReLU:
		{
			CLeakyReLULayer::CParam param;
			archive >> param.Alpha;
			result.SetParam( param );
			break;
		}
		case AF_HardSigmoid:
		{
			CHardSigmoidLayer::CParam param;
			archive >> param.Slope >> param.Bias;
			result.SetParam( param );
			break;
		}
		case AF_Power:
		{
			CPowerLayer::CParam param;
			archive >> param.Exponent;
			result.SetParam( param );
			break;
		}
		case AF_GELU:
		{
			CGELULayer::CParam param;
			int intMode = 0;
			archive >> intMode;
			param.Mode = static_cast<CGELULayer::TCalculationMode>( intMode );
			result.SetParam( param );
			break;
		}
		case AF_Abs:
		case AF_Sigmoid:
		case AF_Tanh:
		case AF_HardTanh:
		case AF_HSwish:
		case AF_Exp:
		case AF_Log:
		case AF_Erf:
			break;
		default:
			NeoAssert( false );
	}

	return result;
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
CLinearLayer::CLinearLayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CCnnLinearLayer" )
{
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

CActivationDesc CLinearLayer::GetDesc() const
{
	CParam param{ multiplier, freeTerm };
	return { AF_Linear, param };
}

void CLinearLayer::RunOnce()
{
	for( int i = 0; i < inputBlobs.Size(); ++i ) {
		const int dataSize = outputBlobs[i]->GetDataSize();

		if( inputBlobs[i]->GetDataType() == CT_Float ) {
			linearRunOnce( inputBlobs[i]->GetData<const float>(), multiplier, freeTerm, dataSize, outputBlobs[i]->GetData() );
		} else {
			linearRunOnce( inputBlobs[i]->GetData<const int>(), static_cast<int>( multiplier ),
				static_cast< int >( freeTerm ), dataSize, outputBlobs[i]->GetData<int>() );
		}
	}
}

void CLinearLayer::BackwardOnce()
{
	for( int i = 0; i < outputBlobs.Size(); ++i ) {
		CConstFloatHandle outputDiffPtr = outputDiffBlobs[i]->GetData();
		CFloatHandle inputDiffPtr = inputDiffBlobs[i]->GetData();
		int dataSize = outputDiffBlobs[i]->GetDataSize();

		if( multiplier != 1.f ) {
			CFloatHandleStackVar multiplierValue( MathEngine() );
			multiplierValue.SetValue( multiplier );
			MathEngine().VectorMultiply( outputDiffPtr, inputDiffPtr, dataSize, multiplierValue );
		} else if( outputDiffPtr != inputDiffPtr ) {
			MathEngine().VectorCopy( inputDiffPtr, outputDiffPtr, dataSize );
		}
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
	SetAlpha( DefaultAlpha );
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

CActivationDesc CELULayer::GetDesc() const
{
	return { AF_ELU, CParam{ GetAlpha() } };
}

void CELULayer::RunOnce()
{
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

CReLULayer::CReLULayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CCnnReLULayer" ),
	upperThreshold( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) )
{
	SetUpperThreshold( DefaultUpperThreshold );
}

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

CActivationDesc CReLULayer::GetDesc() const
{
	return { AF_ReLU, CParam{ GetUpperThreshold() } };
}

void CReLULayer::RunOnce()
{
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
	SetAlpha( DefaultAlpha );
}

CActivationDesc CLeakyReLULayer::GetDesc() const
{
	return { AF_LeakyReLU, CParam{ GetAlpha() } };
}

void CLeakyReLULayer::RunOnce()
{
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

CActivationDesc CHSwishLayer::GetDesc() const
{
	return { AF_HSwish };
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

CActivationDesc CAbsLayer::GetDesc() const
{
	return { AF_Abs };
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

CActivationDesc CSigmoidLayer::GetDesc() const
{
	return { AF_Sigmoid };
}

void CSigmoidLayer::RunOnce()
{
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

CActivationDesc CTanhLayer::GetDesc() const
{
	return { AF_Tanh };
}

void CTanhLayer::RunOnce()
{
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

CActivationDesc CHardTanhLayer::GetDesc() const
{
	return { AF_HardTanh };
}

void CHardTanhLayer::RunOnce()
{
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
	SetSlope( DefaultSlope );
	paramBlobs.Add( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) );
	SetBias( DefaultBias );
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

CActivationDesc CHardSigmoidLayer::GetDesc() const
{
	return { AF_HardSigmoid, CParam{ GetSlope(), GetBias() } };
}

void CHardSigmoidLayer::RunOnce()
{
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

CPowerLayer::CPowerLayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CCnnPowerLayer" )
{
}

void CPowerLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( PowerLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );
	archive.Serialize( exponent );
}

CActivationDesc CPowerLayer::GetDesc() const
{
	return { AF_Power, CParam{ exponent } };
}

void CPowerLayer::RunOnce()
{
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

CActivationDesc CExpLayer::GetDesc() const
{
	return { AF_Exp };
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

CActivationDesc CLogLayer::GetDesc() const
{
	return { AF_Log };
}

void CLogLayer::RunOnce()
{
	MathEngine().VectorLog( inputBlobs[0]->GetData(), outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize() );
}

void CLogLayer::BackwardOnce()
{
	if( inputBlobs[0].Ptr() == outputBlobs[0].Ptr() || inputBlobs[0].Ptr() == nullptr ) {
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

CActivationDesc CErfLayer::GetDesc() const
{
	return { AF_Erf };
}

void CErfLayer::Reshape()
{
	CheckInput1();
	CheckOutputs();
	CheckLayerArchitecture( inputDescs[0].GetDataType() == CT_Float, "Layer works only with float data" );
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
