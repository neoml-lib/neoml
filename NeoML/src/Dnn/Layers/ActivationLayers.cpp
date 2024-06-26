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

//---------------------------------------------------------------------------------------------------

template<class T>
static void linearRunOnce( const CTypedMemoryHandle<const T>& input, T multiplier, T freeTerm, int dataSize,
	const CTypedMemoryHandle<T>& output )
{
	IMathEngine& mathEngine = *input.GetMathEngine();
	CTypedMemoryHandle<const T> currInput = input;

	if( multiplier != static_cast<T>( 1 ) ) {
		mathEngine.VectorMultiply( currInput, output, dataSize, static_cast<T>( multiplier ) );
		currInput = output;
	}

	if( freeTerm != static_cast<T>( 0 ) ) {
		mathEngine.VectorAddValue( currInput, output, dataSize, static_cast<T>( freeTerm ) );
		currInput = output;
	}

	if( currInput != output ) {
		mathEngine.VectorCopy( output, currInput, dataSize );
	}
}

void CLinearLayer::RunOnce()
{
	const int dataSize = outputBlobs[0]->GetDataSize();
	if( inputBlobs[0]->GetDataType() == CT_Float ) {
		linearRunOnce( inputBlobs[0]->GetData<const float>(), multiplier,
			freeTerm, dataSize, outputBlobs[0]->GetData() );
	} else {
		linearRunOnce( inputBlobs[0]->GetData<const int>(), static_cast<int>( multiplier ),
			static_cast<int>( freeTerm ), dataSize, outputBlobs[0]->GetData<int>() );
	}
}

void CLinearLayer::BackwardOnce()
{
	CConstFloatHandle outputDiffPtr = outputDiffBlobs[0]->GetData();
	CFloatHandle inputDiffPtr = inputDiffBlobs[0]->GetData();
	const int dataSize = outputDiffBlobs[0]->GetDataSize();

	if( multiplier != 1.f ) {
		MathEngine().VectorMultiply( outputDiffPtr, inputDiffPtr, dataSize, multiplier );
	} else if( outputDiffPtr != inputDiffPtr ) {
		MathEngine().VectorCopy( inputDiffPtr, outputDiffPtr, dataSize );
	}
}

constexpr int linearLayerVersion = 2000;

void CLinearLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( linearLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );

	archive.Serialize( multiplier );
	archive.Serialize( freeTerm );
}

CLayerWrapper<CLinearLayer> Linear( float multiplier, float freeTerm )
{
	return CLayerWrapper<CLinearLayer>( "Linear", [=]( CLinearLayer* result ) {
		result->SetMultiplier( multiplier );
		result->SetFreeTerm( freeTerm );
	} );
}

//---------------------------------------------------------------------------------------------------

constexpr int eluLayerVersion = 2001;

void CELULayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( eluLayerVersion, CDnn::ArchiveMinSupportedVersion );

	if( version < 2001 ) {
		NeoAssert( archive.IsLoading() );
		paramBlobs.Add( CDnnBlob::CreateVector( MathEngine(), CT_Float, 1 ) );
	}
	CBaseInPlaceLayer::Serialize( archive );

	if( version < 2001 ) {
		SetAlpha( paramBlobs[0]->GetData().GetValue() );
		paramBlobs.SetSize( 0 );
	} else {
		archive.Serialize( alpha );
	}
}

void CELULayer::RunOnce()
{
	MathEngine().VectorELU( inputBlobs[0]->GetData(), outputBlobs[0]->GetData(),
		outputBlobs[0]->GetDataSize(), alpha );
}

void CELULayer::BackwardOnce()
{
	MathEngine().VectorELUDiffOp( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize(), alpha );
}

CLayerWrapper<CELULayer> Elu( float alpha )
{
	return CLayerWrapper<CELULayer>( "Elu", [=]( CELULayer* result ) {
		result->SetAlpha( alpha );
	} );
}

//---------------------------------------------------------------------------------------------------

constexpr int reluLayerVersion = 2000;

void CReLULayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( reluLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );

	archive.Serialize( upperThreshold );
}

void CReLULayer::RunOnce()
{
	CConstFloatHandle inputPtr = inputBlobs[0]->GetData();
	CFloatHandle outputPtr = outputBlobs[0]->GetData();
	const int dataSize = outputBlobs[0]->GetDataSize();
	
	MathEngine().VectorReLU( inputPtr, outputPtr, dataSize, upperThreshold );
}

void CReLULayer::BackwardOnce()
{
	MathEngine().VectorReLUDiffOp( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize(), upperThreshold );
}

CLayerWrapper<CReLULayer> Relu( float threshold )
{
	return CLayerWrapper<CReLULayer>( "Relu", [=] ( CReLULayer* result ) {
		result->SetUpperThreshold( threshold );
	} );
}

//---------------------------------------------------------------------------------------------------

constexpr int leakyReluLayerVersion = 2001;

void CLeakyReLULayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( leakyReluLayerVersion, CDnn::ArchiveMinSupportedVersion );

	if( version < 2001 ) {
		NeoAssert( archive.IsLoading() );
		paramBlobs.Add( CDnnBlob::CreateVector( MathEngine(), CT_Float, 1 ) );
	}
	CBaseInPlaceLayer::Serialize( archive );

	if( version < 2001 ) {
		SetAlpha( paramBlobs[0]->GetData().GetValue() );
		paramBlobs.SetSize( 0 );
	} else {
		archive.Serialize( alpha );
	}
}

void CLeakyReLULayer::RunOnce()
{
	CConstFloatHandle inputPtr = inputBlobs[0]->GetData();
	CFloatHandle outputPtr = outputBlobs[0]->GetData();
	const int dataSize = outputBlobs[0]->GetDataSize();

	MathEngine().VectorLeakyReLU( inputPtr, outputPtr, dataSize, alpha );
}

void CLeakyReLULayer::BackwardOnce()
{
	MathEngine().VectorLeakyReLUDiffOp( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize(), alpha );
}

CLayerWrapper<CLeakyReLULayer> LeakyRelu( float alpha )
{
	return CLayerWrapper<CLeakyReLULayer>( "LeakyRelu", [=]( CLeakyReLULayer* result ) {
		result->SetAlpha( alpha );
	} );
}

//---------------------------------------------------------------------------------------------------

constexpr int hswishLayerVersion = 2000;

void CHSwishLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( hswishLayerVersion, CDnn::ArchiveMinSupportedVersion );
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

constexpr int absLayerVersion = 2000;

void CAbsLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( absLayerVersion, CDnn::ArchiveMinSupportedVersion );
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
	const int dataSize = inputBlobs[0]->GetDataSize();

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

constexpr int sigmoidLayerVersion = 2000;

void CSigmoidLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( sigmoidLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );
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

constexpr int tanhLayerVersion = 2000;

void CTanhLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( tanhLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );
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

constexpr int hardTanhLayerVersion = 2000;

void CHardTanhLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( hardTanhLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );
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

constexpr int HardSigmoidLayerVersion = 2002;

void CHardSigmoidLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( HardSigmoidLayerVersion, CDnn::ArchiveMinSupportedVersion );

	if( version == 2001 ) {
		NeoAssert( archive.IsLoading() );
		paramBlobs.Add( CDnnBlob::CreateVector( MathEngine(), CT_Float, 1 ) );
		paramBlobs.Add( CDnnBlob::CreateVector( MathEngine(), CT_Float, 1 ) );
	}
	CBaseInPlaceLayer::Serialize( archive );

	if( version >= 2002 ) {
		archive.Serialize( slope );
		archive.Serialize( bias );
	} else if( version == 2001 ) {
		SetSlope( paramBlobs[0]->GetData().GetValue() );
		SetBias( paramBlobs[1]->GetData().GetValue() );
		paramBlobs.SetSize( 0 );
	} else {
		NeoAssert( archive.IsLoading() );
		SetSlope( CParam::DefaultSlope );
		SetBias( CParam::DefaultBias );
	}
}

void CHardSigmoidLayer::RunOnce()
{
	MathEngine().VectorHardSigmoid( inputBlobs[0]->GetData(), outputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize(),
		slope, bias );
}

void CHardSigmoidLayer::BackwardOnce()
{
	MathEngine().VectorHardSigmoidDiffOp( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetDataSize(), slope, bias );
}

CLayerWrapper<CHardSigmoidLayer> HardSigmoid( float slope, float bias )
{
	return CLayerWrapper<CHardSigmoidLayer>( "HardSigmoid", [=]( CHardSigmoidLayer* result ) {
		result->SetSlope( slope );
		result->SetBias( bias );
	} );
}

//---------------------------------------------------------------------------------------------------

constexpr int powerLayerVersion = 2000;

void CPowerLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( powerLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );
	archive.Serialize( exponent );
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

constexpr int expLayerVersion = 0;

void CExpLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( expLayerVersion );
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

constexpr int logLayerVersion = 0;

void CLogLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( logLayerVersion );
	CBaseInPlaceLayer::Serialize( archive );
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

constexpr int erfLayerVersion = 0;

void CErfLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( erfLayerVersion );
	CBaseLayer::Serialize( archive );
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
	MathEngine().VectorMultiply( inputDiff, inputDiff, dataSize, /*2/sqrt(pi)*/1.1283791671f );
	MathEngine().VectorEltwiseMultiply( inputDiff, outputDiffBlobs[0]->GetData(), inputDiff, dataSize );
}

CLayerWrapper<CErfLayer> Erf()
{
	return CLayerWrapper<CErfLayer>( "Erf" );
}

} // namespace NeoML
