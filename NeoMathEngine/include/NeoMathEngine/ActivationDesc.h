/* Copyright © 2017-2023 ABBYY

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

#pragma once

#include <type_traits>

#include <NeoMathEngine/NeoMathEngineException.h>

namespace NeoML {

// Supported activation functions
enum TActivationFunction {
	AF_Linear,
	AF_ELU,
	AF_ReLU,
	AF_LeakyReLU,
	AF_Abs,
	AF_Sigmoid,
	AF_Tanh,
	AF_HardTanh,
	AF_HardSigmoid,
	AF_Power,
	AF_HSwish,
	AF_GELU,
	AF_Exp,
	AF_Log,
	AF_Erf,

	AF_Count
};

// Parameters of linear activation
struct NEOMATHENGINE_API CLinearActivationParam final {
	static constexpr float DefaultMultiplier = 1.f;
	static constexpr float DefaultFreeTerm = 0.f;
	float Multiplier = DefaultMultiplier;
	float FreeTerm = DefaultFreeTerm;
};

// Parameters of ELU activation
struct NEOMATHENGINE_API CELUActivationParam final {
	static constexpr float DefaultAlpha = 0.01f;
	float Alpha = DefaultAlpha;
};

// Parameters of ReLU activation
struct NEOMATHENGINE_API CReLUActivationParam final {
	static constexpr float DefaultUpperThreshold = 0.f;
	float UpperThreshold = DefaultUpperThreshold;
};

// Parameters of leaky ReLU activation
struct NEOMATHENGINE_API CLeakyReLUActivationParam final {
	static constexpr float DefaultAlpha = 0.01f;
	float Alpha = DefaultAlpha;
};

// Parameters of HardSigmoid activation
struct NEOMATHENGINE_API CHardSigmoidActivationParam final {
	static constexpr float DefaultSlope = 0.5f;
	static constexpr float DefaultBias = 0.5f;
	float Slope = DefaultSlope;
	float Bias = DefaultBias;
};

// Parameters of power activation
struct NEOMATHENGINE_API CPowerActivationParam final {
	static constexpr float DefaultExponent = 0.f;
	float Exponent = DefaultExponent;
};

// Parameters of GELU activation
struct NEOMATHENGINE_API CGELUActivationParam final {
	// CDF can be calculated using the error function (slow) or using an approximation.
	// The approximate method is used by default.
	enum class TCalculationMode {
		// x * 0.5( 1 + erf( x / sqrt(2) ) )
		Precise,
		// x * sigmoid(1.702x)
		SigmoidApproximate,

		Count
	};

	static const TCalculationMode DefaultCalculationMode = TCalculationMode::SigmoidApproximate;
	TCalculationMode Mode = DefaultCalculationMode;
};

// Name of activation and its parameters (if any)
class NEOMATHENGINE_API CActivationDesc {
public:
	// For non-parametrized activations or default parameters
	CActivationDesc( TActivationFunction _type ) : type( _type ), isParamStored( false ) {}

	// For explicitly setting activation parameters. 'Param' must be a 'CParam' struct from the correspondent layer
	template<class Param>
	CActivationDesc( TActivationFunction _type, const Param& param ) : type( _type ) { SetParam( param ); }

	// The activation selected during the instance construction.
	TActivationFunction GetType() const { return type; }

	// Changing/setting parameters of the selected activation.
	// 'Param' must be a 'CParam' struct from the correspondent layer.
	template<class Param>
	void SetParam( const Param& param );

	// Are the parameters set
	bool HasParam() const { return isParamStored; }

	// Get parameters of the activation.
	// The parameters must be set (HasParam),
	// 'Param' must be a 'CParam' struct from the correspondent layer.
	template<class Param>
	Param GetParam() const;

private:
	std::aligned_union_t<1,
		CLinearActivationParam,
		CELUActivationParam,
		CReLUActivationParam,
		CLeakyReLUActivationParam,
		CHardSigmoidActivationParam,
		CPowerActivationParam,
		CGELUActivationParam> paramValue;
	TActivationFunction type;
	bool isParamStored;

	template<class T>
	void assertIsTypeCompatible() const;
};

template <class Param>
void CActivationDesc::SetParam( const Param& param ) {
	assertIsTypeCompatible<Param>();
	new( &paramValue ) Param( param );
	isParamStored = true;
}

template <class Param>
Param CActivationDesc::GetParam() const {
	assertIsTypeCompatible<Param>();
	if( isParamStored ) {
		return *reinterpret_cast<const Param*>( &paramValue );
	} else {
		return Param{};
	}
}

template <class T>
void CActivationDesc::assertIsTypeCompatible() const {
	static_assert( AF_Count == 15, "AF_Count != 15" );

	// compile-time check: something not even looking like CParam is given.
	static_assert( std::is_same<CLinearActivationParam, T>::value || 
		std::is_same<CELUActivationParam, T>::value || 
		std::is_same<CReLUActivationParam, T>::value ||
		std::is_same<CLeakyReLUActivationParam, T>::value ||
		std::is_same<CHardSigmoidActivationParam, T>::value || 
		std::is_same<CPowerActivationParam, T>::value || 
		std::is_same<CGELUActivationParam, T>::value, "Not CParam is given." );

	bool isSame = false;
	switch( type )
	{
		case AF_Linear:
			isSame = std::is_same<CLinearActivationParam, T>::value;
			break;
		case AF_ELU:
			isSame = std::is_same<CELUActivationParam, T>::value;
			break;
		case AF_ReLU:
			isSame = std::is_same<CReLUActivationParam, T>::value;
			break;
		case AF_LeakyReLU:
			isSame = std::is_same<CLeakyReLUActivationParam, T>::value;
			break;
		case AF_HardSigmoid:
			isSame = std::is_same<CHardSigmoidActivationParam, T>::value;
			break;
		case AF_Power:
			isSame = std::is_same<CPowerActivationParam, T>::value;
			break;
		case AF_GELU:
			isSame = std::is_same<CGELUActivationParam, T>::value;
			break;
		case AF_Abs:
		case AF_Sigmoid:
		case AF_Tanh:
		case AF_HardTanh:
		case AF_HSwish:
		case AF_Exp:
		case AF_Log:
		case AF_Erf:
		default:
			isSame = false;
	}
	ASSERT_EXPR( isSame );
}

} // namespace NeoML
