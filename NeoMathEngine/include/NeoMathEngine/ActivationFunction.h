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

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <NeoMathEngine/NeoMathEngineException.h>

namespace NeoML {

// Supported activation functions
enum TActivationFunction {
	AF_Linear = 0,
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

	AF_Count,

	AF_None = -1
};

// Returns default param first param value for given activation function
// Returns 0.f if activation doesn't use first param
inline float ActivationDefaultParam1( TActivationFunction activation )
{
	static_assert( AF_Count == 12, "AF_Count != 12" );
	switch( activation ) {
		case AF_Linear:
			return 1.f; // Multiplier
		case AF_ELU:
			return 0.01f; // Alpha
		case AF_ReLU:
			return 0.f; // Upper threshold
		case AF_LeakyReLU:
			return 0.01f; // Alpha (negative slope)
		case AF_HardSigmoid:
			return 0.5f; // Slope
		case AF_Power:
			return 0.f; // Exponent
		case AF_GELU:
			return 1.702f; // Multiplier
		// Other activations don't have params
		case AF_Abs:
		case AF_Sigmoid:
		case AF_Tanh:
		case AF_HardTanh:
		case AF_HSwish:
		case AF_None:
			return 0.f;
		case AF_Count:
		default:
			ASSERT_EXPR( false );
			return 0.f;
	}
	return 0.f;
}

inline float ActivationDefaultParam2( TActivationFunction activation )
{
	static_assert( AF_Count == 12, "AF_Count != 12" );
	switch( activation ) {
		case AF_Linear:
			return 0.f; // Free term
		case AF_HardSigmoid:
			return 0.5f; // Bias
		// Other activations don't have second param
		case AF_ELU:
		case AF_ReLU:
		case AF_LeakyReLU:
		case AF_Abs:
		case AF_Sigmoid:
		case AF_Tanh:
		case AF_HardTanh:
		case AF_Power:
		case AF_HSwish:
		case AF_GELU:
		case AF_None:
			return 0.f;
		case AF_Count:
		default:
			ASSERT_EXPR( false );
			return 0.f;
	}
	return 0.f;
}

// Structure used for creating activation descriptors in MathEngine
struct CActivationInfo {
	// Constructors are not explicit
	// It allows to use implicit conversion from AF_* constants
	// or from initializer lists like { AF_Linear, 0.5f }
	CActivationInfo( TActivationFunction type ) :
		Type( type ), Param1( ActivationDefaultParam1( type ) ), Param2( ActivationDefaultParam2( type ) ) {}
	CActivationInfo( TActivationFunction type, float param1 ) :
		Type( type ), Param1( param1 ), Param2( ActivationDefaultParam2( type ) ) {}
	CActivationInfo( TActivationFunction type, float param1, float param2 ) :
		Type( type ), Param1( param1 ), Param2( param2 ) {}

	TActivationFunction Type; // Activation function
	float Param1; // First parameter
	float Param2; // Second parameter
};

} // namespace NeoML
