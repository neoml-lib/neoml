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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/FloatVector.h>
#include <NeoML/TraditionalML/Problem.h>
#include <NeoML/TraditionalML/CrossValidation.h>

namespace NeoML {

const double MaxExpArgument = 30; // the maximum argument of an exponential function

// The sigmoid coefficients
// S(x) = 1 / 1 + exp( A * x + B );
struct CSigmoid {
	double A;
	double B;

	CSigmoid() : A( 0 ), B( 0 ) {}

	bool IsValid() const { return ( A < 0 ); }
	double DistanceToProbability( double distance ) const;
};

inline double CSigmoid::DistanceToProbability( double distance ) const
{
	NeoAssert( IsValid() );
	const double value = A * distance + B;
	if( MaxExpArgument < value ) {
		return 0;
	}
	if( value < -MaxExpArgument ) {
		return 1;
	}
	return 1 / ( 1 + exp( value ) );
}

inline CArchive& operator << ( CArchive& archive, const CSigmoid& sigmoid )
{
	archive << sigmoid.A;
	archive << sigmoid.B;
	return archive;
}

inline CArchive& operator >> ( CArchive& archive, CSigmoid& sigmoid )
{
	archive >> sigmoid.A;
	archive >> sigmoid.B;
	return archive;
}

// Calculates coefficients of the sigmoid that is used for probability assessment of the classification result.
// The method is known as Platt scaling.
void NEOML_API CalcSigmoidCoefficients( const IProblem& trainingClassificationData,
	const CArray<double>& classificatorOutput, CSigmoid& coefficients );

// Calculates the sigmoid coefficients using the CLinearBinaryClassifierBuilder cross-validation result.
void NEOML_API CalcSigmoidCoefficients( const CCrossValidationResult& crossValidationResult,
	CSigmoid& coefficients );

} // namespace NeoML
