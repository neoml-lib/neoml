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

#include <LinearBinaryModel.h>

namespace NeoML {

ILinearRegressionModel::~ILinearRegressionModel() = default;

REGISTER_NEOML_MODEL( CLinearBinaryModel, LinearBinaryModelName )

CLinearBinaryModel::CLinearBinaryModel( const CFloatVector& _plane, const CSigmoid& sigmoidCoefficients ) :
	plane( _plane ),
	coefficients( sigmoidCoefficients )
{
}

bool CLinearBinaryModel::Classify( const CFloatVectorDesc& data, CClassificationResult& result ) const
{
	const double distance = LinearFunction( plane, data );
	return classify( distance, result );
}

// Calculates classification result from the distance to the separating plane
bool CLinearBinaryModel::classify( double distance, CClassificationResult& result ) const
{
	const double probability = coefficients.DistanceToProbability( distance );;

	result.ExceptionProbability = CClassificationProbability( 0.0 );
	result.Probabilities.SetSize( 2 );

	if( probability < 1 - probability ) {
		result.PreferredClass = 0;
	} else {
		result.PreferredClass = 1;
	}
	result.Probabilities[1] = CClassificationProbability( probability );
	result.Probabilities[0] = CClassificationProbability( 1 - probability );
	return true;
}

void CLinearBinaryModel::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );

	if( archive.IsStoring() ) {
		archive << plane;
		archive << coefficients;
	} else if( archive.IsLoading() ) {
		NeoAssert( plane.IsNull() );
		archive >> plane;
		archive >> coefficients;
	} else {
		NeoAssert( false );
	}
}

double CLinearBinaryModel::Predict( const CFloatVectorDesc& data ) const
{
	return LinearFunction( plane, data );
}

} // namespace NeoML
