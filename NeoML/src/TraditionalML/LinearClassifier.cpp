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

#include <NeoML/TraditionalML/LinearClassifier.h>
#include <NeoML/TraditionalML/OneVersusAll.h>
#include <NeoML/TraditionalML/TrustRegionNewtonOptimizer.h>
#include <LinearBinaryModel.h>
#include <NeoML/TraditionalML/PlattScalling.h>

namespace NeoML {

ILinearBinaryModel::~ILinearBinaryModel()
{}

// Normalizes the error weight
static double normalizeErrorWeight( const CLinearClassifier::CParams& param, const IProblem& trainingClassificationData )
{
	double totalWeight = 0;
	const int vectorCount = trainingClassificationData.GetVectorCount();
	for( int i = 0; i < vectorCount; ++i ) {
		totalWeight += trainingClassificationData.GetVectorWeight( i );
	}
	return param.ErrorWeight / totalWeight;
}

// Normalizes the error weight
static double normalizeErrorWeight( const CLinearClassifier::CParams& param, const IRegressionProblem& problem )
{
	double totalWeight = 0;
	const int vectorCount = problem.GetVectorCount();
	for( int i = 0; i < vectorCount; ++i ) {
		totalWeight += problem.GetVectorWeight( i );
	}
	return param.ErrorWeight / totalWeight;
}

// Creates a class for the loss function
static CFunctionWithHessian* createOptimizedFunction( const CLinearClassifier::CParams& param, const IProblem& trainingClassificationData )
{
	static_assert( EF_Count == 4, "EF_Count != 4" );

	const double errorWeight = param.NormalizeError ? normalizeErrorWeight( param, trainingClassificationData )
		: param.ErrorWeight;

	switch( param.Function ) {
		case EF_SquaredHinge:
			return FINE_DEBUG_NEW CSquaredHinge( trainingClassificationData, errorWeight, param.L1Coeff, param.ThreadCount );
		case EF_LogReg:
			return FINE_DEBUG_NEW CLogRegression( trainingClassificationData, errorWeight, param.L1Coeff, param.ThreadCount );
		case EF_SmoothedHinge:
			return FINE_DEBUG_NEW CSmoothedHinge( trainingClassificationData, errorWeight, param.L1Coeff, param.ThreadCount );
		case EF_L2_Regression:
			// not for classification
		default:
			NeoAssert(false);
			return 0;
	};
}

//---------------------------------------------------------------------------------------------------------

CLinearClassifier::CLinearClassifier( const CParams& _params ) :
	params( _params ),
	log( 0 ),
	function( 0 )
{
}

CLinearClassifier::~CLinearClassifier()
{
	if( function != 0 ) {
		delete function;
	}
}

CPtr<IRegressionModel> CLinearClassifier::TrainRegression( const IRegressionProblem& problem )
{
	if( function != 0 ) {
		delete function; // delete the old loss function
	}
	const double errorWeight = params.NormalizeError ? normalizeErrorWeight( params, problem ) : params.ErrorWeight;
	NeoAssert( params.Function == EF_L2_Regression );
	function = FINE_DEBUG_NEW CL2Regression( problem, errorWeight, 1e-6, params.L1Coeff, params.ThreadCount );
	const double tolerance = max( 1e-6, params.Tolerance );

	CTrustRegionNewtonOptimizer optimizer( function, tolerance, params.MaxIterations );
	CFloatVector initialPlane( problem.GetFeatureCount() + 1 );
	initialPlane.Nullify();
	optimizer.SetInitialArgument( initialPlane );
	optimizer.Optimize();

	CFloatVector plane = optimizer.GetOptimalArgument();
	CSigmoid sigmoidCoefficients;
	return FINE_DEBUG_NEW CLinearBinaryModel( plane, sigmoidCoefficients );
}

CPtr<IModel> CLinearClassifier::Train( const IProblem& trainingClassificationData )
{
	if( trainingClassificationData.GetClassCount() > 2 ) {
		return COneVersusAll( *this ).Train( trainingClassificationData );
	}

	if( function != 0 ) {
		delete function; // delete the old loss function
	}
	function = createOptimizedFunction( params, trainingClassificationData );
	const int vectorsCount = trainingClassificationData.GetVectorCount();

	double tolerance = 0;
	if( params.Tolerance >= 0 ) {
		tolerance = params.Tolerance;
	} else {
		// Calculating tolerance
		int positiveCount = 0;
	
		for(int i = 0; i < vectorsCount; i++ ) {
			if( trainingClassificationData.GetBinaryClass( i ) > 0 ) {
				positiveCount++;
			}
		}
		tolerance = 0.01 * max( min(positiveCount, vectorsCount  - positiveCount), 1 ) / vectorsCount;
	}

	CTrustRegionNewtonOptimizer optimizer( function, tolerance, params.MaxIterations );
	CFloatVector initialPlane( trainingClassificationData.GetFeatureCount() + 1 );
	initialPlane.Nullify();
	optimizer.SetInitialArgument( initialPlane );
	optimizer.Optimize();

	CFloatVector plane = optimizer.GetOptimalArgument();
	CSigmoid sigmoidCoefficients;
	if( params.SigmoidCoefficients.IsValid() ) {
		sigmoidCoefficients = params.SigmoidCoefficients;
	} else {
		CSparseFloatMatrixDesc matrix = trainingClassificationData.GetMatrix();
		CSparseFloatVectorDesc vector;
		CArray<double> distances;
		for( int i = 0; i < vectorsCount; i++ ) {
			matrix.GetRow( i, vector );
			distances.Add( LinearFunction( plane, vector ) );
		}
		CalcSigmoidCoefficients( trainingClassificationData, distances, sigmoidCoefficients );
	}

	return FINE_DEBUG_NEW CLinearBinaryModel( plane, sigmoidCoefficients );
}

} // namespace NeoML
