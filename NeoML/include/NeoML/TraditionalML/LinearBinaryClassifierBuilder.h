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
#include <NeoML/TraditionalML/ClassificationResult.h>
#include <NeoML/TraditionalML/TrainingModel.h>
#include <NeoML/TraditionalML/PlattScalling.h>

namespace NeoML {

class CFunctionWithHessian;

// The error function types
enum TErrorFunction {
	EF_SquaredHinge,	// squared hinge function
	EF_LogReg,	// logistical regression function
	EF_SmoothedHinge, // one half of a hyperbolic function
	EF_L2_Regression, // the L2 regression function

	EF_Count, // this constant is equal to the number of function types
};

DECLARE_NEOML_MODEL_NAME( LinearBinaryModelName, "FmlLinearBinaryModel" )

// Trained classification model interface
class NEOML_API ILinearBinaryModel : public IModel {
public:
	virtual ~ILinearBinaryModel();

	// Gets the dividing plane
	virtual CFloatVector GetPlane() const = 0;

	// Gets the sigmoid coefficients
	virtual const CSigmoid& GetSigmoid() const = 0;
};

DECLARE_NEOML_MODEL_NAME( LinearRegressionModelName, "FmlLinearBinaryModel" )

// Trained regression model interface
class NEOML_API ILinearRegressionModel : public IRegressionModel {
public:
	virtual ~ILinearRegressionModel();

	// Gets the dividing plane
	virtual CFloatVector GetPlane() const = 0;
};

// Linear binary classifier training algorithm
class NEOML_API CLinearBinaryClassifierBuilder : public ITrainingModel, public IRegressionTrainingModel {
public:
	// Classification parameters
	struct CParams {
		TErrorFunction Function; // error function
		int MaxIterations; // maximum number of algorithm iterations
		double ErrorWeight;	// the error weight relative to the regularization coefficient
		CSigmoid SigmoidCoefficients; // the predefined sigmoid function coefficients
		double Tolerance; // the stop criterion
		bool NormalizeError; // specifies if the error should be normalized
		float L1Coeff; // the L1 regularization coefficient; set to 0 to use the L2 regularization instead
		int ThreadCount; // the number of processing threads to be used while training the model

		CParams( TErrorFunction func, double errorWeight = 1, int maxIterations = 1000,
				const CSigmoid& coefficients = CSigmoid(), double tolerance = -1, 
				bool normalizeError = false, float l1Coeff = 0.f, int threadCount = 1 ) :
			Function( func ),
			MaxIterations( maxIterations ),
			ErrorWeight( errorWeight ),
			SigmoidCoefficients( coefficients ),
			Tolerance( tolerance ),
			NormalizeError( normalizeError ),
			L1Coeff( l1Coeff ),
			ThreadCount( threadCount )
		{
			NeoPresume( errorWeight > 0 );
			NeoPresume( threadCount >= 1 );
		}
	};

	explicit CLinearBinaryClassifierBuilder( const CParams& params );
	virtual ~CLinearBinaryClassifierBuilder();

	// Sets a text stream for logging processing
	void SetLog( CTextStream* newLog ) { log = newLog; }

	// Trains a regression model
	CPtr<IRegressionModel> TrainRegression( const IRegressionProblem& problem ) override;

	// ITrainingModel interface methods:
	CPtr<IModel> Train( const IProblem& trainingClassificationData ) override;

private:
	const CParams params; // classification parameters
	CTextStream* log; // logging stream
	CFunctionWithHessian* function; // error function
};

} // namespace NeoML
