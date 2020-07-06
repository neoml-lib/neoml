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
#include <NeoML/TraditionalML/SMOptimizer.h>

namespace NeoML {

DECLARE_NEOML_MODEL_NAME( SvmBinaryModelName, "FmlSvmBinaryModel" )

// Support-vector machine binary classifier
class NEOML_API ISvmBinaryModel : public IModel {
public:
	virtual ~ISvmBinaryModel();

	// Gets the kernel type
	virtual CSvmKernel::TKernelType GetKernelType() const = 0;

	// Gets the support vectors
	virtual CSparseFloatMatrix GetVectors() const = 0;

	// Gets the support vector coefficients
	virtual const CArray<double>& GetAlphas() const = 0;

	// Gets the free term
	virtual double GetFreeTerm() const = 0;
};

// Binary SVM training algorithm
class NEOML_API CSvmBinaryClassifierBuilder : public ITrainingModel {
public:
	// Classification parameters
	struct CParams {
		CSvmKernel::TKernelType KernelType; // the type of error function used
		double ErrorWeight; // the weight of the error relative to the regularization function
		int MaxIterations; // the maximum number of algorithm iterations
		int Degree; // Gaussian kernel degree
		double Gamma; // the coefficient before the kernel (used for KT_Poly, KT_RBF, KT_Sigmoid).
		double Coeff0; // the free term in the kernel (used for KT_Poly, KT_Sigmoid).
		double Tolerance; // the solution precision and the stop criterion
		int ThreadCount; // the number of processing threads used

		CParams( CSvmKernel::TKernelType kerneltype, double errorWeight = 1., int maxIterations = 10000,
				int degree = 1, double gamma = 1., double coeff0 = 1., double tolerance = 0.1, int threadCount = 1 ) :
			KernelType( kerneltype ),
			ErrorWeight( errorWeight ),
			MaxIterations( maxIterations ),
			Degree( degree ),
			Gamma( gamma ),
			Coeff0( coeff0 ),
			Tolerance( tolerance ),
			ThreadCount( threadCount )
		{
		}
	};

	explicit CSvmBinaryClassifierBuilder( const CParams& params );

	// Sets the text stream for logging processing
	void SetLog( CTextStream* newLog ) { log = newLog; }

	// ITrainingModel interface methods:
	// The resulting IModel is either a ILinearBinaryModel (if the KT_Linear kernel was used)
	// or a ISvmBinaryModel (if some other kernel was used)
	CPtr<IModel> Train( const IProblem& trainingClassificationData ) override;

private:
	const CParams params; // classification parameters
	CTextStream* log; // Logging stream
};

} // namespace NeoML
