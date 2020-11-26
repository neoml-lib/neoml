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
#include <math.h>
#include <NeoML/TraditionalML/FloatVector.h>
#include <NeoML/TraditionalML/Problem.h>
#include <NeoML/TraditionalML/SvmKernel.h>

namespace NeoML {

class CKernelMatrix;

// The classification rule:
//
// Sum(alpha_i*y_i*K(x_i, x)) + freeTerm <> 0
// 
// The function to optimize:
//
//	min 0.5(\alpha^T Q \alpha) - e^T \alpha
//
//		y_i = +1 or -1
//		0 <= alpha_i <= C_i
//		y^T \alpha = 0
//

// The optimizer for a support-vector machine that uses SMO
class NEOML_API CSMOptimizer {
public:
	// kernel is the SVM kernel function
	// data contains the training set
	// tolerance is the required precision
	// cacheSize is the cache size in MB
	CSMOptimizer(const CSvmKernel& kernel, const IProblem& data, int maxIter, double errorWeight, double tolerance,
		bool shrinking = false, int cacheSize = 100);
	~CSMOptimizer();

	// Calculates the optimal multipliers for the support vectors
	void Optimize( CArray<double>& alpha, float& freeTerm );

	// Sets a text stream for logging processing
	// By default logging is off (set to null to turn off)
	void SetLog(CTextStream* newLog) { log = newLog; }

private:
	static const double Inf; // +infinity
	static const double Tau; // infinitesimal number

	const CPtr<const IProblem> data; // the training set
	int maxIter; // maximal iteration
	const double errorWeight; // the error weight relative to the regularizer (the relative weight of the data set)
	const double tolerance; // the stop criterion
	bool shrinking; // do shrinking or not
	const CKernelMatrix* Q; // the kernel matrix: CQMatrix(i, j) = K(i, j)*y_i*y_j
	CTextStream* log; // the logging stream

	// optimize variables
	CArray<double> gradient; // gradient
	double* g; // gradient raw pointer
	CArray<double> gradient0; // gradient, if we treat free variables as 0
	double* g0; // gradient0 raw pointer array
	double* alpha; // alpha
	CArray<double> weightsMultErrorWeight; // vector of weigths * errorWeight
	const double* C; // C raw pointer array
	int l; // problem length
	const float* y; // vector of [-1,1] class labels
	const double* QD; // matrix diagonal

	enum { LOWER_BOUND, UPPER_BOUND, FREE }; 
	CArray<char> alphaStatusArray; // alpha statuses
	char* alphaStatus; // alpha status raw pointer array
	CArray<int> activeSetArray; // active set array
	int* activeSet; // active set raw pointer array
	int activeSize; // active set size

	void reconstructGradient();
	void updateAlphaStatus( int i );
	bool selectWorkingSet( int& outI, int& outJ ) const;
	void optimizePair( int i, int j ) const;
	float calculateFreeTerm() const;
};

inline void CSMOptimizer::updateAlphaStatus( int i )
{
	if( alpha[i] >= C[i] ) {
		alphaStatus[i] = UPPER_BOUND;
	} else if( alpha[i] <= 0 ) {
		alphaStatus[i] = LOWER_BOUND;
	} else {
		alphaStatus[i] = FREE;
	}
}

} // namespace NeoML
