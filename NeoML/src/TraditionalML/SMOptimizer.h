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

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//  min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//      y^T \alpha = \delta
//      y_i = +1 or -1
//      0 <= alpha_i <= Cp for y_i = 1
//      0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//  Q, p, y, Cp, Cn, and an initial feasible point \alpha
//  l is the size of vectors and matrices
//  eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
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
	void SetLog( CTextStream* newLog ) { log = newLog; }

private:
	enum TAlphaStatus {
		AS_LowerBound,
		AS_UpperBound,
		AS_Free
	}; 

	static const double Inf; // +infinity
	static const double Tau; // infinitesimal number

	const CPtr<const IProblem> data; // the training set
	int maxIter; // maximal iteration
	const double errorWeight; // the error weight relative to the regularizer (the relative weight of the data set)
	const double tolerance; // the stop criterion
	bool shrinking; // do shrinking or not
	const CKernelMatrix* Q; // the kernel matrix: CQMatrix(i, j) = K(i, j)*y_i*y_j
	CTextStream* log; // the logging stream

	// optimizer variables, we use raw pointers to speed up optimization
	CArray<double> gradient; // gradient
	double* g; // gradient raw pointer
	CArray<double> gradient0; // gradient, if we treat free variables as 0
	double* g0; // gradient0 raw pointer array
	CArray<double> alphaArray; // gradient
	double* alpha; // alpha
	CArray<double> weightsMultErrorWeight; // vector of weigths * errorWeight
	const double* C; // C raw pointer array
	int l; // problem length
	const float* y; // vector of [-1,1] class labels
	const double* QD; // matrix diagonal

	CArray<TAlphaStatus> alphaStatusArray; // alpha statuses
	TAlphaStatus* alphaStatus; // alpha status raw pointer array
	CArray<int> activeSetArray; // active set array
	int* activeSet; // active set raw pointer array
	int activeSize; // active set size

	void updateAlphaStatus( int i );
	bool isShrunk();
	void reconstructGradient();
	bool selectWorkingSet( int& outI, int& outJ ) const;
	void optimizePair( int i, int j );
	void shrink();
	float calculateFreeTerm() const;
};

inline void CSMOptimizer::updateAlphaStatus( int i )
{
	if( alpha[i] >= C[i] ) {
		alphaStatus[i] = AS_UpperBound;
	} else if( alpha[i] <= 0 ) {
		alphaStatus[i] = AS_LowerBound;
	} else {
		alphaStatus[i] = AS_Free;
	}
}

} // namespace NeoML
