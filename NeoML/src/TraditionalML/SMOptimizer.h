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
class CSMOptimizer {
public:
	// kernel is the SVM kernel function
	// data contains the training set
	// tolerance is the required precision
	// cacheSize is the cache size in MB
	CSMOptimizer(const CSvmKernel& kernel, const IProblem& data, int maxIter, double errorWeight, double tolerance,
		bool doShrinking, int cacheSize = 200);
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
	bool doShrinking; // do shrinking or not
	CKernelMatrix* kernelMatrix; // the kernel matrix
	CTextStream* log; // the logging stream

	// optimizer variables, we use raw pointers to speed up optimization
	CArray<double> gradient; // gradient
	double* g; // gradient raw pointer
	CArray<double> gradient0; // gradient, if we treat free variables as 0
	double* g0; // gradient0 raw pointer array
	CArray<double> alphaArray; // gradient
	double* alpha; // alpha
	CArray<double> weightsMultErrorWeightArray; // vector of weigths * errorWeight
	double* weightsMultErrorWeight; // a raw pointer to weightsMultErrorWeightArray data
	int vectorCount; // a problem length
	const float* y; // vector of [-1,1] class labels
	const double* matrixDiagonal; // matrix diagonal

	CArray<TAlphaStatus> alphaStatusArray; // alpha statuses
	TAlphaStatus* alphaStatus; // alpha status raw pointer array
	CArray<int> activeSetArray; // active set array
	int* activeSet; // active set raw pointer array
	int activeSize; // number of vector that are being optimized in this moment
	bool isShrunk; // whether the problem was shrunk or not

	bool findMaxViolatingIndices( int& outI, int& outJ ) const;
	void optimizeIndices( int i, int j );
	void updateAlphaStatusAndGradient0( int i );
	void reconstructGradient();
	void swapIndices( int i, int j );
	void shrink();
	bool canBeShrunk( int i, double gMax1, double gMax2 );
	float calculateFreeTerm() const;
};

inline bool CSMOptimizer::canBeShrunk( int i, double gMax1, double gMax2 )
{
	if( alphaStatus[i] == AS_UpperBound ) {
		if( y[i] == 1 ) {
			return -g[i] > gMax1;
		} else {
			return -g[i] > gMax2;
		}
	} else if( alphaStatus[i] == AS_LowerBound ) {
		if( y[i] == 1 ) {
			return g[i] > gMax2;
		} else {
			return g[i] > gMax1;
		}
	} else {
		return false;
	}
}

} // namespace NeoML
