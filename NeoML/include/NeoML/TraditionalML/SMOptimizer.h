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

namespace NeoML {

class CKernelMatrix;

// The SVM kernel
class NEOML_API CSvmKernel {
public:
	enum TKernelType {
		KT_Undefined = 0,
		KT_Linear,
		KT_Poly,
		KT_RBF,
		KT_Sigmoid
	};

	CSvmKernel() : kernelType( KT_Undefined ), degree( 0 ), gamma( 0 ), coef0( 0 ) {}
	CSvmKernel(TKernelType kernelType, int degree, double gamma, double coef0);

	// The kernel type
	TKernelType KernelType() const { return kernelType; }
	// Calculates the kernel value on given vectors
	double Calculate(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const;
	double Calculate(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const;

	friend CArchive& operator << ( CArchive& archive, const CSvmKernel& center );
	friend CArchive& operator >> ( CArchive& archive, CSvmKernel& center );

private:
	TKernelType kernelType;
	int degree;
	double gamma;
	double coef0;

	double linear(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const;
	double linear(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const;
	double poly(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const;
	double poly(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const;
	double rbf(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const;
	double rbf(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const;
	double sigmoid(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const;
	double sigmoid(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const;
};

inline CArchive& operator << ( CArchive& archive, const CSvmKernel& kernel )
{
	CSvmKernel::TKernelType kernelType = kernel.kernelType;
	archive.SerializeEnum( kernelType );
	archive << kernel.degree;
	archive << kernel.gamma;
	archive << kernel.coef0;
	return archive;
}

inline CArchive& operator >> ( CArchive& archive, CSvmKernel& kernel )
{
	archive.SerializeEnum( kernel.kernelType );
	archive >> kernel.degree;
	archive >> kernel.gamma;
	archive >> kernel.coef0;
	return archive;
}

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
	CSMOptimizer(const CSvmKernel& kernel, const IProblem& data, double errorWeight, double tolerance, int cacheSize = 200000000);
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
	const double errorWeight; // the error weight relative to the regularizer (the relative weight of the data set)
	const double tolerance; // the stop criterion
	const CKernelMatrix* Q; // the kernel matrix: CQMatrix(i, j) = K(i, j)*y_i*y_j
	CTextStream* log; // the logging stream

	void findMaximalViolatingPair( const CArray<double>& alpha, const CArray<double>& gradient,
		int& i, double& Gmax, int&j, double& Gmin ) const;
	void optimizePair( int i, int j, CArray<double>& alpha, CArray<double>& gradient );
	float calculateFreeTerm( const CArray<double>& alpha, const CArray<double>& gradient ) const;

	// Retrieves the vector weight
	double getVectorWeight( int index ) const { return data->GetVectorWeight( index ) * errorWeight; }
};

} // namespace NeoML
