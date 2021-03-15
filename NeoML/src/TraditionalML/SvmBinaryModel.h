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

#include <NeoML/TraditionalML/Svm.h>

namespace NeoML {

// The binary SVM classifier
class CSvmBinaryModel : public ISvmBinaryModel {
public:
	CSvmBinaryModel() : freeTerm( 0 ) {}
	CSvmBinaryModel( const CSvmKernel& kernel, const IProblem& problem, const CArray<double>& alpha, double freeTerm );

	// For serialization
	static CPtr<IModel> Create() { return FINE_DEBUG_NEW CSvmBinaryModel(); }

	// IModel interface methods
	virtual int GetClassCount() const { return 2; }
	virtual bool Classify( const CSparseFloatVectorDesc& data, CClassificationResult& result ) const;
	virtual bool Classify( const CFloatVector& data, CClassificationResult& result ) const;
	virtual void Serialize( CArchive& archive );

	// ISvmBinaryModel interface methods
	virtual CSvmKernel::TKernelType GetKernelType() const { return kernel.KernelType(); }
	virtual CSparseFloatMatrix GetVectors() const { return matrix; }
	virtual const CArray<double>& GetAlphas() const { return alpha; }
	virtual double GetFreeTerm() const { return freeTerm; }

protected:
	virtual ~CSvmBinaryModel() {} // delete prohibited

private:
	CSvmKernel kernel; // the kernel
	double freeTerm; // the free term
	CSparseFloatMatrix matrix; // the support vectors
	CArray<double> alpha; // the coefficients
};

} // namespace NeoML
