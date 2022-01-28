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
	CSvmBinaryModel() = default;
	CSvmBinaryModel( const CSvmKernel& kernel, const IProblem& problem, const CArray<double>& alpha, double freeTerm );

	// For serialization
	static CPtr<IModel> Create() { return FINE_DEBUG_NEW CSvmBinaryModel(); }

	// IModel interface methods
	int GetClassCount() const override { return 2; }
	bool Classify( const CFloatVectorDesc& data, CClassificationResult& result ) const override;
	void Serialize( CArchive& archive ) override;

	// ISvmBinaryModel interface methods
	CSvmKernel::TKernelType GetKernelType() const override { return kernel.KernelType(); }
	CSparseFloatMatrix GetVectors() const override { return matrix; }
	const CArray<double>& GetAlphas() const override { return alpha; }
	double GetFreeTerm() const override { return freeTerm; }

protected:
	~CSvmBinaryModel() override = default; // delete prohibited

private:
	CSvmKernel kernel; // the kernel
	double freeTerm{}; // the free term
	CSparseFloatMatrix matrix; // the support vectors
	CArray<double> alpha; // the coefficients
};

} // namespace NeoML
