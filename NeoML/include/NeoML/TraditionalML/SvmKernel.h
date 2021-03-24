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

namespace NeoML {

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
	double Calculate(const CFloatVectorDesc& x1, const CFloatVectorDesc& x2) const;
	double Calculate(const CFloatVector& x1, const CFloatVectorDesc& x2) const { return Calculate( x1.GetDesc(), x2 ); }

	friend CArchive& operator << ( CArchive& archive, const CSvmKernel& center );
	friend CArchive& operator >> ( CArchive& archive, CSvmKernel& center );

private:
	TKernelType kernelType;
	int degree;
	double gamma;
	double coef0;

	double linear(const CFloatVectorDesc& x1, const CFloatVectorDesc& x2) const;
	double poly(const CFloatVectorDesc& x1, const CFloatVectorDesc& x2) const;
	double rbf(const CFloatVectorDesc& x1, const CFloatVectorDesc& x2) const;
	double sigmoid(const CFloatVectorDesc& x1, const CFloatVectorDesc& x2) const;

	double rbfDenseBySparse( const CFloatVectorDesc& x1, const CFloatVectorDesc& x2 ) const;
	double rbfDenseByDense( const CFloatVectorDesc& x1, const CFloatVectorDesc& x2 ) const;
	double rbfSparseBySparse( const CFloatVectorDesc& x1, const CFloatVectorDesc& x2 ) const;
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

} // namespace NeoML
