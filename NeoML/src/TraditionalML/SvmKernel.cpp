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

// This method is described in
// "Working Set Selection Using Second Order Information for Training Support Vector Machines"
// Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin, 
// "Journal of Machine Learning Research" 6 (2005) 1889–1918
// http://www.csie.ntu.edu.tw/~cjlin/papers/quadworkset.pdf

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/SvmKernel.h>

namespace NeoML {

// Raise a number to a power: base**times
inline double power(double base, int times)
{
	double tmp = base, ret = 1.0;
	for(int t = times; t > 0; t /= 2)
	{
		if(t % 2 == 1) {
			ret *= tmp;
		}
		tmp = tmp * tmp;
	}
	return ret;
}

CSvmKernel::CSvmKernel(TKernelType kernelType, int degree, double gamma, double coef0) :
	kernelType(kernelType), degree(degree), gamma(gamma), coef0(coef0)
{
}

// The linear kernel
double CSvmKernel::linear(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const
{
	return DotProduct(x1, x2);
}

double CSvmKernel::linear(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const
{
	return DotProduct(x1, x2);
}

// The polynomial kernel
double CSvmKernel::poly(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const
{
	return power(gamma * DotProduct(x1, x2) + coef0, degree);
}

double CSvmKernel::poly(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const
{
	return power(gamma * DotProduct(x1, x2) + coef0, degree);
}

// The Gaussian kernel
double CSvmKernel::rbf(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const
{
	double square = 0;
	int i, j;
	double diff;
	for( i = 0, j = 0; i < x1.Size && j < x2.Size; ) {
		if( x1.Indexes[i] == x2.Indexes[j] ) {
			diff = x1.Values[i] - x2.Values[j];
			i++;
			j++;
		} else if( x1.Indexes[i] < x2.Indexes[j] ) {
			diff = x1.Values[i];
			i++;
		} else {
			diff = x2.Values[j];
			j++;
		}
		square += diff * diff;
	}
	for(; i < x1.Size; i++) {
		diff = x1.Values[i];
		square += diff * diff;
	}
	for(; j < x2.Size; j++) {
		diff = x2.Values[j];
		square += diff * diff;
	}
	return exp(-gamma * square);
}

double CSvmKernel::rbf(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const
{
	double square = 0;
	int i, j;
	double diff;
	for( i = 0, j = 0; i < x1.Size() && j < x2.Size; ) {
		if( i == x2.Indexes[j] ) {
			diff = x1[i] - x2.Values[j];
			i++;
			j++;
		} else if( i < x2.Indexes[j] ) {
			diff = x1[i];
			i++;
		} else {
			diff = x2.Values[j];
			j++;
		}
		square += diff * diff;
	}
	for(; i < x1.Size(); i++) {
		diff = x1[i];
		square += diff * diff;
	}
	for(; j < x2.Size; j++) {
		diff = x2.Values[j];
		square += diff * diff;
	}
	return exp(-gamma * square);
}

// The sigmoid kernel
double CSvmKernel::sigmoid(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const
{
	return tanh(gamma * DotProduct(x1, x2) + coef0);
}

double CSvmKernel::sigmoid(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const
{
	return tanh(gamma * DotProduct(x1, x2) + coef0);
}

double CSvmKernel::Calculate(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const
{
	switch( kernelType ) {
		case KT_Linear:
			return linear(x1, x2);
		case KT_Poly:
			return poly(x1, x2);
		case KT_RBF:
			return rbf(x1, x2);
		case KT_Sigmoid:
			return sigmoid(x1, x2);
		default:
			NeoAssert(false);
			return 0;
	}
}

double CSvmKernel::Calculate(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const
{
	switch( kernelType ) {
		case KT_Linear:
			return linear(x1, x2);
		case KT_Poly:
			return poly(x1, x2);
		case KT_RBF:
			return rbf(x1, x2);
		case KT_Sigmoid:
			return sigmoid(x1, x2);
		default:
			NeoAssert(false);
			return 0;
	}
}

} // namespace NeoML
