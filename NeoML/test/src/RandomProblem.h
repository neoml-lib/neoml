/* Copyright © 2021 ABBYY Production LLC

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

namespace NeoMLTest {

// implementation of creation random dense dataset and convertion it into sparse
template<typename TLabel>
class CRandomProblemImpl : public virtual IObject {
public:
	CRandomProblemImpl( int height, int width, float* values, const TLabel* _labels, const float* _weights );

	static CPtr<CRandomProblemImpl> Random( CRandom& rand, int samples, int features, int labels );
	CPtr<CRandomProblemImpl> CreateSparse() const;

	CSparseFloatMatrix Matrix;
	int LabelsCount;
	const TLabel* Labels;
	const float* Weights;

	// memory holders when applicable
	CArray<float> Values;
	CArray<int> PointerB;
	CArray<int> PointerE;
	CArray<TLabel> LabelsArr;
	CArray<float> WeightsArr;

private:
	CRandomProblemImpl() = default;
};

// classification random problem impl
class CClassificationRandomProblem : public IProblem {
public:
	CClassificationRandomProblem( int height, int width, float* values, const int* _labels, const float* _weights );

	CSparseFloatVectorDesc GetVector( int index ) const { return GetMatrix().GetRow( index ); }

	static CPtr<CClassificationRandomProblem> Random( CRandom& rand, int samples, int features, int labels );
	CPtr<CClassificationRandomProblem> CreateSparse() const;

	// IProblem interface methods:
	int GetClassCount() const override { return impl->LabelsCount; }
	int GetFeatureCount() const override { return GetMatrix().Width; }
	bool IsDiscreteFeature( int ) const override { return false; }
	int GetVectorCount() const override { return GetMatrix().Height; }
	int GetClass( int index ) const override { return impl->Labels[index]; }
	CSparseFloatMatrixDesc GetMatrix() const override { return impl->Matrix.GetDesc(); }
	double GetVectorWeight( int index ) const override { return impl->Weights[index]; };

protected:
	~CClassificationRandomProblem() override = default;

private:
	CClassificationRandomProblem() = default;
	CPtr< CRandomProblemImpl<int> > impl;
};

// regression random problem impl
class CRegressionRandomProblem : public IRegressionProblem {
public:
	CRegressionRandomProblem( int height, int width, float* values, const double* _labels, const float* _weights );

	CSparseFloatVectorDesc GetVector( int index ) const { return GetMatrix().GetRow( index ); }

	static CPtr<CRegressionRandomProblem> Random( CRandom& rand, int samples, int features, int labels );
	CPtr<CRegressionRandomProblem> CreateSparse() const;

	// IProblem interface methods:
	int GetFeatureCount() const override { return GetMatrix().Width; }
	int GetVectorCount() const override { return GetMatrix().Height; }
	double GetValue( int index ) const override { return impl->Labels[index]; }
	CSparseFloatMatrixDesc GetMatrix() const override { return impl->Matrix.GetDesc(); }
	double GetVectorWeight( int index ) const override { return impl->Weights[index]; };

protected:
	~CRegressionRandomProblem() override = default;

private:
	CRegressionRandomProblem() = default;
	CPtr< CRandomProblemImpl<double> > impl;
};

} // namespace NeoMLTest
