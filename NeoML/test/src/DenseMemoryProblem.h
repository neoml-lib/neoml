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

class CDenseMemoryProblem : public IProblem {
public:
	CDenseMemoryProblem( int height, int width, float* values, const int* _classes, const float* _weights );

	CSparseFloatVectorDesc GetVector( int index ) const { return desc.GetRow( index ); }

	// IProblem interface methods:
	int GetClassCount() const override { return classCount; }
	int GetFeatureCount() const override { return desc.Width; }
	bool IsDiscreteFeature( int ) const override { return false; }
	int GetVectorCount() const override { return desc.Height; }
	int GetClass( int index ) const override { return classes[index]; }
	CSparseFloatMatrixDesc GetMatrix() const override { return desc; }
	double GetVectorWeight( int index ) const override { return weights[index]; };

	static CPtr<CDenseMemoryProblem> Random( int samples, int features, int classes );
	CPtr<CMemoryProblem> CreateSparse() const;

protected:
	~CDenseMemoryProblem() override = default;

private:
	CDenseMemoryProblem() = default;

	CSparseFloatMatrixDesc desc;
	int classCount;
	const int* classes;
	const float* weights;

	// memory holders when applicable
	CArray<float> valuesArr;
	CArray<int> classesArr;
	CArray<float> weightsArr;
};

} // namespace NeoMLTest
