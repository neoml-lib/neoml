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

#include <NeoML/TraditionalML/Problem.h>

namespace NeoML {

// The data subset for cross-validation
// The input data set is split into partsCount parts; the subset corresponds to one of these parts (partIndex)
// if testSet is true, and all other parts together (except the partIndex) if testSet is false
// It is guaranteed that the ratio of classes in each part is (almost) the same as in the input data
class NEOML_API CStratifiedCrossValidationSubProblem : public ISubProblem {
public:
	CStratifiedCrossValidationSubProblem( const IProblem* problem, int partsCount, int partIndex, bool testSet );

	// The index of the element in the original data set
	int GetOriginalIndex( int index ) const { return translateIndex( index ); }

	// IProblem interface methods
	int GetClassCount() const override { return problem->GetClassCount(); }
	int GetFeatureCount() const override { return problem->GetFeatureCount(); }
	bool IsDiscreteFeature( int index ) const override { return problem->IsDiscreteFeature( index ); }
	int GetVectorCount() const override { return vectorsCount; }
	int GetClass( int index ) const override { return problem->GetClass( translateIndex( index ) ); }
	CFloatMatrixDesc GetMatrix() const override { return matrix; }
	double GetVectorWeight( int index ) const override { return problem->GetVectorWeight( translateIndex( index ) ); }
	int GetDiscretizationValue( int index ) const override { return problem->GetDiscretizationValue( index ); }

protected:
	// delete prohibited
	virtual ~CStratifiedCrossValidationSubProblem() {}

private:
	const CPtr<const IProblem> problem; // the original data
	const int partsCount; // the number of parts
	const int partIndex; // the index of the current part
	bool testSet; // indicates if this is the test subset
	int vectorsCount; // the number of vectors in the subset
	CArray< CArray<int> > objectsPerPart; // the list of object indices for each of the parts
	int minPartSize; // the minimum number of objects in each part (the total number of elements / the number of parts)
	int objectsBeforeTestPart; // the number of objects before the test part
	CArray<int> pointerB; // the pointers to the vector beginnings
	CArray<int> pointerE; // the pointers to the vector ends
	CFloatMatrixDesc matrix; // the problem matrix descriptor

	int translateIndex( int index ) const;
	void buildObjectsLists();
};

} // namespace NeoML
