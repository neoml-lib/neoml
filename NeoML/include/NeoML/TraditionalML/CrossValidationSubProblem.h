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

// Data subset for cross-validation
// The input data is split into partsCount parts, with the subset equal to one of these parts (with the partIndex index) for testSet == true
// and to the rest of the parts together (except the partIndex index) for testSet == false
class NEOML_API CCrossValidationSubProblem : public ISubProblem {
public:
	CCrossValidationSubProblem( const IProblem* problem, int partsCount, int partIndex, bool testSet );

	// Gets the index of a vector in the initial data set
	virtual int GetOriginalIndex( int index ) const { return translateIndex( index ); }

	// IProblem interface methods
	virtual int GetClassCount() const { return problem->GetClassCount(); }
	virtual int GetFeatureCount() const { return problem->GetFeatureCount(); }
	virtual bool IsDiscreteFeature( int index ) const { return problem->IsDiscreteFeature( index ); }
	virtual int GetVectorCount() const { return vectorsCount; }
	virtual int GetClass( int index ) const { return problem->GetClass( translateIndex( index ) ); }
	virtual CSparseFloatMatrixDesc GetMatrix() const { return matrix; }
	virtual double GetVectorWeight( int index ) const { return problem->GetVectorWeight( translateIndex( index ) ); }
	virtual int GetDiscretizationValue( int index ) const { return problem->GetDiscretizationValue( index ); }

protected:
	virtual ~CCrossValidationSubProblem() {} // delete operator prohibited

private:
	const CPtr<const IProblem> problem; // the input data
	const int partsCount; // the number of subsets
	const int partIndex; // the index of the current subset
	bool testSet; // indicates if this is the testing or the training subset
	int vectorsCount; // the number of vectors in the subset
	CArray<int> pointerB; // vector start pointers
	CArray<int> pointerE; // vector end pointers
	CSparseFloatMatrixDesc matrix; // the matrix descriptor for the problem

	int translateIndex( int index ) const;
};

} // namespace NeoML
