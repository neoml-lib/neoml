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

/////////////////////////////////////////////////////////////////////////////////////////
// A multivariate regression problem created from a regression problem for a function with number values

class CMultivariateRegressionOverUnivariate : public IMultivariateRegressionProblem {
public:
	explicit CMultivariateRegressionOverUnivariate( const IRegressionProblem* inner );

	// Gets the number of features
	int GetFeatureCount() const override;

	// The number of vectors in the data set
	int GetVectorCount() const override;

	// Gets all vectors in the data set as a matrix
	CSparseFloatMatrixDesc GetMatrix() const override;
	// Gets the vector weight
	double GetVectorWeight( int index ) const override;

	// Gets the length of the function value vector
	int GetValueSize() const override;
	// Gets the function value for the vector
	CFloatVector GetValue( int index ) const override;

private:
	// The inner regression problem
	const CPtr<const IRegressionProblem> inner;
};

/////////////////////////////////////////////////////////////////////////////////////////
// A multivariate regression problem created from a classification problem

class CMultivariateRegressionOverClassification :
	public IMultivariateRegressionProblem {
public:
	explicit CMultivariateRegressionOverClassification( const IProblem* inner );

	// Gets the number of features
	int GetFeatureCount() const override;

	// The number of vectors in the data set
	int GetVectorCount() const override;
	// Gets all vectors in the data set as a matrix
	CSparseFloatMatrixDesc GetMatrix() const override;
	// Gets the vector weight
	double GetVectorWeight( int index ) const override;

	// Gets the length of the function value vector
	int GetValueSize() const override;
	// Gets the function value for the vector
	CFloatVector GetValue( int index ) const override;

private:
	// The inner classification problem
	const CPtr<const IProblem> inner;
	// The values for each of the classes
	// The ith vector contains 1 in position i and 0 in other positions
	CArray<CFloatVector> classValues;
};

/////////////////////////////////////////////////////////////////////////////////////////
// A multivariate regression problem created from a binary classification problem

class CMultivariateRegressionOverBinaryClassification :
	public IMultivariateRegressionProblem {
public:
	explicit CMultivariateRegressionOverBinaryClassification( const IProblem* inner );

	// Gets the number of features
	int GetFeatureCount() const override;

	// The number of vectors in the data set
	int GetVectorCount() const override;
	// Gets all vectors in the data set as a matrix
	CSparseFloatMatrixDesc GetMatrix() const override;
	// Gets the vector weight
	double GetVectorWeight( int index ) const override;

	// Gets the length of the function value vector
	int GetValueSize() const override;
	// Gets the function value for the vector
	CFloatVector GetValue( int index ) const override;

private:
	// The inner classification problem
	const CPtr<const IProblem> inner;
	// The values for each of the two classes
	CFloatVector classValues[2];
};

/////////////////////////////////////////////////////////////////////////////////////////
// A problem view without the elements with null weight
// Can be used only with an asssumption that the original matrix won't be changed during this class usage

class CProblemNotNullWeightsView : public IProblem {
public:
	explicit CProblemNotNullWeightsView( const IProblem* inner );
	~CProblemNotNullWeightsView();

	// The number of classes
	int GetClassCount() const override;

	// The number of features
	int GetFeatureCount() const override;

	// Indicates if the specified feature is discrete
	bool IsDiscreteFeature( int index ) const override;

	// The number of vectors
	int GetVectorCount() const override;

	// The correct class number for a vector with a given index in [0, GetClassCount())
	int GetClass( int index ) const override;

	// Gets all input vectors as a matrix
	CSparseFloatMatrixDesc GetMatrix() const override;

	// The vector weight
	double GetVectorWeight( int index ) const override;

	// forbid copy/move
	CProblemNotNullWeightsView( const CProblemNotNullWeightsView& ) = delete;
	CProblemNotNullWeightsView( CProblemNotNullWeightsView&& ) = delete;
	CProblemNotNullWeightsView& operator=( CProblemNotNullWeightsView ) = delete;

private:
	// The inner problem
	const CPtr<const IProblem> inner;
	// The original matrix desc view over the elements with not null weight only
	CSparseFloatMatrixDesc viewMatrixDesc;
	// The array containing pairs of viewed and original indices
	CArray<int> notNullWeightElementsIndices;

	int calculateOriginalIndex( int viewedIndex ) const;
};

/////////////////////////////////////////////////////////////////////////////////////////

} // namespace NeoML

