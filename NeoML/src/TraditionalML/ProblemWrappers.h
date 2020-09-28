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
// A not null elements view mechanism
// On initialize calculates not null indices and remaps pointers to sparse matrix vectors

template<class TProblem>
class CNotNullWeightsView {
public:
	CNotNullWeightsView( const TProblem* problem );
	virtual ~CNotNullWeightsView();

	// Calculates the index as if we had the matrix without null weighted elements
	int CalculateOriginalIndex( int viewedIndex ) const;

	// forbid copy/move
	CNotNullWeightsView( const CNotNullWeightsView& ) = delete;
	CNotNullWeightsView( CNotNullWeightsView&& ) = delete;
	CNotNullWeightsView& operator=( CNotNullWeightsView ) = delete;

protected:
	// The original matrix desc view over the elements with not null weight only
	CSparseFloatMatrixDesc ViewMatrixDesc;

private:
	// The array containing pairs of viewed and original indices
	CArray<int> notNullWeightElementsIndices;
	// Number of null weighted elements
	int nullWeightElementsCount;
};

template<class TProblem>
inline int CNotNullWeightsView<TProblem>::CalculateOriginalIndex( int viewedIndex ) const
{
	return nullWeightElementsCount == 0 ? viewedIndex : notNullWeightElementsIndices[viewedIndex];
}

/////////////////////////////////////////////////////////////////////////////////////////
// An IMultivatiateRegressionProblem view without the elements with null weight
// Can be used only with an asssumption that the original matrix won't be changed during this class usage

class CMultivariateRegressionProblemNotNullWeightsView : public IMultivariateRegressionProblem,
	private CNotNullWeightsView<IMultivariateRegressionProblem> {
public:
	explicit CMultivariateRegressionProblemNotNullWeightsView( const IMultivariateRegressionProblem* inner );
	~CMultivariateRegressionProblemNotNullWeightsView() override = default;

	// The number of features
	int GetFeatureCount() const override;

	// The number of vectors in the input data set
	int GetVectorCount() const override;

	// Gets all input vectors as a matrix
	CSparseFloatMatrixDesc GetMatrix() const override;

	// The vector weight
	double GetVectorWeight( int index ) const override;

	// The length of the function value vector
	int GetValueSize() const override;

	// The value of the function on the input vector with the given index
	CFloatVector GetValue( int index ) const override;

private:
	// The inner problem
	const CPtr<const IMultivariateRegressionProblem> inner;
};

/////////////////////////////////////////////////////////////////////////////////////////

} // namespace NeoML

#include <ProblemWrappers.inl>

