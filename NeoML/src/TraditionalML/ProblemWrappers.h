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
	virtual int GetFeatureCount() const override;

	// The number of vectors in the data set
	virtual int GetVectorCount() const override;

	// Gets all vectors in the data set as a matrix
	virtual CSparseFloatMatrixDesc GetMatrix() const override;
	// Gets the vector weight
	virtual double GetVectorWeight( int index ) const override;

	// Gets the length of the function value vector
	virtual int GetValueSize() const override;
	// Gets the function value for the vector
	virtual CFloatVector GetValue( int index ) const override;

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
	virtual int GetFeatureCount() const override;

	// The number of vectors in the data set
	virtual int GetVectorCount() const override;
	// Gets all vectors in the data set as a matrix
	virtual CSparseFloatMatrixDesc GetMatrix() const override;
	// Gets the vector weight
	virtual double GetVectorWeight( int index ) const override;

	// Gets the length of the function value vector
	virtual int GetValueSize() const override;
	// Gets the function value for the vector
	virtual CFloatVector GetValue( int index ) const override;

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
	virtual int GetFeatureCount() const override;

	// The number of vectors in the data set
	virtual int GetVectorCount() const override;
	// Gets all vectors in the data set as a matrix
	virtual CSparseFloatMatrixDesc GetMatrix() const override;
	// Gets the vector weight
	virtual double GetVectorWeight( int index ) const override;

	// Gets the length of the function value vector
	virtual int GetValueSize() const override;
	// Gets the function value for the vector
	virtual CFloatVector GetValue( int index ) const override;

private:
	// The inner classification problem
	const CPtr<const IProblem> inner;
	// The values for each of the two classes
	CFloatVector classValues[2];
};

/////////////////////////////////////////////////////////////////////////////////////////

} // namespace NeoML

