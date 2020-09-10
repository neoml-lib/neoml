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
#include <NeoML/TraditionalML/SparseFloatMatrix.h>
#include <float.h>

namespace NeoML {

// The default sampling value
const int DefaultDiscretizationValue = 100;

// The input data for classification training
// This interface is implemented by the client
class NEOML_API IProblem : virtual public IObject {
public:
	virtual ~IProblem();

	// The number of classes
	virtual int GetClassCount() const = 0;

	// The number of features
	virtual int GetFeatureCount() const = 0;

	// Indicates if the specified feature is discrete
	virtual bool IsDiscreteFeature( int index ) const = 0;

	// The number of vectors
	virtual int GetVectorCount() const = 0;

	// The correct class number for a vector with a given index in [0, GetClassCount())
	virtual int GetClass( int index ) const = 0;

	// The correct class in case of binary classification: +1 or -1
	double GetBinaryClass( int index ) const { return ( GetClass( index ) != 0 ) ? 1. : -1; }

	// Gets all input vectors as a matrix of size GetVectorCount() x GetFeaturesCount()
	virtual CSparseFloatMatrixDesc GetMatrix() const = 0;

	// The vector weight
	virtual double GetVectorWeight( int index ) const = 0;

	// The sampling value
	// For discrete features, it is reasonable to set this value to the number of different values the feature can take
	virtual int GetDiscretizationValue( int ) const { return DefaultDiscretizationValue; }
};

// The input data for regression training
// This interface is implemented by the client
class NEOML_API IBaseRegressionProblem : virtual public IObject {
public:
	virtual ~IBaseRegressionProblem();

	// The number of features
	virtual int GetFeatureCount() const = 0;

	// The number of vectors in the input data set
	virtual int GetVectorCount() const = 0;

	// Gets all input vectors as a matrix of size GetVectorCount() x GetFeaturesCount()
	virtual CSparseFloatMatrixDesc GetMatrix() const = 0;

	// The vector weight
	virtual double GetVectorWeight( int index ) const = 0;
};

// The input data for regression in case the function returns a number
class NEOML_API IRegressionProblem : public IBaseRegressionProblem {
public:
	virtual ~IRegressionProblem();

	// The value of the function on the vector with the given index
	virtual double GetValue( int index ) const = 0;
};

// The input data for regression in case the function returns a vector
class NEOML_API IMultivariateRegressionProblem : public IBaseRegressionProblem {
public:
	virtual ~IMultivariateRegressionProblem();

	// The length of the function value vector
	virtual int GetValueSize() const = 0;

	// The value of the function on the input vector with the given index
	virtual CFloatVector GetValue( int index ) const = 0;
};

// The interface for accumulating vectors in a data set
class NEOML_API IDataAccumulator : public IProblem {
public:
	virtual ~IDataAccumulator();

	// Adds a vector
	virtual void AddVector( const CSparseFloatVector& vector, double weight, int classIndex ) = 0;

	// Stops accumulating data
	virtual void Finish() = 0;
};

} // namespace NeoML
