# Input Data Interfaces

<!-- TOC -->

- [Input Data Interfaces](#input-data-interfaces)
	- [For classification](#for-classification)
		- [Sample implementation](#sample-implementation)
	- [For regression](#for-regression)

<!-- /TOC -->

If you are going to train a classification or regression model, you will need to represent the input data as an object implementing one of the interfaces described below.

## For classification

The input data set for training a classification model should be represented by an `IProblem` interface. The object should provide all the information for training. The main data is a set of vectors, and each vector contains the feature values for a single object. `IProblem` should also present the information about the classes and features to be used:

- *GetClassCount* — the number of classes
- *GetFeatureCount* — the number of features (that is, a single vector length)
- *IsDiscreteFeature* — indicates if the feature with the given index is discrete
- *GetVectorCount* — the number of vectors (and objects to be classified)
- *GetClass* — the class of the vector with the given number; classes are numbered from 0 to (*GetClassCount* - 1)
- *GetVector* — the vector with the given index
- *GetMatrix* — the whole training set as a matrix (of the *GetFeatureCount* * *GetVectorCount* size)
- *GetVectorWeight* — the vector weight

```c++
class NEOML_API IProblem : virtual public IObject {
public:
	virtual ~IProblem() = 0;

	// The number of classes
	virtual int GetClassCount() const = 0;

	// The number of features
	virtual int GetFeatureCount() const = 0;

	// Indicates if the feature is discrete
	virtual bool IsDiscreteFeature( int index ) const = 0;

	// The number of input vectors
	virtual int GetVectorCount() const = 0;

	// Get the number of class [0, GetClassCount()) for the given vector
	virtual int GetClass( int index ) const = 0;

	// Get a vector
	virtual CFloatVectorDesc GetVector( int index ) const = 0;

	// The training set as a matrix
	virtual CFloatMatrixDesc GetMatrix() const = 0;

	// The vector weight
	virtual double GetVectorWeight( int index ) const = 0;
};
```

### Sample implementation

The library provides the `CMemoryProblem` class that implements the `IProblem` interface. It stores all data in memory.

## For regression

The input data set for training a regression model should be represented by an object implementing an `IRegressionProblem` interface (if the function returns a number) or an `IMultivariateRegressionProblem` interface (if the function returns a vector). The base interface for both is `IBaseRegressionProblem`.

This object should contain all data for model training. The main data is a set of vectors, and each vector contains the feature values for a single object:

- *GetFeatureCount* — the number of features (that is, a single vector length)
- *GetVectorCount* — the number of vectors (and objects to be classified)
- *GetVector* — the vector with the given index
- *GetMatrix* — the whole training set as a matrix (of the *GetFeatureCount* * *GetVectorCount* size)
- *GetVectorWeight* — the vector weight
- *GetValue* — the value of the function on the given vector (a single number for `IRegressionProblem`, a vector for `IMultivariateRegressionProblem`);
- *GetValueSize* — (for `IMultivariateRegressionProblem` only) the length of the vector function value.

```c++
class IBaseRegressionProblem : virtual public IObject {
public:
	// The number of features
	virtual int GetFeatureCount() const = 0;

	// The number of input vectors
	virtual int GetVectorCount() const = 0;

	// Get a vector
	virtual CFloatVectorDesc GetVector( int index ) const = 0;

	// The training set as a matrix
	virtual CFloatMatrixDesc GetMatrix() const = 0;

	// The vector weight
	virtual double GetVectorWeight( int index ) const = 0;
};

// The data for training a regression model of a function with number value
class IRegressionProblem : public IBaseRegressionProblem {
public:
	// The function value on a vector
	virtual double GetValue( int index ) const = 0;
};

// The data for training a regression model of a function with vector value
class IMultivariateRegressionProblem : public IBaseRegressionProblem {
public:
	// The length of the function value
	virtual int GetValueSize() const = 0;
	// The function value on a vector
	virtual CFloatVector GetValue( int index ) const = 0;
};
```
