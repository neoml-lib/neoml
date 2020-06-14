# Training Interfaces

<!-- TOC -->

- [Training Interfaces](#training-interfaces)
	- [For classification](#for-classification)
	- [For regression](#for-regression)

<!-- /TOC -->

All classification training classes provide the `ITrainingModel` interface; all regression training classes provide the `IRegressionTrainingModel` interface.

## For classification

The `ITrainingModel` interface exposes a `Train` method that accepts the input data as an object implementing the `IProblem` interface and returns the model implementing the `IModel` interface.

```c++
class NEOML_API ITrainingModel {
public:
	virtual ~ITrainingModel() = 0;

	// Train a classifier on the provided input data
	virtual CPtr<IModel> Train( const IProblem& trainingClassificationData ) = 0;
};
```

## For regression

The `IRegressionTrainingModel` interface exposes a `TrainRegression` method that accepts the input data as an object implementing the `IRegressionProblem` interface and returns the model implementing the `IRegressionModel` interface.

```c++
class NEOML_API IRegressionTrainingModel {
public:
	virtual ~RegressionITrainingModel() = 0;

	// Train a regression model using the provided input values
	virtual CPtr<IRegressionModel> TrainRegression( const IRegressionProblem& problem ) = 0;
};
```
