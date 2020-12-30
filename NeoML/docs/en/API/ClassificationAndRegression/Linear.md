# Linear Classifier CLinear

<!-- TOC -->

- [Linear Classifier CLinear](#linear-classifier-clinear)
	- [Training settings](#training-settings)
		- [Loss function](#loss-function)
	- [Model](#model)
		- [For classification](#for-classification)
		- [For regression](#for-regression)
	- [Sample](#sample)

<!-- /TOC -->

A linear classifier finds a hyperplane that divides the feature space in half.

In **NeoML** library this method is implemented by the  `CLinear` class. It exposes a `Train` method for creating a classification model and a `TrainRegression` method for creating a linear regression model.

## Training settings

The parameters are represented by a `CLinear::CParams` structure.

- *Function* — the loss function.
- *MaxIterations* — the maximum number of iterations allowed.
- *ErrorWeight* — the error weight relative to the regularization coefficient.
- *SigmoidCoefficients* — the predefined sigmoid function coefficients.
- *Tolerance* — the stop criterion.
- *NormalizeError* — specifies if the error should be normalized.
- *L1Coeff* — the L1 regularization coefficient; set to `0` to use the L2 regularization instead.
- *ThreadCount* — the number of processing threads to be used while training the model.

### Loss function

The following loss functions are supported:

- *EF_SquaredHinge* — squared hinge function
- *EF_LogReg* — logistical regression function
- *EF_SmoothedHinge* — one half of a hyperbolic function
- *EF_L2_Regression* — the L2 regression function

## Model

### For classification

```c++
// Classification model interface
class NEOML_API ILinearBinaryModel : public IModel {
public:
	virtual ~ILinearBinaryModel() = 0;

	// Get the dividing hyperplane
	virtual CFloatVector GetPlane() const = 0;

	// Serialize the model
	virtual void Serialize( CArchive& ) = 0;
};
```

### For regression

```c++
// Regression model interface
class NEOML_API ILinearRegressionModel : public IRegressionModel {
public:
	virtual ~ILinearRegressionModel() = 0;

	// Get the dividing hyperplane
	virtual CFloatVector GetPlane() const = 0;

	// Serialize the model
	virtual void Serialize( CArchive& ) = 0;
};
```

## Sample

Here is a simple example of training a linear classification model. The input data is represented by an object implementing the [`IProblem`](Problems.md) interface.

```c++
CPtr<Model> buildModel( IProblem* data )
{
	CLinear::CParams params;
	params.Function = EF_SquaredHinge;
	params.L1Coeff = 0.05;
	params.ThreadCount = 4;

	CLinear builder( params );
	return builder.Train( *data );
}
```