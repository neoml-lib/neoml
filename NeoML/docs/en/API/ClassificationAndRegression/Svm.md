# Support-Vector Machine CSvm

- [Support-Vector Machine CSvm](#support-vector-machine-scvm)
	- [Training settings](#training-settings)
	- [Model](#model)
	- [Sample](#sample)

Support-vector machine translates the input data into vectors in a high-dimensional space and searches for a maximum-margin dividing hyperplane.

In **NeoML** library this method is implemented by the `CSvm` class. It exposes a `Train` method for creating a model for binary classification.

## Training settings

The parameters are represented by a `CSvm::CParams` structure.

- *KernelType* — the type of the kernel function
- *ErrorWeight* — the error weight relative to the regularization function
- *MaxIterations* — the maximum number of algorithm iterations
- *Degree* — the degree for a gaussian kernel
- *Gamma* — the kernel coefficient (for `KT_Poly`, `KT_RBF`, `KT_Sigmoid`)
- *Coeff0* — the kernel free term (for `KT_Poly`, `KT_Sigmoid`)
- *Tolerance* — the algorithm precision, the stop criterion
- *ThreadCount* — the number of processing threads to be used while training
- *MulticlassMode* - the approach used in multiclass task

## Model

The trained model implements the [`ILinearBinaryModel`](Linear.md#for-classification) interface if a `KT_Linear` kernel is used; or `MuticlassMode` model if number of classes > 2; otherwise, it implements the `ISvmBinaryModel` interface.

```c++
// SVM binary classifier interface
class ISvmBinaryModel : public IModel {
public:
	virtual ~ISvmBinaryModel();

	// Get the kernel
	virtual CSvmKernel::TKernelType GetKernelType() const = 0;

	// Get the support vectors
	virtual CSparseFloatMatrix GetVectors() const = 0;

	// Get the support vector coefficients
	virtual const CArray<double>& GetAlphas() const = 0;

	// Get the free term
	virtual double GetFreeTerm() const = 0;

	// Serialize the model
	virtual void Serialize( CArchive& ) = 0;
};
```

## Sample

Here is a simple example of training a support-vector classification model. The input data is represented by an object implementing the [`IProblem`](Problems.md) interface.

```c++
CPtr<Model> buildModel( IProblem* data )
{
	CSvm::CParams params( CSvmKernel::KT_RBF );
	CSvm builder( params );
	return builder.Train( *data );
}
```