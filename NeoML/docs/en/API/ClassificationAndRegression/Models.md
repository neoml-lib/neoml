# Trained Model Interfaces

<!-- TOC -->

- [Trained Model Interfaces](#trained-model-interfaces)
    - [For classification](#for-classification)
    - [For regression](#for-regression)
    - [Load and save a model](#load-and-save-a-model)
        - [Save example](#save-example)
        - [Load example](#load-example)

<!-- /TOC -->

A trained model for classification or regression will implement one of the common interfaces: `IModel`, `IRegressionModel`,  or `IMultivariateRegressionModel`. These interfaces provide the methods that allow you to use the model for classification or prediction, save and load it.

## For classification

All classification models implement the `IModel` interface. It provides the `Classify` method for classifying data and the `Serialize` method for saving and loading the model to external storage.

```c++
class NEOML_API IModel : virtual public IObject {
public:
	virtual ~IModel() = 0;

	// The number of classes
	virtual int GetClassCount() const = 0;

	// Classifies the input vector and returns true if successful, false otherwise
	virtual bool Classify( const CSparseFloatVectorDesc& data, CClassificationResult& result ) const = 0;

	// Serializes the model
	virtual void Serialize( CArchive& archive ) = 0;
};
```

## For regression

The regression models implement the `IRegressionModel` (for functions that return a number) and the `IMultivariateRegressionModel` (for functions that return a vector) interfaces. They provide the `Predict` method for predicting the function value on a given vector and the `Serialize` method for saving and loading the model to external storage.

```c++
// Regression model for a function that returns a number
class IRegressionModel : virtual public IObject {
public:
	// Predict the function value on a vector
	virtual double Predict( const CSparseFloatVector& data ) const = 0;
	virtual double Predict( const CFloatVector& data ) const = 0;
	virtual double Predict( const CSparseFloatVectorDesc& desc ) const = 0;

	// Serialize the model
	virtual void Serialize( CArchive& archive ) = 0;
};

// Regression model for a function that returns a vector
class IMultivariateRegressionModel : virtual public IObject {
public:
	// Predict the function value on a vector
	virtual CFloatVector MultivariatePredict( const CSparseFloatVector& data ) const = 0;
	virtual CFloatVector MultivariatePredict( const CFloatVector& data ) const = 0;

	// Serialize the model
	virtual void Serialize( CArchive& archive ) = 0;
};
```

## Load and save a model

Use the `Serialize` method and the `CArchive` class to save and load a model.

### Save example

```c++
void StoreModel( CArchive& archive, IModel& model )
{
	CString modelName = GetModelName( &model );
	archive << modelName;
	model.Serialize( archive );
}
```

### Load example

```c++
CPtr<IModel> LoadModel( CArchive& archive )
{
	CString name;
	archive >> name;
	CPtr<IModel> result = CreateModel<IModel>( name );
	result->Serialize( archive );
	return result;
}
```
