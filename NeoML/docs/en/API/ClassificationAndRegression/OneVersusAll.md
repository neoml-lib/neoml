# One Versus All Classification COneVersusAll

<!-- TOC -->

- [One Versus All Classification COneVersusAll](#one-versus-all-classification-coneversusall)
	- [Training settings](#training-settings)
	- [Model](#model)
	- [Classification result](#classification-result)
	- [Sample](#sample)

<!-- /TOC -->

One vs. all method provides a way to solve a multi-class classification problem using only binary classifiers.

The original classification problem is represented as a series of binary classification problems, one for each class, that determine the probability for the object to belong to this class. The object will be assigned to the class for which the largest probability was returned.

In **NeoML** library this method is implemented by the `COneVersusAll` class. It exposes a `Train` method for creating a classification model.

## Training settings

The only parameter the algorithm requires is the pointer to the basic binary classification method, represented by an object that implements the [ITrainingModel](TrainingModels.md) interface.

## Model

The trained model is an ensemble of binary classification models. It implements the `IOneVersusAllModel` interface:

```c++
class NEOML_API IOneVersusAllModel : public IModel {
public:
	virtual ~IOneVersusAllModel() = 0;

	// Get the binary classifiers as IModel objects
	virtual const CObjectArray<IModel>& GetModels() const = 0;

	// Get the classification result with the info on normalized probabilities
	virtual bool ClassifyEx( const CSparseFloatVector& data,
		COneVersusAllClassificationResult& result ) const = 0;
	virtual bool ClassifyEx( const CSparseFloatVectorDesc& data,
		COneVersusAllClassificationResult& result ) const = 0;

	// Serialize the model
	virtual void Serialize( CArchive& ) = 0;
};
```

## Classification result

In addition to the standard `Classify` method, the one-versus-all model provides the `ClassifyEx` method that returns the result of the `COneVersusAllClassificationResult` type.

```c++
struct NEOML_API COneVersusAllClassificationResult : public CClassificationResult {
public:
	double SigmoidSum;
};
```

* *SigmoidSum* — the sigmoid sum which you may use to calculate the non-normalized probabilities returned by the binary classifiers.

## Sample

Here is a simple example of training a one-versus-all model using a linear binary classifier.

```c++
CLinear linear( EF_LogReg );
	
COneVersusAll oneVersusAll( linear );
CPtr<IModel> model = oneVersusAll.Train( *trainData );
```