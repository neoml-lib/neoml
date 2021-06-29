# One Versus One Classification COneVersusOne

<!-- TOC -->

- [One Versus One Classification COneVersusOne](#one-versus-one-classification-coneversusone)
	- [Training settings](#training-settings)
	- [Model](#model)
	- [Classification result](#classification-result)
	- [Sample](#sample)

<!-- /TOC -->

One vs. one method provides a way to solve a multi-class classification problem using only binary classifiers.

The original classification problem is represented as a series of binary classification problems, one for each pair of classes, that determine the pairwise probabilities for the object to belong to one class or another.

Afterfwards the optimal probabilities for each class are found by solving an optimization task, which is described in Section 4 of [this article](https://www.csie.ntu.edu.tw/~cjlin/papers/svmprob/svmprob.pdf).

In **NeoML** library this method is implemented by the `COneVersusOne` class. It exposes a `Train` method for creating a classification model.

## Training settings

The only parameter the algorithm requires is the pointer to the basic binary classification method, represented by an object that implements the [ITrainingModel](TrainingModels.md) interface.

## Model

The trained model is an ensemble of binary classification models. It implements the [`IModel` interface](Models.md#for-classification).

## Classification result

It provides the standard `Classify` method which writes the result into the given [`CClassificationResult`](README.md#classification-result).

## Sample

Here is a simple example of training a one-versus-one model using a linear binary classifier.

```c++
CLinear linear( EF_LogReg );

COneVersusOne oneVersusOne( linear );
CPtr<IModel> model = oneVersusOne.Train( *trainData );
```
