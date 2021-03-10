# Solving Classification and Regression Problems

<!-- TOC -->

- [Solving Classification and Regression Problems](#solving-classification-and-regression-problems)
	- [Algorithms](#algorithms)
		- [Gradient tree boosting](#gradient-tree-boosting)
		- [Linear classifier](#linear-classifier)
		- [Support-vector machine](#support-vector-machine)
		- [Decision tree](#decision-tree)
		- [One versus all method](#one-versus-all-method)
	- [Auxiliary interfaces](#auxiliary-interfaces)
		- [Problem interface](#problem-interface)
		- [Training interfaces](#training-interfaces)
		- [Model interfaces](#model-interfaces)
		- [Classification result](#classification-result)

<!-- /TOC -->

In **NeoML** library, you can find various methods for solving classification and regression problems. 

## Algorithms

### Gradient tree boosting

Gradient boosting creates an ensemble of decision trees for solving classification and regression problems. The ensemble is populated step-by-step; on each step, a new decision tree is built using random subsets of features and training data.

Gradient boosting is implemented by the [CGradientBoost](GradientBoost.md) class, while the trained models are represented by the `IGradientBoostModel` and `IGradientBoostRegressionModel` interfaces for classification and regression respectively.

### Linear classifier

A linear binary classifier finds a hyperplane that divides the feature space in half.

It is implemented by the [CLinearBinaryClassifierBuilder](Linear.md) class that trains the `ILinearBinaryModel` model for classification and the `ILinearRegressionModel` model for linear regression.

### Support-vector machine

Support-vector machine translates the input data into vectors in a high-dimensional space and searches for a maximum-margin dividing hyperplane.

It is implemented by the [CSvmBinaryClassifierBuilder](Svm.md) class. The trained model is represented by the `ILinearBinaryModel` or `ISvmBinaryModel` interface, depending on the type of kernel used.

### Decision tree

This classification method involves comparing the object features with a set of threshold values; the result tells us to move to one of the children nodes. Once we reach a leaf node we assign the object to the class this node represents.

The decision tree is implemented by the [CDecisionTree](DecisionTree.md) class, while the trained model implements the `IDecisionTreeModel` or [`IOneVersusAllModel`](OneVersusAll.md#model) interface depending on the number of classes.

### One versus all method

This method helps solve a multi-class classification problem using only binary classifiers.

It is implemented by the [COneVersusAll](OneVersusAll.md) class. The trained multi-class classification model implements the `IOneVersusAllModel` interface.

## Auxiliary interfaces

All the methods for model training implement common interfaces, accept the input data of the same type and train models that may be accessed using the common interface.

### Problem interface

The input data for training is passed as a pointer to an object implementing one of the following interfaces:

- `IProblem` — for classification
- `IRegressionProblem` — for regression of a function that returns a number
- `IMultivariateProblem` — for regression of a function that returns a vector

See the detailed [description](Problems.md).

### Training interfaces

All classification training algorithms implement the `ITrainingModel` interface; all regression training algorithms implement the `IRegressionTrainingModel` interface. [See more...](TrainingModels.md)

### Model interfaces

The trained models implement the `IModel`, `IRegressionModel`, `IMultivariateRegressionModel` interfaces. Read more details [here](Models.md).

### Classification result

The result of classification returned by the `Classify` and `ClassifyEx` methods of the trained models is represented by a `CClassificationResult` structure.

```c++
struct NEOML_API CClassificationResult {
public:
	int PreferredClass;
	CClassificationProbability ExceptionProbability;
	CArray<CClassificationProbability> Probabilities;
};
```

- *PreferredClass* — the number of the class to which the input object is assigned
- *ExceptionProbability* — the probability that the input object fits none of the classes
- *Probabilities* — the array of probabilities for the object to belong to each of the classes
