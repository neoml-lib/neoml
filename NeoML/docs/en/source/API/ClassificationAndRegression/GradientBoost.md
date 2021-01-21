# Gradient Boosting Classifier CGradientBoost

<!-- TOC -->

- [Gradient Boosting Classifier CGradientBoost](#gradient-boosting-classifier-cgradientboost)
	- [Training setting](#training-settings)
		- [Loss function](#loss-function)
		- [Tree builder](#tree-builder)
	- [Model](#model)
		- [For classification](#for-classification)
		- [For regression](#for-regression)
	- [QuickScorer model](#quickscorer-model)
		- [QuickScorer for classification](#quickscorer-for-classification)
		- [QuickScorer for regression](#quickscorer-for-regression)
	- [Sample](#sample)

<!-- /TOC -->

Gradient boosting method for classification and regression creates an ensemble of decision trees using random subsets of features and input data.

In **NeoML** library this method is implemented by the  `CGradientBoost` class. It exposes a `Train` method for creating a classification model and a `TrainRegression` method for creating a regression model.

The algorithm only accepts continuous features. If your data is characterized by discrete features you will need to transform them into continuous ones (for example, using binarization).

## Training settings

The parameters are represented by a `CGradientBoost::CParams` structure.

- *LossFunction* — the loss function to be used.
- *IterationsCount* — the maximum number of iterations (that is, the number of trees in the ensemble).
- *LearningRate* — the multiplier for each classifier.
- *Subsample* — the fraction of input data that is used for building one tree; may be from 0 to 1.
- *Subfeature* — the fraction of features that is used for building one tree; may be from 0 to 1.
- *Random* — the random numbers generator for selecting *Subsample* vectors and *Subfeature* features out of the whole.
- *MaxTreeDepth* — the maximum depth of each tree.
- *MaxNodesCount* — the maximum number of nodes in a tree (set to `-1` for no limitation).
- *L1RegFactor* — the L1 regularization factor.
- *L2RegFactor* — the L2 regularization factor.
- *PruneCriterionValue* — the value of criterion difference when the nodes should be merged (set to `0` to never merge).
- *ThreadCount* — the number of processing threads to be used while training the model.
- *TreeBuilder* — the type of tree builder used (*GBTB_Full* or *GBTB_FastHist*, see [below](#tree-builder));
- *MaxBins* — the largest possible histogram size to be used in *GBTB_FastHist* mode;
- *MinSubsetWeight* — the minimum subtree weight (set to `0` to have no lower limit).

Note that the *L1RegFactor*, *L2RegFactor*, *PruneCriterionValue* parameters are applied to the values depending on the total vector weight in the corresponding tree node. Therefore when setting up these parameters, you need to take into consideration the number and weights of the vectors in your training data set.

### Loss function

The following loss functions are supported:

- *LF_Exponential* — [classification only] exponential loss function: `L(x, y) = exp(-(2y - 1) * x)`;
- *LF_Binomial* — [classification only] binomial loss function: `L(x, y) = ln(1 + exp(-x)) - x * y`;
- *LF_SquaredHinge* — [classification only] smoothed square hinge: `L(x, y) = max(0, 1 - (2y - 1)* x) ^ 2`;
- *LF_L2* — quadratic loss function: `L(x, y) = (y - x)^2 / 2`.

### Tree builder

The algorithm supports two different tree builder modes:

- *GBTB_Full* — all the feature values are used for splitting the nodes.
- *GBTB_FastHist* — the steps of a histogram created from the feature values will be used for splitting the nodes. The *MaxBins* parameter limits the possible size of the histogram.

## Model

The algorithm can train a classification model described by the `IGradientBoostModel` interface or a regression model described by the `IGradientBoostRegressionModel` interface.

### For classification

```c++
// Gradient boosting model interface
class NEOML_API IGradientBoostModel : public IModel {
public:
	virtual ~IGradientBoostModel() = 0;

	// Get the tree ensemble
	virtual const CArray<CGradientBoostEnsemble>& GetEnsemble() const = 0;

	// Serialize the model
	virtual void Serialize( CArchive& ) = 0;

	// Get the learning rate
	virtual double GetLearningRate() const = 0;

	// Get the loss function
	virtual CGradientBoost::TLossFunction GetLossFunction() const = 0;

	// Get the classification results for all tree ensembles [1..k], 
	// with k taking values from 1 to the total number of trees
	virtual bool ClassifyEx( const CSparseFloatVector& data, CArray<CClassificationResult>& results ) const = 0;
	virtual bool ClassifyEx( const CSparseFloatVectorDesc& data, CArray<CClassificationResult>& results ) const = 0;

	// Calculate feature usage statistics
	// Returns the number of times each feature was used for node splitting
	virtual void CalcFeatureStatistics( int maxFeature, CArray<int>& result ) const = 0;

	// Reduce the number of trees in the ensemble to the given cutoff value
	virtual void CutNumberOfTrees( int numberOfTrees ) = 0;
};
```

### For regression

```c++
// Gradient boosting regression model interface
class NEOML_API IGradientBoostRegressionModel : public IRegressionModel, public IMultivariateRegressionModel {
public:
	virtual ~IGradientBoostRegressionModel() = 0;
	
    // Get the tree ensemble
	virtual const CArray<CGradientBoostEnsemble>& GetEnsemble() const = 0;

	// Serialize the model
	virtual void Serialize( CArchive& ) = 0;

	// Get the learning rate
	virtual double GetLearningRate() const = 0;

	// Get the loss function
	virtual CGradientBoost::TLossFunction GetLossFunction() const = 0;

	// Calculate feature usage statistics
	// Returns the number of times each feature was used for node splitting
	virtual void CalcFeatureStatistics( int maxFeature, CArray<int>& result ) const = 0;
};
```

## QuickScorer model

The library also provides the [QuickScorer](http://ecmlpkdd2017.ijs.si/papers/paperID718.pdf) algorithm for speeding up the prediction rate of the model. In some cases, the prediction rate of a model optimized with this algorithm can increase up to 10 times.

`CGradientBoostQuickScorer` class is used for optimization.

```c++
// QuickScorer model optimization method
class NEOML_API CGradientBoostQuickScorer {
public:
	// Build a IGradientBoostQSModel using the given IGradientBoostModel
	CPtr<IGradientBoostQSModel> Build( const IGradientBoostModel& gradientBoostModel );

	// Build a IGradientBoostQSRegressionModel using the given IGradientBoostRegressionModel
	CPtr<IGradientBoostQSRegressionModel> BuildRegression( const IGradientBoostRegressionModel& gradientBoostModel );
};
```

The models produced by this class implement the `IGradientBoostQSModel` or `IGradientBoostQSRegressionModel` interface.

### QuickScorer for classification

```c++
// Optimized classification model interface
class NEOML_API IGradientBoostQSModel : public IModel {
public:
	virtual ~IGradientBoostQSModel();
    
	// Serialize the model
	virtual void Serialize( CArchive& ) = 0;

	// Get the learning rate
	virtual double GetLearningRate() const = 0;

	// Get the classification results for all tree ensembles [1..k], 
	// with k taking values from 1 to the total number of trees
	virtual bool ClassifyEx( const CSparseFloatVector& data, CArray<CClassificationResult>& results ) const = 0;
	virtual bool ClassifyEx( const CSparseFloatVectorDesc& data, CArray<CClassificationResult>& results ) const = 0;
};
```

### QuickScorer for regression

```c++
// Optimized regression model interface
class IGradientBoostQSRegressionModel : public IRegressionModel {
public:
	// Serialize the model
	virtual void Serialize( CArchive& ) = 0;

	// Get the learning rate
	virtual double GetLearningRate() const = 0;
};
```

## Sample

Here is a simple example of training a model by gradient boosting. The input data is represented by an object implementing the [`IProblem`](Problems.md) interface.

```c++
CPtr<IModel> buildModel( IProblem* data )
{
	CGradientBoost::CParams params;
	params.LossFunction = CGradientBoost::LF_Exponential;
	params.IterationsCount = 100;
	params.LearningRate = 0.1;
	params.MaxTreeDepth = 10;
	params.ThreadCount = 4;
	params.Subsample = 0.5;
	params.Subfeature = 1;
	params.MaxBins = 64;
	params.TreeBuilder = GBTB_FastHist;
	params.MinSubsetWeight = 10;

	CGradientBoost boosting( params );
	return boosting.Train( *problem );
}
```
