# Decision Tree Classifier CDecisionTree

<!-- TOC -->

- [Decision Tree Classifier CDecisionTree](#decision-tree-classifier-cdecisiontree)
	- [Training settings](#training-settings)
	- [Model](#model)
	- [Sample](#sample)

<!-- /TOC -->

Decision tree is a classification method that involves comparing the object features with a set of threshold values; the result tells us to move to one of the children nodes. Once we reach a leaf node we assign the object to the class this node represents.

In **NeoML** library this method is implemented by the  `CDecisionTree` class. It exposes a `Train` method that allows you to train a multi-class classification model.

## Training settings

The parameters are represented by a `CDecisionTree::CParams` structure.

- *MinContinuousSubsetSize* — [for continuous features] the minimum number of vectors corresponding to a node subtree.
- *MinDiscreteSubsetSize* — [for discrete features] the minimum number of vectors corresponding to a node subtree.
- *MinDiscreteSubsetPart* — [for discrete features] the minimum weight of the vectors in a subtree relative to the parent node weight (may be from 0 to 1).
- *MinContinuousSubsetPart* — [for continuous features] the minimum weight of the vectors in a subtree relative to the parent node weight (may be from 0 to 1).
- *MinSplitSize* — the minimum number of vectors in a node subtree when it may be divided further.
- *MaxTreeDepth* — the maximum depth of the tree.
- *MaxNodesCount* — the maximum number of nodes.
- *SplitCriterion* — the criterion for subset splitting.
- *ConstNodeThreshold* — the ratio of the equal elements in the subset which should be the threshold for creating a constant node (may be from 0 to 1).
- *RandomSelectedFeaturesCount* — no more than this number of randomly selected features will be used for each node. Set the value to `-1` to use all features every time.

## Model

The trained model is a tree with every non-leaf node containing a pair (feature index, a set of its values), and every leaf node containing the probability for the object to belong to one of the classes.

To classify a given vector, the algorithm travels along the tree from the root node, in each node comparing the value of the specified feature with the set threshold; the result of comparison selects the next node.

The model is described by an `IDecisionTreeModel` interface:

```c++
// Trained classification model interface
class NEOML_API IDecisionTreeModel : public IModel {
public:
	virtual ~IDecisionTreeModel() = 0;

	// Gets the number of children nodes
	virtual int GetChildrenCount() const = 0;

	// Gets the child node with the specified index
	virtual CPtr<IDecisionTreeModel> GetChild( int index ) const = 0;

	// Gets the node information
	virtual void GetNodeInfo( CDecisionTreeNodeInfo& info ) const = 0;

	// Serialize the model
	virtual void Serialize( CArchive& ) = 0;
};
```

## Sample

See a simple example of training a decision tree classifier. The input data set contains only vectors with continuous features.

```c++
CPtr<IDecisionTreeModel> buildModel( IProblem* data )
{
	CDecisionTree::CParams param;
	param.MinContinuousSubsetPart = 0.10; // Each subtree should contain no less than 10% of all nodes
	param.MinContinuousSubsetSize = 128; // Suppose only continuous features are used
	param.MinSplitSize = 16; // Each subtree should contain no less than 16 nodes
	param.MaxTreeDepth = 10; // The tree depth shouldn't be more than 10
	param.SplitCriterion = CDecisionTree::SC_InformationGain;

	CDecisionTree builder( param );

	return dynamic_cast<IDecisionTreeModel*>( builder.Train( data ).Ptr() );
}
```