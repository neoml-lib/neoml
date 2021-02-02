/* Copyright © 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#pragma once

#include <RegressionTreeModel.h>
#include <GradientBoostFastHistProblem.h>
#include <GradientBoostVectorSetStatistics.h>
#include <NeoML/TraditionalML/Model.h>

namespace NeoML {

// Tree builder
class CGradientBoostFastHistTreeBuilder : public virtual IObject {
public:
	// Tree building parameters
	struct CParams {
		float L1RegFactor; // the L1 regularization factor
		float L2RegFactor; // the L2 regularization factor
		float MinSubsetHessian; // the minimum hessian value for a subtree
		int ThreadCount; // the number of processing threads to be used
		int MaxTreeDepth; // the maximum tree depth
		float PruneCriterionValue; // the value of criterion difference when the nodes should be merged (set to 0 to never merge)
		int MaxNodesCount; // the maximum number of nodes in a tree (set to NotFound == -1 for no limitation)
		int MaxBins; // the maximum histogram size for a feature
		float MinSubsetWeight; // the minimum subtree weight
	};

	CGradientBoostFastHistTreeBuilder( const CParams& params, CTextStream* logStream );

	// Builds a tree
	CPtr<IRegressionTreeModel> Build( const CGradientBoostFastHistProblem& problem,
		const CArray<double>& gradients, const CArray<double>& hessians, const CArray<float>& weights );

protected:
	virtual ~CGradientBoostFastHistTreeBuilder() {} // delete prohibited

private:
	const CParams params; // classifier parameters
	CTextStream* const logStream; // the logging stream

	// A node in the tree
	struct CNode {
		int Level; // the level of the node in the final tree
		int VectorSetPtr; // a pointer to the start of the vector set of the node
		int VectorSetSize; // the size of the vector set of the node
		int HistPtr; // a pointer to the histogram created on the vectors of the node
		CGradientBoostVectorSetStatistics<double> Statistics; // statistics of the vectors of the node
		int SplitFeatureId; // the identifier of the feature used to split this node
		int Left; // the pointer to the left child
		int Right; // the pointer to the right child

		CNode( int level, int vectorSetPtr, int vectorSetSize ) :
			Level( level ),
			VectorSetPtr( vectorSetPtr ),
			VectorSetSize( vectorSetSize ),
			HistPtr( NotFound ),
			SplitFeatureId( NotFound ),
			Left( NotFound ),
			Right( NotFound ),
			Statistics( 1 )
		{}
	};

	int histSize; // histogram size
	CArray<CNode> nodes; // the final tree nodes
	CArray<int> nodeStack; // the stack used to build the tree using depth-first search
	CArray<int> vectorSet; // the array that stores the vector sets for the nodes
	CArray<int> freeHists; // free histograms list
	CArray<CGradientBoostVectorSetStatistics<double>> histStats; // the array for storing histograms
	CArray<int> idPos; // the identifier positions in the current histogram
	CArray<int> histIds; // histogram bins identifiers
	CArray<CGradientBoostVectorSetStatistics<double>> tempHistStats; // a temporary array for building histograms
	int classCount; // the dimension of prediction value

	// Caching the buffers
	mutable CArray<double> splitGainsByThreadBuffer;
	mutable CArray<int> splitIdsBuffer;

	void initVectorSet( int size );
	void initHistData( const CGradientBoostFastHistProblem& problem );
	int allocHist();
	void freeHist( int ptr );
	void subHist( int firstPtr, int secondPtr );
	void buildHist( const CGradientBoostFastHistProblem& problem, const CNode& node,
		const CArray<double>& gradients, const CArray<double>& hessians, const CArray<float>& weights,
		CGradientBoostVectorSetStatistics<double>& stats );
	void addVectorToHist( const int* vectorPtr, int vectorSize, double gradients, double hessian, float weight,
		CGradientBoostVectorSetStatistics<double>* stats );
	int evaluateSplit( const CGradientBoostFastHistProblem& problem, const CNode& node ) const;
	void applySplit( const CGradientBoostFastHistProblem& problem, int node, int& leftNode, int& rightNode );
	bool prune( int node );
	CPtr<CRegressionTreeModel> buildTree( int node, const CArray<int>& featureIndexes, const CArray<float>& cuts ) const;
};

} // namespace NeoML
