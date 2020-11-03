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

#include <GradientBoostFullProblem.h>
#include <NeoML/TraditionalML/Model.h>

namespace NeoML {

struct CThreadStatistics;
struct CGradientBoostNodeStatistics;
class CRegressionTreeModel;

// Tree builder
class CGradientBoostFullTreeBuilder : public virtual IObject {
public:
	// Tree building parameters
	struct CParams {
		float L1RegFactor; // L1 regularization factor
		float L2RegFactor; // L2 regularization factor
		float MinSubsetHessian; // the minimum hessian value for a subtree
		int ThreadCount; // the number of processing threads to be used
		int MaxTreeDepth; // the maximum tree depth
		float PruneCriterionValue; // the value of criterion difference when the nodes should be merged (set to 0 to never merge)
		int MaxNodesCount; // the maximum number of nodes in a tree (set to NotFound == -1 for no limitation)
		float MinSubsetWeight; // the minimum subtree weight
	};

	CGradientBoostFullTreeBuilder( const CParams& params, CTextStream* logStream );

	// Builds the tree
	CPtr<IRegressionModel> Build( const CGradientBoostFullProblem& problem,
		const CArray<double>& gradients, double gradientsSum, const CArray<double>& hessians, double hessiansSum,
		const CArray<float>& weights, float weightsSum );

protected:
	virtual ~CGradientBoostFullTreeBuilder() {} // delete prohibited

private:
	const CParams params; // classifier parameters
	CTextStream* const logStream; // the logging stream
	// The leaf cache
	// The index of each vector points to the leaf (of a partially built tree) to which this vector belongs
	// When starting, all vectors belong to root
	CArray<CGradientBoostNodeStatistics*> classifyNodesCache;
	CArray<CGradientBoostNodeStatistics*> curLevelStatistics; // current level statistucs
	CArray<int> splitFeatures; // the indices of the split features for this level
	CArray<int> vectorNodes; // distribution of the current level vectors into subtrees
	int nodesCount; // the number of nodes in the tree

	CPtr<CGradientBoostNodeStatistics> initialize( const CGradientBoostFullProblem& problem,
		double gradientSum, double hessianSum, float weightSum );
	bool buildTreeLevel( const CGradientBoostFullProblem& problem, int level, const CArray<double>& gradients,
		const CArray<double>& hessians, const CArray<float>& weights );
	void distributeVectorsByNodes( const CGradientBoostFullProblem& problem, int level );
	void findSplits( const CGradientBoostFullProblem& problem, const CArray<double>& gradients,
		const CArray<double>& hessians, const CArray<float>& weights );
	void findBinarySplits( int threadNumber, const CArray<double>& gradients, const CArray<double>& hessians,
		const CArray<float>& weights, int feature, const int* ptr, int size );
	void findSplits( int threadNumber, const CArray<double>& gradients, const CArray<double>& hessians,
		const CArray<float>& weights, int feature, const CFloatVectorElement* ptr, int size );
	void checkSplit( int feature, float firstValue, float secondValue, CThreadStatistics& statistics ) const;
	void mergeThreadResults();
	bool split();
	bool prune( CGradientBoostNodeStatistics& node ) const;
	CPtr<CRegressionTreeModel> buildModel( const CArray<int>& usedFeatures, CGradientBoostNodeStatistics& node ) const;
};

} // namespace NeoML
