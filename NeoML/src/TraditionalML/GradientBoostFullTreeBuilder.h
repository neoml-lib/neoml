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
#include <GradientBoostStatisticsSingle.h>
#include <GradientBoostStatisticsMulti.h>

namespace NeoML {

class CRegressionTree;
class CLinkedRegressionTree;

template<class T>
struct CThreadStatistics;
template<class T>
struct CGradientBoostNodeStatistics;

// Tree building parameters
struct CGradientBoostFullTreeBuilderParams {
	float L1RegFactor; // L1 regularization factor
	float L2RegFactor; // L2 regularization factor
	float MinSubsetHessian; // the minimum hessian value for a subtree
	int ThreadCount; // the number of processing threads to be used
	int MaxTreeDepth; // the maximum tree depth
	float PruneCriterionValue; // the value of criterion difference when the nodes should be merged (set to 0 to never merge)
	int MaxNodesCount; // the maximum number of nodes in a tree (set to NotFound == -1 for no limitation)
	float MinSubsetWeight; // the minimum subtree weight
	float DenseTreeBoostCoefficient; // the dense tree boost coefficient 
};

// Tree builder
template <class T>
class CGradientBoostFullTreeBuilder : public virtual IObject {
public:
	CGradientBoostFullTreeBuilder( const CGradientBoostFullTreeBuilderParams& params, CTextStream* logStream );

	// Builds the tree
	CPtr<CRegressionTree> Build( const CGradientBoostFullProblem& problem,
		const CArray<typename T::Type>& gradients, const typename T::Type& gradientsSum,
		const CArray<typename T::Type>& hessians, const typename T::Type& hessiansSum,
		const CArray<float>& weights, float weightsSum );

protected:
	virtual ~CGradientBoostFullTreeBuilder() {} // delete prohibited

private:
	const CGradientBoostFullTreeBuilderParams params; // classifier parameters
	CTextStream* const logStream; // the logging stream
	// The leaf cache
	// The index of each vector points to the leaf (of a partially built tree) to which this vector belongs
	// When starting, all vectors belong to root
	CArray<CGradientBoostNodeStatistics<T>*> classifyNodesCache;
	CArray<CGradientBoostNodeStatistics<T>*> curLevelStatistics; // current level statistucs
	CArray<int> splitFeatures; // the indices of the split features for this level
	CArray<int> vectorNodes; // distribution of the current level vectors into subtrees
	int nodesCount; // the number of nodes in the tree

	CPtr<CGradientBoostNodeStatistics<T>> initialize( const CGradientBoostFullProblem& problem,
		const typename T::Type& gradientSum, const typename T::Type& hessianSum, float weightSum );
	bool buildTreeLevel( const CGradientBoostFullProblem& problem, int level, const CArray<typename T::Type>& gradients,
		const CArray<typename T::Type>& hessians, const CArray<float>& weights );
	void distributeVectorsByNodes( const CGradientBoostFullProblem& problem, int level );
	void findSplits( const CGradientBoostFullProblem& problem, const CArray<typename T::Type>& gradients,
		const CArray<typename T::Type>& hessians, const CArray<float>& weights );
	void findBinarySplits( int threadNumber, const CArray<typename T::Type>& gradients, const CArray<typename T::Type>& hessians,
		const CArray<float>& weights, int feature, const int* ptr, int size );
	void findSplits( int threadNumber, const CArray<typename T::Type>& gradients, const CArray<typename T::Type>& hessians,
		const CArray<float>& weights, int feature, const CFloatVectorElement* ptr, int size );
	void checkSplit( int feature, float firstValue, float secondValue, CThreadStatistics<T>& statistics ) const;
	void mergeThreadResults();
	bool split();
	bool prune( CGradientBoostNodeStatistics<T>& node ) const;
	CPtr<CLinkedRegressionTree> buildModel( const CArray<int>& usedFeatures, CGradientBoostNodeStatistics<T>& node ) const;
};

} // namespace NeoML
