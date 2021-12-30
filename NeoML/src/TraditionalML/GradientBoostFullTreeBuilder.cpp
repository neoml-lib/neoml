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

#include <common.h>
#pragma hdrstop

#include <GradientBoostFullTreeBuilder.h>
#include <LinkedRegressionTree.h>
#include <GradientBoostFullProblem.h>
#include <NeoMathEngine/OpenMP.h>

namespace NeoML {

// The statistics for one thread
template<class T>
struct CThreadStatistics {
	// Current data
	// The value of the feature and the current split threshold are not stored here but iterated through locally
	// The statistics for nodes are stored because we have to go through the whole tree level when looking for the optimal split threshold
	// 
	// The statistics calculated on the left and on the right of the current split threshold
	T CurLeftStatistics;
	T CurRightStatistics;
	// The previous threshold value
	float Prev;

	// The best result
	int FeatureIndex;
	float Threshold;
	float Criterion;
	T LeftStatistics;
	T RightStatistics;

	// The total statistics
	const T& TotalStatistics;

	CThreadStatistics( const CThreadStatistics& other );
	explicit CThreadStatistics( float criterion, const T& totalStatistics );
};

template<class T>
inline CThreadStatistics<T>::CThreadStatistics( const CThreadStatistics& other ):
	CurLeftStatistics( other.CurLeftStatistics ),
	CurRightStatistics( other.CurRightStatistics ),
	Prev( other.Prev ),
	FeatureIndex( other.FeatureIndex ),
	Threshold( other.Threshold ),
	Criterion( other.Criterion ),
	LeftStatistics( other.LeftStatistics ),
	RightStatistics( other.RightStatistics ),
	TotalStatistics( other.TotalStatistics )
{
}

template<class T>
inline CThreadStatistics<T>::CThreadStatistics( float criterion, const T& totalStatistics ) :
	CurLeftStatistics( totalStatistics.ValueSize() ),
	CurRightStatistics( totalStatistics.ValueSize() ),
	Prev( 0.0 ),
	FeatureIndex( NotFound ),
	Threshold( 0.0 ),
	Criterion( criterion ),
	TotalStatistics( totalStatistics )
{
}

// The node statistics
template<class T>
struct CGradientBoostNodeStatistics : public virtual IObject {
	// The level of the node
	const int Level;
	// The total statistics for all vectors in the node
	const T TotalStatistics;
	// The current statistics used by the threads when looking for the best split of this node
	// Each thread works with a subset of features which it searches for the best feature and split threshold
	// After the threads have finished their search the best overall result is selected 
	// and written directly into the node statistics (the FeatureIndex and Threshold fields)
	CArray<CThreadStatistics<T>> ThreadStatistics;

	// The feature used for split (specified by the index in the subproblem)
	// If NotFound (-1) the node is a leaf
	int FeatureIndex;
	// The split threshold
	float Threshold;
	// The child nodes
	CPtr<CGradientBoostNodeStatistics<T>> Left;
	CPtr<CGradientBoostNodeStatistics<T>> Right;
	T LeftStatistics;
	T RightStatistics;

	explicit CGradientBoostNodeStatistics( int level, const T& totalStatistics );

	void InitThreadStatistics( int threadCount, float l1RegFactor, float l2RegFactor );
};

template<class T>
inline CGradientBoostNodeStatistics<T>::CGradientBoostNodeStatistics( int level, const T& totalStatistics ) :
	Level( level ),
	TotalStatistics( totalStatistics ),
	FeatureIndex( NotFound ),
	Threshold( 0.0 ),
	LeftStatistics( totalStatistics.ValueSize() ),
	RightStatistics( totalStatistics.ValueSize() )
{
}

template<class T>
inline void CGradientBoostNodeStatistics<T>::InitThreadStatistics( int threadCount, float l1RegFactor, float l2RegFactor )
{
	const float criterion = static_cast<float>( TotalStatistics.CalcCriterion( l1RegFactor, l2RegFactor ) );
	ThreadStatistics.Add( CThreadStatistics<T>( criterion, TotalStatistics ), threadCount );
}

//------------------------------------------------------------------------------------------------------------

template<class T>
CGradientBoostFullTreeBuilder<T>::CGradientBoostFullTreeBuilder( const CGradientBoostFullTreeBuilderParams& _params, CTextStream* _logStream ) :
	params( _params ),
	logStream( _logStream ),
	nodesCount( 0 )
{
	NeoAssert( params.MaxTreeDepth > 0 );
	NeoAssert( params.MaxNodesCount > 0 || params.MaxNodesCount == NotFound );
	NeoAssert( abs( params.MinSubsetHessian ) > 0 );
	NeoAssert( params.ThreadCount > 0 );
	NeoAssert( params.MinSubsetWeight >= 0 );
	NeoAssert( params.DenseTreeBoostCoefficient >= 0 );
}

template<class T>
CPtr<CRegressionTree> CGradientBoostFullTreeBuilder<T>::Build( const CGradientBoostFullProblem& problem,
	const CArray<typename T::Type>& gradients, const typename T::Type& gradientsSum,
	const CArray<typename T::Type>& hessians, const typename T::Type& hessiansSum,
	const CArray<double>& weights, double weightsSum )
{
	if( logStream != 0 ) {
		*logStream << L"\nGradient boost float problem tree building started:\n";
	}

	// Creating the statistics tree root
	CPtr<CGradientBoostNodeStatistics<T>> root = initialize( problem, gradientsSum, hessiansSum, weightsSum );

	// Building the tree level by level
	for( int i = 0; i < params.MaxTreeDepth; i++ ) {
		if( !buildTreeLevel( problem, i, gradients, hessians, weights ) ) {
			break;
		}
	}

	// Pruning
	if( params.PruneCriterionValue != 0 ) {
		prune( *root );
	}

	if( logStream != 0 ) {
		*logStream << L"\nGradient boost float problem tree building finished:\n";
	}

	// Building the final tree from the statistics tree
	return buildModel( problem.GetUsedFeatureIndexes(), *root ).Ptr();
}

// Initializes the builder
template<class T>
CPtr<CGradientBoostNodeStatistics<T>> CGradientBoostFullTreeBuilder<T>::initialize( const CGradientBoostFullProblem& problem,
	const typename T::Type& gradientSum, const typename T::Type& hessianSum, double weightSum )
{
	// Creating the root and filling in its statistics
	T totalStatistics( gradientSum, hessianSum, weightSum );
	CPtr<CGradientBoostNodeStatistics<T>> root = FINE_DEBUG_NEW CGradientBoostNodeStatistics<T>( 0, totalStatistics );
	root->InitThreadStatistics( params.ThreadCount, params.L1RegFactor, params.L2RegFactor );
	curLevelStatistics.DeleteAll();
	curLevelStatistics.Add( root );
	nodesCount = 1;

	// To start with, all vectors are put into root
	classifyNodesCache.Empty();
	classifyNodesCache.Add( root, problem.GetUsedVectorCount() );

	vectorNodes.Empty();
	vectorNodes.Add( 0, problem.GetUsedVectorCount() );
	return root;
}

// Builds one level of the tree
template<class T>
bool CGradientBoostFullTreeBuilder<T>::buildTreeLevel( const CGradientBoostFullProblem& problem, int level,
	const CArray<typename T::Type>& gradients, const CArray<typename T::Type>& hessians, const CArray<double>& weights )
{
	if( logStream != 0 ) {
		*logStream << L"\nBuild level " << level << L":\n";
	}

	// Distribute the vectors over the current level nodes
	if( level > 0 ) { // we have already performed the calculations for the root
		distributeVectorsByNodes( problem, level );
	}

	// Try splitting the nodes on the current level looking for the optimal splitting feature for each node
	findSplits( problem, gradients, hessians, weights );

	// Select the best overall result from the threads' results
	mergeThreadResults();

	// Now we have processed all vectors, perform the final split
	return split();
}

// Distributes the vectors over the current level nodes
template<class T>
void CGradientBoostFullTreeBuilder<T>::distributeVectorsByNodes( const CGradientBoostFullProblem& problem, int level )
{
	NeoPresume( level > 0 );

	// Get the features to be used for splitting on this level
	splitFeatures.DeleteAll();
	for( int i = 0; i < curLevelStatistics.Size(); i++ ) {
		if( curLevelStatistics[i]->FeatureIndex != NotFound ) {
			splitFeatures.Add( curLevelStatistics[i]->FeatureIndex );
		}
	}
	// Sort and remove duplicates
	splitFeatures.QuickSort< Ascending<int> >();
	int newSize = 1;
	for( int i = 1; i < splitFeatures.Size(); i++ ) {
		if( splitFeatures[newSize - 1] != splitFeatures[i] ) {
			splitFeatures[newSize] = splitFeatures[i];
			newSize++;
		}
	}
	splitFeatures.SetSize( newSize );

	// Distribute the vectors
	NEOML_OMP_NUM_THREADS(params.ThreadCount)
	{
		const int threadNumber = OmpGetThreadNum();
		NeoAssert( threadNumber < params.ThreadCount );
		int i = threadNumber;
		while( i < splitFeatures.Size() ) {
			const int index = splitFeatures[i];
			if( problem.IsUsedFeatureBinary( index ) ) {
				const int* ptr = reinterpret_cast<const int*>( problem.GetUsedFeatureDataPtr( index ) );
				int len = problem.GetUsedFeatureDataSize( index );
				for( int j = 0; j < len; j++ ) {
					CGradientBoostNodeStatistics<T>* statistics = classifyNodesCache[ptr[j]];
					if( statistics == 0 ) {
						continue;
					}
					if( statistics->FeatureIndex == index ) {
						vectorNodes[ptr[j]] = level; // always to the right subtree because value == 1.
					}
				}
			} else {
				const CFloatVectorElement* ptr = reinterpret_cast<const CFloatVectorElement*>( problem.GetUsedFeatureDataPtr( index ) );
				int len = problem.GetUsedFeatureDataSize( index );
				for( int j = 0; j < len; j++ ) {
					if( ptr[j].Index == NotFound ) {
						continue;
					}
					CGradientBoostNodeStatistics<T>* statistics = classifyNodesCache[ptr[j].Index];
					if( statistics == 0 ) {
						continue;
					}
					if( statistics->FeatureIndex == index ) {
						vectorNodes[ptr[j].Index] = ptr[j].Value <= statistics->Threshold ? -level : level;
					}
				}
			}

			i += OmpGetThreadCount();
		}
	}

	// Distribute the vectors to the selected subtrees
	curLevelStatistics.DeleteAll();
	for( int i = 0; i < classifyNodesCache.Size(); i++ ) {
		CGradientBoostNodeStatistics<T>* statistics = classifyNodesCache[i];
		if( statistics == 0 ) {
			continue;
		}

		if( statistics->FeatureIndex != NotFound ) {
			if( abs( vectorNodes[i] ) != level ) {
				classifyNodesCache[i] = ( 0 <= statistics->Threshold ) ? statistics->Left : statistics->Right;
			} else {
				classifyNodesCache[i] = ( vectorNodes[i] < 0 ) ? statistics->Left : statistics->Right;
			}
			statistics = classifyNodesCache[i];
		} else if( statistics->Level < level ) {
			classifyNodesCache[i] = 0; // the vector has been put into a leaf on a higher level already
		}

		if( statistics->Level == level && statistics->ThreadStatistics.IsEmpty() ) {
			statistics->InitThreadStatistics( params.ThreadCount, params.L1RegFactor, params.L2RegFactor );
			curLevelStatistics.Add( statistics );
		}
	}
}

// Finds the best split for each node using the features in the specified range
template<class T>
void CGradientBoostFullTreeBuilder<T>::findSplits( const CGradientBoostFullProblem& problem,
	const CArray<typename T::Type>& gradients, const CArray<typename T::Type>& hessians, const CArray<double>& weights )
{
	NEOML_OMP_NUM_THREADS(params.ThreadCount)
	{
		const int threadNumber = OmpGetThreadNum();
		NeoAssert( threadNumber < params.ThreadCount );
		int i = threadNumber;
		while( i < problem.GetUsedFeatureCount() ) {
			if( problem.IsUsedFeatureBinary( i ) ) {
				findBinarySplits( threadNumber, gradients, hessians, weights, i,
					reinterpret_cast<const int*>( problem.GetUsedFeatureDataPtr( i ) ), problem.GetUsedFeatureDataSize( i ) );
			} else {
				findSplits( threadNumber, gradients, hessians, weights, i,
					 reinterpret_cast<const CFloatVectorElement*>( problem.GetUsedFeatureDataPtr( i ) ), problem.GetUsedFeatureDataSize( i ) );
			}
			i += OmpGetThreadCount();
		}
	}
}

// Finds the best split for each node using the specified binary feature
template<class T>
void CGradientBoostFullTreeBuilder<T>::findBinarySplits( int threadNumber,
	const CArray<typename T::Type>& gradients, const CArray<typename T::Type>& hessians,
	const CArray<double>& weights, int feature, const int* ptr, int size )
{
	if( size == 0 ) {
		// The feature has no values
		return;
	}

	// Accumulate the node statistics
	for( int i = 0; i < size; i++ ) {
		const int vectorIndex = ptr[i];
		if( classifyNodesCache[vectorIndex] == 0 ) {
			continue;
		}
		CThreadStatistics<T>& statistics = classifyNodesCache[vectorIndex]->ThreadStatistics[threadNumber];
		if( statistics.Prev == 0 ) {
			// We have come across this node for the first time
			statistics.CurRightStatistics.Erase();
			statistics.Prev = 1.0;
		}

		statistics.CurRightStatistics.Add( gradients, hessians, weights, vectorIndex );
	}

	// Try splitting using the accumulated statistics
	for( int j = 0; j < curLevelStatistics.Size(); j++ ) {
		CThreadStatistics<T>& curStatistics = curLevelStatistics[j]->ThreadStatistics[threadNumber];
		if( curStatistics.Prev == 0 ) {
			continue;
		}
		curStatistics.Prev = 0;
		
		curStatistics.CurLeftStatistics = curLevelStatistics[j]->TotalStatistics;
		curStatistics.CurLeftStatistics.Sub( curStatistics.CurRightStatistics );

		checkSplit( feature, 0.0, 1.0, curStatistics );
	}
}

// Find the best split for each node using the specified feature
template<class T>
void CGradientBoostFullTreeBuilder<T>::findSplits( int threadNumber,
	const CArray<typename T::Type>& gradients, const CArray<typename T::Type>& hessians,
	const CArray<double>& weights, int feature, const CFloatVectorElement* ptr, int size )
{
	if( size == 0 ) {
		// The feature has no values
		return;
	}

	// Iterate from the smallest to the largest and then back, so we don't need to calculate statistics for 0

	// First process all negative values
	for( int i = 0; i < size; i++ ) {
		const int vectorIndex = ptr[i].Index;
		if( vectorIndex == NotFound ) { // zero value
			NeoPresume( ptr[i].Value == 0 );
			if( i == 0 ) {
				break; // cannot split: nothing in the left subtree
			}

			for( int j = 0; j < curLevelStatistics.Size(); j++ ) {
				CThreadStatistics<T>& curStatistics = curLevelStatistics[j]->ThreadStatistics[threadNumber];
				curStatistics.CurRightStatistics = curLevelStatistics[j]->TotalStatistics;
				curStatistics.CurRightStatistics.Sub( curStatistics.CurLeftStatistics );
				if( curStatistics.Prev != 0 ) { // this node only has positive values
					checkSplit( feature, curStatistics.Prev, 0, curStatistics );
					curStatistics.Prev = 0;
				}
			}
			break;
		}
		if( classifyNodesCache[vectorIndex] == 0 ) {
			continue;
		}
		CThreadStatistics<T>& statistics = classifyNodesCache[vectorIndex]->ThreadStatistics[threadNumber];
		if( statistics.Prev == 0 ) {
			// We have come across this node for the first time
			statistics.CurLeftStatistics.Erase();
			statistics.Prev = ptr[i].Value;
		}

		if( statistics.Prev != ptr[i].Value ) { // equal values should be in the same subtree
			statistics.CurRightStatistics = classifyNodesCache[vectorIndex]->TotalStatistics;
			statistics.CurRightStatistics.Sub( statistics.CurLeftStatistics );
			checkSplit( feature, statistics.Prev, ptr[i].Value, statistics );
			statistics.Prev = ptr[i].Value;
		}

		statistics.CurLeftStatistics.Add( gradients, hessians, weights, vectorIndex );
	}

	// Now process the positive values
	for( int i = size - 1; i >= 0; i-- ) {
		const int vectorIndex = ptr[i].Index;
		if( vectorIndex == NotFound ) { // zero value
			NeoPresume( ptr[i].Value == 0 );
			if( i == size - 1 ) {
				break; // cannot split: nothing in the right subtree
			}

			for( int j = 0; j < curLevelStatistics.Size(); j++ ) {
				CThreadStatistics<T>& curStatistics = curLevelStatistics[j]->ThreadStatistics[threadNumber];
				if( curStatistics.Prev != 0 ) { // this node only has negative values
					curStatistics.CurLeftStatistics = curLevelStatistics[j]->TotalStatistics;
					curStatistics.CurLeftStatistics.Sub( curStatistics.CurRightStatistics );
					checkSplit( feature, 0.0, curStatistics.Prev, curStatistics );
					curStatistics.Prev = 0.0;
				}
			}
			break;
		}

		if( classifyNodesCache[vectorIndex] == 0 ) {
			continue;
		}
		CThreadStatistics<T>& statistics = classifyNodesCache[vectorIndex]->ThreadStatistics[threadNumber];

		if( statistics.Prev == 0 ) {
			// We have come across this node for the first time
			statistics.CurRightStatistics.Erase();
			statistics.Prev = ptr[i].Value;
		}

		if( statistics.Prev != ptr[i].Value ) { // equal values should be in the same subtree
			statistics.CurLeftStatistics = classifyNodesCache[vectorIndex]->TotalStatistics;
			statistics.CurLeftStatistics.Sub( statistics.CurRightStatistics );
			checkSplit( feature, ptr[i].Value, statistics.Prev, statistics );
			statistics.Prev = ptr[i].Value;
		}

		statistics.CurRightStatistics.Add( gradients, hessians, weights, vectorIndex );
	}
}

// Checks if splitting using the specified values is possible
template<class T>
void CGradientBoostFullTreeBuilder<T>::checkSplit( int feature, float firstValue, float secondValue,
	CThreadStatistics<T>& statistics ) const
{
	T leftStatistics( statistics.CurLeftStatistics );
	T rightStatistics( statistics.CurRightStatistics );
	
	double criterion;
	if( !T::CalcCriterion( criterion, leftStatistics, rightStatistics, statistics.TotalStatistics,
		params.L1RegFactor, params.L2RegFactor, params.MinSubsetHessian, params.MinSubsetWeight, params.DenseTreeBoostCoefficient ) )
	{
		return;
	}

	if( statistics.Criterion < static_cast<float>( criterion ) 
		|| ( statistics.Criterion == static_cast<float>( criterion ) && statistics.FeatureIndex > feature ) )
	{
		statistics.FeatureIndex = feature;
		statistics.Criterion = static_cast<float>( criterion );
		if( fabs( firstValue - secondValue ) > 1e-10 ) {
			statistics.Threshold = ( firstValue + secondValue ) / 2;
		} else {
			statistics.Threshold = firstValue;
		}
		statistics.LeftStatistics = leftStatistics;
		statistics.RightStatistics = rightStatistics;
	}
}

// Merge the data from different threads
template<class T>
void CGradientBoostFullTreeBuilder<T>::mergeThreadResults()
{
	for( int i = 0; i < curLevelStatistics.Size(); i++ ) {
		float criterion = static_cast<float>( curLevelStatistics[i]->TotalStatistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor ) );
		for( int j = 0; j < params.ThreadCount; j++ ) {
			const CThreadStatistics<T>& currThreadStatistics = curLevelStatistics[i]->ThreadStatistics[j];
			// The check for equivalence has been added to have a determinate result
			if( currThreadStatistics.FeatureIndex != NotFound ) {
				if( currThreadStatistics.Criterion > criterion
					|| ( currThreadStatistics.Criterion == criterion
						&& currThreadStatistics.FeatureIndex < curLevelStatistics[i]->FeatureIndex ) )
				{
					criterion = currThreadStatistics.Criterion;
					curLevelStatistics[i]->FeatureIndex = currThreadStatistics.FeatureIndex;
					curLevelStatistics[i]->Threshold = currThreadStatistics.Threshold;
					curLevelStatistics[i]->LeftStatistics = currThreadStatistics.LeftStatistics;
					curLevelStatistics[i]->RightStatistics = currThreadStatistics.RightStatistics;
				}
			}
		}
	}
}

// Splits the nodes on the specified level
// Will return true if new nodes were added after splitting
template<class T>
bool CGradientBoostFullTreeBuilder<T>::split()
{
	bool result = false;
	// Perform the split
	for( int i = 0; i < curLevelStatistics.Size(); i++ ) {
		CGradientBoostNodeStatistics<T>* statistics = curLevelStatistics[i];
		if( statistics->FeatureIndex != NotFound
			&& ( nodesCount + 2 <= params.MaxNodesCount || params.MaxNodesCount == NotFound ) )
		{
			// The new node is not a leaf
			if( logStream != 0 ) {
				*logStream << L"Split result: index = " << statistics->FeatureIndex
					<< L" threshold = " << statistics->Threshold
					<< L", criterion = " << statistics->LeftStatistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor )
						 + statistics->RightStatistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor )
					<< L" )\n";
			}

			statistics->Left = FINE_DEBUG_NEW CGradientBoostNodeStatistics<T>( statistics->Level + 1, statistics->LeftStatistics );
			statistics->Right = FINE_DEBUG_NEW CGradientBoostNodeStatistics<T>( statistics->Level + 1, statistics->RightStatistics );
			nodesCount += 2;
			result = true;
		} else {
			// The new node is a leaf
			if( logStream != 0 ) {
				*logStream << L"Split result: created const node.\t\t"
					<< L", criterion = " << statistics->TotalStatistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor )
					<< L" )\n";
			}
			statistics->FeatureIndex = NotFound;
		}
	}
	return result;
}

// Merges some of the nodes and prunes the tree
template<class T>
bool CGradientBoostFullTreeBuilder<T>::prune( CGradientBoostNodeStatistics<T>& node ) const
{
	if( node.Left == 0 ) {
		NeoAssert( node.Right == 0 );
		// No children
		return true;
	}
	NeoAssert( node.Right != 0 );

	if( !prune( *node.Left ) || !prune( *node.Right ) ) {
		return false;
	}

	const double oneNodeCriterion = node.TotalStatistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor );
	const double splitCriterion = node.Left->TotalStatistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor ) +
		node.Right->TotalStatistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor );

	if( splitCriterion - oneNodeCriterion < params.PruneCriterionValue ) {
		node.Left.Release();
		node.Right.Release();
		node.FeatureIndex = NotFound;
		return true;
	}
	return false;
}

// Builds the final model
template<class T>
CPtr<CLinkedRegressionTree> CGradientBoostFullTreeBuilder<T>::buildModel( const CArray<int>& usedFeatures,
	CGradientBoostNodeStatistics<T>& node ) const
{
	CPtr<CLinkedRegressionTree> result = FINE_DEBUG_NEW CLinkedRegressionTree();

	if( node.FeatureIndex == NotFound ) {
		typename T::Type values;
		node.TotalStatistics.LeafValue( values );
		result->InitLeafNode( values );
	} else {
		CPtr<CLinkedRegressionTree> left = buildModel( usedFeatures, *node.Left );
		CPtr<CLinkedRegressionTree> right = buildModel( usedFeatures, *node.Right );
		result->InitSplitNode( *left, *right, usedFeatures[node.FeatureIndex], node.Threshold );
	}

	return result;
}

template class CGradientBoostFullTreeBuilder<CGradientBoostStatisticsSingle>;
template class CGradientBoostFullTreeBuilder<CGradientBoostStatisticsMulti>;

} // namespace NeoML
