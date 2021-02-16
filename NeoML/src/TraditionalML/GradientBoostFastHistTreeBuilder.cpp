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

#include <GradientBoostFastHistTreeBuilder.h>
#include <LinkedRegressionTree.h>
#include <NeoMathEngine/OpenMP.h>

namespace NeoML {

CGradientBoostFastHistTreeBuilder::CGradientBoostFastHistTreeBuilder( const CParams& _params, CTextStream* _logStream ) :
	params( _params ),
	logStream( _logStream ),
	histSize( NotFound )
{
	NeoAssert( params.MaxTreeDepth > 0 );
	NeoAssert( params.MaxNodesCount > 0 || params.MaxNodesCount == NotFound );
	NeoAssert( abs( params.MinSubsetHessian ) > 0 );
	NeoAssert( params.ThreadCount > 0 );
	NeoAssert( params.MaxBins > 1 );
	NeoAssert( params.MinSubsetWeight >= 0 );
}

CPtr<CRegressionTree> CGradientBoostFastHistTreeBuilder::Build( const CGradientBoostFastHistProblem& problem,
	const CArray<double>& gradients, const CArray<double>& hessians, const CArray<float>& weights )
{
	NeoAssert( gradients.Size() == hessians.Size() );

	if( logStream != 0 ) {
		*logStream << L"\nGradient boost float problem tree building started:\n";
	}

	// Initialization
	initVectorSet( problem.GetUsedVectorCount() );
	initHistData( problem );

	// Creating the tree root
	CNode root( 0, 0, vectorSet.Size() );
	root.HistPtr = allocHist();
	buildHist( problem, root, gradients, hessians, weights, root.Statistics );
	nodes.Empty();
	nodes.Add( root );

	// Building the tree using depth-first search, which needs less memory for histograms
	nodeStack.Empty();
	nodeStack.Add( 0 );

	const CArray<int>& featureIndexes = problem.GetFeatureIndexes();
	const CArray<float>& cuts = problem.GetFeatureCuts();

	// Building starts from the root
	while( !nodeStack.IsEmpty() ) {
		const int node = nodeStack.Last();
		nodeStack.DeleteLast();

		// Calculating the best identifier for the split
		nodes[node].SplitFeatureId = evaluateSplit( problem, nodes[node] );
		if( nodes[node].SplitFeatureId != NotFound ) {
			// The split is possible
			if( logStream != 0 ) {
				*logStream << L"Split result: index = " << featureIndexes[nodes[node].SplitFeatureId]
					<< L" threshold = " << cuts[nodes[node].SplitFeatureId]
					<< L" ( gradient = " << nodes[node].Statistics.TotalGradient()
					<< L", hessian = " << nodes[node].Statistics.TotalHessian()
					<< L", criterion = " << nodes[node].Statistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor )
					<< L" )\n";
			}

			// Splitting
			int leftNode = NotFound;
			int rightNode = NotFound;
			applySplit( problem, node, leftNode, rightNode );
			nodes[node].Left = leftNode;
			nodeStack.Add( leftNode );
			nodes[node].Right = rightNode;
			nodeStack.Add( rightNode );
			// Building the smaller histogram and generating the other one by substraction
			if( nodes[leftNode].VectorSetSize < nodes[rightNode].VectorSetSize ) {
				nodes[leftNode].HistPtr = allocHist();
				buildHist( problem, nodes[leftNode], gradients, hessians, weights, nodes[leftNode].Statistics );
				subHist( nodes[node].HistPtr, nodes[leftNode].HistPtr );
				nodes[rightNode].HistPtr = nodes[node].HistPtr;
				nodes[rightNode].Statistics = nodes[node].Statistics;
				nodes[rightNode].Statistics.Sub( nodes[leftNode].Statistics );
			} else {
				nodes[rightNode].HistPtr = allocHist();
				buildHist( problem, nodes[rightNode], gradients, hessians, weights, nodes[rightNode].Statistics );
				subHist( nodes[node].HistPtr, nodes[rightNode].HistPtr );
				nodes[leftNode].HistPtr = nodes[node].HistPtr;
				nodes[leftNode].Statistics = nodes[node].Statistics;
				nodes[leftNode].Statistics.Sub( nodes[rightNode].Statistics );
			}
		} else {
			// The node could not be split
			if( logStream != 0 ) {
				*logStream << L"Split result: created const node.\t\t"
					<< L" ( gradient = " << nodes[node].Statistics.TotalGradient()
					<< L", hessian = " << nodes[node].Statistics.TotalHessian()
					<< L", criterion = " << nodes[node].Statistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor )
					<< L" )\n";
			}
			freeHist( nodes[node].HistPtr );
			nodes[node].HistPtr = NotFound;
		}
		
	}

	if( logStream != 0 ) {
		*logStream << L"\nGradient boost float problem tree building finished:\n";
	}

	// Pruning
	if( params.PruneCriterionValue != 0 ) {
		prune( 0 );
	}

	return buildTree( 0, featureIndexes, cuts ).Ptr();
}

// Initializes the array of node vector sets
void CGradientBoostFastHistTreeBuilder::initVectorSet( int size )
{
	// For a start, all vectors are assigned to the root node
	vectorSet.SetSize( size );
	for( int i = 0; i < size; i++ ) {
		vectorSet[i] = i;
	}
}

// Initializes the array storing the histograms
void CGradientBoostFastHistTreeBuilder::initHistData( const CGradientBoostFastHistProblem& problem )
{
	// Only the features that are used will be present in the histograms
	const CArray<int>& usedFeatures = problem.GetUsedFeatures();
	const CArray<int>& featurePos = problem.GetFeaturePos();

	idPos.Empty();
	idPos.Add( NotFound, featurePos.Last() );
	histIds.Empty();
	for( int i = 0; i < usedFeatures.Size(); i++ ) {
		const int featureIndex = usedFeatures[i];
		for( int j = featurePos[featureIndex]; j < featurePos[featureIndex + 1]; j++ ) {
			idPos[j] = histIds.Size();
			histIds.Add( j );
		}
	}

	histSize = histIds.Size();

	// The histogram size of tree depth + 1 is sufficient
	histStats.SetSize( histSize * ( params.MaxTreeDepth + 1 ) );
	freeHists.Empty();
	for( int i = 0; i <= params.MaxTreeDepth; i++ ) {
		freeHists.Add( i * histSize ); // a histogram is identified by the pointer to its start in the histData array
	}
}

// Gets a free histogram 
// A histogram is identified by the pointer to its start in the histData array
int CGradientBoostFastHistTreeBuilder::allocHist()
{
	NeoAssert( !freeHists.IsEmpty() );

	int result = freeHists.Last();
	freeHists.DeleteLast();
	return result;
}

// Free the unnecessary histogram
// A histogram is identified by the pointer to its start in the histData array
void CGradientBoostFastHistTreeBuilder::freeHist( int ptr )
{
	freeHists.Add( ptr );
}

// Subtract histograms
void CGradientBoostFastHistTreeBuilder::subHist( int firstPtr, int secondPtr )
{
	for( int i = 0; i < histSize; i++ ) {
		histStats[firstPtr + i].Sub( histStats[secondPtr + i] );
	}
}

// Build a histogram on the vectors of the given node
void CGradientBoostFastHistTreeBuilder::buildHist( const CGradientBoostFastHistProblem& problem, const CNode& node,
	const CArray<double>& gradients, const CArray<double>& hessians, const CArray<float>& weights,
	CGradientBoostStatisticsSingle& totalStats )
{
	CGradientBoostStatisticsSingle* histStatsPtr = histStats.GetPtr() + node.HistPtr;
	::memset( reinterpret_cast<char*>( histStatsPtr ), 0, histSize * sizeof( CGradientBoostStatisticsSingle ) );

	totalStats.Erase();

	const bool isOmp = ( node.VectorSetSize > 4 * params.ThreadCount ); // check if using OpenMP makes sense
	if( isOmp ) {
		// There are many vectors in the set, so we'll use several threads to build the histogram
		CArray<CGradientBoostStatisticsSingle> results;
		results.SetSize( params.ThreadCount );

		tempHistStats.SetSize( params.ThreadCount * histSize );
		::memset( reinterpret_cast<char*>( tempHistStats.GetPtr() ), 0, tempHistStats.Size() * sizeof( CGradientBoostStatisticsSingle ) );

		NEOML_OMP_NUM_THREADS(params.ThreadCount)
		{
			const int threadNumber = OmpGetThreadNum();
			NeoAssert( threadNumber < params.ThreadCount );
			int i = threadNumber;
			while( i < node.VectorSetSize ) {
				const int vectorIndex = vectorSet[node.VectorSetPtr + i];
				addVectorToHist( problem.GetUsedVectorDataPtr( vectorIndex ), problem.GetUsedVectorDataSize( vectorIndex ),
					gradients[vectorIndex], hessians[vectorIndex], weights[vectorIndex], tempHistStats.GetPtr() + histSize * threadNumber );
				results[threadNumber].Add( gradients[vectorIndex], hessians[vectorIndex], weights[vectorIndex] );
				i += params.ThreadCount;
			}
		}

		// Merge the threads' results
		for( int i = 0; i < params.ThreadCount; i++ ) {
			totalStats.Add( results[i] );
		}

		NEOML_OMP_FOR_NUM_THREADS( params.ThreadCount )
		for( int i = 0; i < histSize; i++ ) {
			for( int j = 0; j < params.ThreadCount; j++ ) {
				const int index = j * histSize + i;
				histStatsPtr[i].Add( tempHistStats[index] );
			}
		}
	} else {
		// There are few vectors in the set, build the histogram using only one thread
		for( int i = 0; i < node.VectorSetSize; i++ ) {
			const int vectorIndex = vectorSet[node.VectorSetPtr + i];
			addVectorToHist( problem.GetUsedVectorDataPtr( vectorIndex ), problem.GetUsedVectorDataSize( vectorIndex ),
				gradients[vectorIndex], hessians[vectorIndex], weights[vectorIndex], histStatsPtr );
			totalStats.Add( gradients[vectorIndex], hessians[vectorIndex], weights[vectorIndex] );
		}
	}

	// Adding zero values
	const CArray<int>& usedFeatures = problem.GetUsedFeatures();
	const CArray<int>& featurePos = problem.GetFeaturePos();
	const CArray<int>& featureNullValueId = problem.GetFeatureNullValueId();

	NEOML_OMP_FOR_NUM_THREADS( params.ThreadCount )
	for( int i = 0; i < usedFeatures.Size(); i++ ) {
		const int nullFeatureId = featureNullValueId[usedFeatures[i]];

		CGradientBoostStatisticsSingle nullStatistics( totalStats );
		for( int j = featurePos[usedFeatures[i]]; j < featurePos[usedFeatures[i] + 1]; j++ ) {
			nullStatistics.Sub( histStatsPtr[idPos[j]] );
		}
		histStatsPtr[idPos[nullFeatureId]].Add( nullStatistics );
	}
}

// Adds a vector to the histogram
void CGradientBoostFastHistTreeBuilder::addVectorToHist( const int* vectorPtr, int vectorSize, double gradients, double hessian, float weight,
	CGradientBoostStatisticsSingle* stats )
{
	NeoPresume( vectorPtr != 0 );
	NeoPresume( vectorSize >= 0 );

	for( int i = 0; i < vectorSize; i++ ) {
		const int id = idPos[vectorPtr[i]];
		if( id != NotFound ) {
			stats[id].Add( gradients, hessian, weight );
		}
	}
}

// Calculates the optimal feature value for splitting the node
// Returns NotFound if splitting is impossible
int CGradientBoostFastHistTreeBuilder::evaluateSplit( const CGradientBoostFastHistProblem& problem, const CNode& node ) const
{
	if( node.Level >= params.MaxTreeDepth ) {
		// The level limit has been reached
		return NotFound;
	}

	const CArray<int>& usedFeatures = problem.GetUsedFeatures();
	const CArray<int>& featurePos = problem.GetFeaturePos();
	double bestValue = node.Statistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor );
	const CGradientBoostStatisticsSingle* histStatsPtr = histStats.GetPtr() + node.HistPtr;

	// Initializing the search results for each thread
	// The default bestValue is the parent's Gain (the node is not split by default)
	CArray<double>& splitGainsByThread = splitGainsByThreadBuffer;
	splitGainsByThread.DeleteAll();
	splitGainsByThread.Add( bestValue, params.ThreadCount );
	CArray<int>& splitIds = splitIdsBuffer;
	splitIds.DeleteAll();
	splitIds.Add( NotFound, params.ThreadCount );

	NEOML_OMP_NUM_THREADS(params.ThreadCount)
	{
		const int threadNumber = OmpGetThreadNum();
		NeoAssert( threadNumber < params.ThreadCount );
		
		// Iterate through features (a separate subset for each thread)
		for( int i = threadNumber; i < usedFeatures.Size(); i += params.ThreadCount ) {
			CGradientBoostStatisticsSingle left; // the gain for the left node after the split
			CGradientBoostStatisticsSingle right; // for the right node after the split (calculated as the complement to the parent)
			const int firstFeatureIndex = featurePos[usedFeatures[i]];
			const int lastFetureIndex = featurePos[usedFeatures[i] + 1];
			// Iterate through feature values (sorted ascending) looking for the split position
			for( int j = firstFeatureIndex; j < lastFetureIndex; j++ ) {
				const CGradientBoostStatisticsSingle& featureStats = histStatsPtr[idPos[j]];
				left.Add( featureStats );
				right = node.Statistics;
				right.Sub( left );

				// The condition (lower limit) for the resulting node size
				if( right.TotalHessian() < params.MinSubsetHessian
					|| left.TotalHessian() < params.MinSubsetHessian
					|| left.TotalWeight() < params.MinSubsetWeight
					|| right.TotalWeight() < params.MinSubsetWeight )
				{
					continue;
				}

				// Calculating the gain: if the node is split at this position, 
				// the criterion loses the parent node (bestValue) and replaces it by left.CalcCriterion and right.CalcCriterion
				// In the reference paper, a gamma coefficient is also needed for a new node, but we take that into account while pruning
				const double criterion = left.CalcCriterion( params.L1RegFactor, params.L2RegFactor )
					+ right.CalcCriterion( params.L1RegFactor, params.L2RegFactor );
				if( splitGainsByThread[threadNumber] < criterion ) {
					splitGainsByThread[threadNumber] = criterion;
					splitIds[threadNumber] = j;  // this number refers both to the feature and its value
				}
			}
		}
	}

	// Choose the best result over all threads
	int result = NotFound;
	for( int i = 0; i < splitGainsByThread.Size(); i++ ) {
		const double& threadBestGain = splitGainsByThread[i];
		const int threadBestFeature = splitIds[i]; // the coordinate in the array witha all values of all features
		if( bestValue < threadBestGain || ( bestValue == threadBestGain && threadBestFeature < result ) ) {
			bestValue = threadBestGain;
			result = threadBestFeature;
		}
	}
	return result;
}

// Splits a node
void CGradientBoostFastHistTreeBuilder::applySplit( const CGradientBoostFastHistProblem& problem, int node,
	int& leftNode, int& rightNode )
{
	NeoAssert( node >= 0 );

	const CArray<int>& featureIndexes = problem.GetFeatureIndexes();
	const CArray<int>& featureNullValueId = problem.GetFeatureNullValueId();

	const int vectorPtr = nodes[node].VectorSetPtr;
	const int vectorCount = nodes[node].VectorSetSize;
	const int featureIndex = featureIndexes[nodes[node].SplitFeatureId];
	const int nextId = problem.GetFeaturePos()[featureIndex + 1] - 1;

	// Determining to which subtree each vector belongs
	NEOML_OMP_NUM_THREADS(params.ThreadCount)
	{
		const int threadNumber = OmpGetThreadNum();
		NeoAssert( threadNumber < params.ThreadCount );
		int i = threadNumber;
		while( i < vectorCount ) {
			const int* vectorDataPtr = problem.GetUsedVectorDataPtr( vectorSet[vectorPtr + i] );
			const int vectorDataSize = problem.GetUsedVectorDataSize( vectorSet[vectorPtr + i] );

			const int pos = FindInsertionPoint<int, Ascending<int>, int>( nextId, vectorDataPtr, vectorDataSize );
			int vectorFeatureId = NotFound; // the ID of the feature value used for split for this vector
			if( pos == 0 || ( featureIndexes[vectorDataPtr[pos - 1]] != featureIndex ) ) {
				// The vector contains no feature value for the split, therefore this value is 0
				vectorFeatureId = featureNullValueId[featureIndex];
			} else {
				vectorFeatureId = vectorDataPtr[pos - 1];
			}

			if( vectorFeatureId <= nodes[node].SplitFeatureId ) { // the value is smaller for the smaller ID
				// The vector belongs to the left subtree
				vectorSet[vectorPtr + i] = -( vectorSet[vectorPtr + i] + 1 );
			} // To the right subtree otherwise (no action needed)

			i += params.ThreadCount;
		}
	}

	// Reordering the vectors of the node
	int leftIndex = 0;
	int rightIndex = vectorCount - 1;
	while( leftIndex <= rightIndex ) {
		if( vectorSet[vectorPtr + leftIndex] < 0 ) {
			vectorSet[vectorPtr + leftIndex] = -vectorSet[vectorPtr + leftIndex] - 1;
			leftIndex++;
			continue;
		}
		if( vectorSet[vectorPtr + rightIndex] >= 0 ) {
			rightIndex--;
			continue;
		}
		FObj::swap( vectorSet[vectorPtr + leftIndex], vectorSet[vectorPtr + rightIndex] );
	}

	NeoAssert( leftIndex > 0 );
	NeoAssert( vectorCount - leftIndex > 0 );

	// Creating the child nodes
	CNode left( nodes[node].Level + 1, vectorPtr, leftIndex );
	nodes.Add( left );
	leftNode = nodes.Size() - 1;

	CNode right( nodes[node].Level + 1, vectorPtr + leftIndex, vectorCount - leftIndex );
	nodes.Add( right );
	rightNode = nodes.Size() - 1;
}

// Prunes the tree (merging some nodes)
bool CGradientBoostFastHistTreeBuilder::prune( int node )
{
	if( nodes[node].Left == NotFound ) {
		NeoAssert( nodes[node].Right == NotFound );
		// No child nodes
		return true;
	}
	NeoAssert( nodes[node].Right != NotFound );

	if( !prune( nodes[node].Left ) || !prune( nodes[node].Right ) ) {
		return false;
	}

	const double oneNodeCriterion = nodes[node].Statistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor );
	const double splitCriterion = nodes[nodes[node].Left].Statistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor ) +
		nodes[nodes[node].Right].Statistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor );

	if( splitCriterion - oneNodeCriterion < params.PruneCriterionValue ) {
		nodes[node].Left = NotFound;
		nodes[node].Right = NotFound;
		nodes[node].SplitFeatureId = NotFound;
		return true;
	}
	return false;
}

// Builds the final tree
CPtr<CLinkedRegressionTree> CGradientBoostFastHistTreeBuilder::buildTree( int node, const CArray<int>& featureIndexes,
	const CArray<float>& cuts ) const
{
	CPtr<CLinkedRegressionTree> result = FINE_DEBUG_NEW CLinkedRegressionTree();

	if( nodes[node].SplitFeatureId == NotFound ) {
		result->InitLeafNode( -nodes[node].Statistics.TotalGradient() / nodes[node].Statistics.TotalHessian() );
	} else {
		CPtr<CLinkedRegressionTree> left = buildTree( nodes[node].Left, featureIndexes, cuts );
		CPtr<CLinkedRegressionTree> right = buildTree( nodes[node].Right, featureIndexes, cuts );
		result->InitSplitNode( *left, *right, featureIndexes[nodes[node].SplitFeatureId], cuts[nodes[node].SplitFeatureId] );
	}

	return result;
}

} // namespace NeoML
