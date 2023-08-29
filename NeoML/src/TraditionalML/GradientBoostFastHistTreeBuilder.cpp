/* Copyright © 2017-2023 ABBYY

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
#include <GradientBoostThreadTask.h>
#include <LinkedRegressionTree.h>
#include <NeoMathEngine/ThreadPool.h>

namespace NeoML {

namespace {

// Build the histogram
template<typename T>
class CBuildHistogramThreadTask : public IGradientBoostThreadTask {
public:
	using TNode = typename CGradientBoostFastHistTreeBuilder<T>::CNode;
	using TArray = CArray<typename T::Type>;
	// Create a task
	CBuildHistogramThreadTask( IThreadPool&, const CGradientBoostFastHistProblem&,
		const CArray<int>& vectorSet, const CArray<int>& idPos, T* histStats, const TNode&,
		const TArray& gradients, const TArray& hessians, const CArray<double>& weights,
		CArray<T>& tempHistStats, int histSize, int predictSize, T& totalStats, bool isMultiThread );

	// Run the process in the one main thread
	void RunInOneThread();
	// Combine the answer for the parallel run
	void Reduction();
protected:
	// The size of parallelization, max number of elements to perform
	int ParallelizeSize() const override { return Node.VectorSetSize; }
	// Run the process in a separate thread
	void Run( int threadIndex, int startIndex, int count ) override;

	// Adds a vector to the histogram
	void AddVectorToHist( int vectorIndex, T* histStats );

	const CGradientBoostFastHistProblem& Problem;
	const CArray<int>& VectorSet;
	const CArray<int>& IdPos;
	T* HistStats;
	const TNode& Node;
	const TArray& Gradients;
	const TArray& Hessians;
	const CArray<double>& Weights;
	CArray<T>& TempHistStats;
	const int HistSize;
	T& TotalStats;
	const bool IsMultiThread;

	CArray<T> Results{};
};

template<typename T>
CBuildHistogramThreadTask<T>::CBuildHistogramThreadTask(
		IThreadPool& threadPool,
		const CGradientBoostFastHistProblem& problem,
		const CArray<int>& vectorSet,
		const CArray<int>& idPos,
		T* histStats,
		const TNode& node,
		const TArray& gradients,
		const TArray& hessians,
		const CArray<double>& weights,
		CArray<T>& tempHistStats,
		int histSize,
		int predictionSize,
		T& totalStats,
		bool isMultiThread ) :
	IGradientBoostThreadTask( threadPool ),
	Problem( problem ),
	VectorSet( vectorSet ),
	IdPos( idPos ),
	HistStats( histStats ),
	Node( node ),
	Gradients( gradients ),
	Hessians( hessians ),
	Weights( weights ),
	TempHistStats( tempHistStats ),
	HistSize( histSize ),
	TotalStats( totalStats ),
	IsMultiThread( isMultiThread )
{
	for( int i = 0; i < histSize; ++i ) {
		HistStats[i].Erase();
	}

	totalStats.SetSize( predictionSize );
	totalStats.Erase();

	if( !IsMultiThread )
		return;

	const int threadCount = ThreadPool.Size();
	// There are many vectors in the set, so we'll use several threads to build the histogram
	Results.Add( T( predictionSize ), threadCount );

	const int valueSize = HistStats[0].ValueSize();
	TempHistStats.SetSize( threadCount * HistSize );
	for( int t = 0; t < threadCount * HistSize; ++t ) {
		TempHistStats[t].SetSize( valueSize );
		TempHistStats[t].Erase();
	}
}

template<typename T>
void CBuildHistogramThreadTask<T>::RunInOneThread()
{
	NeoAssert( !IsMultiThread );
	// There are few vectors in the set, build the histogram using only one thread
	for( int index = 0; index < ParallelizeSize(); ++index ) {
		const int vectorIndex = VectorSet[Node.VectorSetPtr + index];
		AddVectorToHist( vectorIndex, HistStats );
		TotalStats.Add( Gradients, Hessians, Weights, vectorIndex );
	}
}

template<typename T>
void CBuildHistogramThreadTask<T>::Run( int threadIndex, int startIndex, int count )
{
	NeoAssert( IsMultiThread );
	T* histStats = &( TempHistStats[HistSize * threadIndex] );
	// Build the histogram using all existing threads
	const int endIndex = startIndex + count;
	for( int index = startIndex; index < endIndex; ++index ) {
		const int vectorIndex = VectorSet[Node.VectorSetPtr + index];
		AddVectorToHist( vectorIndex, histStats );
		Results[threadIndex].Add( Gradients, Hessians, Weights, vectorIndex );
	}
}

template<typename T>
void CBuildHistogramThreadTask<T>::Reduction()
{
	NeoAssert( IsMultiThread );
	// Merge the threads' results
	for( int t = 0; t < ThreadPool.Size(); ++t ) {
		TotalStats.Add( Results[t] );
	}
}
// Adds a vector to the histogram
template<class T>
void CBuildHistogramThreadTask<T>::AddVectorToHist( int vectorIndex, T* histStats )
{
	const int* vectorPtr = Problem.GetUsedVectorDataPtr( vectorIndex );
	const int vectorSize = Problem.GetUsedVectorDataSize( vectorIndex );

	NeoPresume( vectorPtr != 0 );
	NeoPresume( vectorSize >= 0 );

	for( int i = 0; i < vectorSize; ++i ) {
		const int id = IdPos[vectorPtr[i]];
		if( id != NotFound ) {
			histStats[id].Add( Gradients, Hessians, Weights, vectorIndex );
		}
	}
}

//-------------------------------------------------------------------------------------------------------------

//
template<typename T>
class CMergeHistogramsThreadTask : public IGradientBoostThreadTask {
public:
	// Create a task
	CMergeHistogramsThreadTask( IThreadPool& threadPool,
			CArray<T>& tempHistStats, T* histStats, int histSize ) :
		IGradientBoostThreadTask( threadPool ),
		TempHistStats( tempHistStats ),
		HistStats( histStats ),
		HistSize( histSize )
	{}
protected:
	// The size of parallelization, max number of elements to perform
	int ParallelizeSize() const override { return HistSize; }
	// Run the process in a separate thread
	void Run( int threadIndex, int startIndex, int count ) override;

	CArray<T>& TempHistStats;
	T* HistStats;
	const int HistSize;
};

template<typename T>
void CMergeHistogramsThreadTask<T>::Run( int /*threadIndex*/, int startIndex, int count )
{
	const int endIndex = startIndex + count;
	for( int index = startIndex; index < endIndex; ++index ) {
		for( int t = 0; t < ThreadPool.Size(); ++t ) {
			HistStats[index].Add( TempHistStats[t * HistSize + index] );
		}
	}
}

//-------------------------------------------------------------------------------------------------------------

// Adding zero values
template<typename T>
class CAddNullStatsThreadTask : public IGradientBoostThreadTask {
public:
	// Create a task
	CAddNullStatsThreadTask( IThreadPool& threadPool,
			const CGradientBoostFastHistProblem& problem,
			const CArray<int>& idPos, T* histStats, const T& totalStats ) :
		IGradientBoostThreadTask( threadPool ),
		IdPos( idPos ),
		UsedFeatures( problem.GetUsedFeatures() ),
		FeaturePos( problem.GetFeaturePos() ),
		FeatureNullValueId( problem.GetFeatureNullValueId() ),
		HistStats( histStats ),
		TotalStats( totalStats )
	{}
protected:
	// The size of parallelization, max number of elements to perform
	int ParallelizeSize() const override { return UsedFeatures.Size(); }
	// Run the process in a separate thread
	void Run( int threadIndex, int startIndex, int count ) override;

	const CArray<int>& IdPos;
	const CArray<int>& UsedFeatures;
	const CArray<int>& FeaturePos;
	const CArray<int>& FeatureNullValueId;
	T* HistStats;
	const T& TotalStats;
};

template<typename T>
void CAddNullStatsThreadTask<T>::Run( int /*threadIndex*/, int startIndex, int count )
{
	const int endIndex = startIndex + count;
	for( int index = startIndex; index < endIndex; ++index ) {
		const int usedIndex = UsedFeatures[index];
		const int nullFeatureId = FeatureNullValueId[usedIndex];
		T nullStatistics( TotalStats );
		for( int j = FeaturePos[usedIndex]; j < FeaturePos[usedIndex + 1]; ++j ) {
			nullStatistics.Sub( HistStats[IdPos[j]] );
		}
		HistStats[IdPos[nullFeatureId]].Add( nullStatistics );
	}
}

//-------------------------------------------------------------------------------------------------------------

// Calculating the gain
template<typename T>
class CCalcSplitGainThreadTask : public IGradientBoostThreadTask {
public:
	using TNode = typename CGradientBoostFastHistTreeBuilder<T>::CNode;
	using TThBuffers = typename CGradientBoostFastHistTreeBuilder<T>::CThreadsBuffers;
	// Create a task
	CCalcSplitGainThreadTask( IThreadPool&, const CGradientBoostFastHistProblem&,
		const CGradientBoostFastHistTreeBuilderParams&, const CArray<int>& idPos,
		TNode& node, const T* histStats, int predictSize, TThBuffers& tb );

	// Combine the answer for the parallel run
	int Reduction();
protected:
	// The size of parallelization, max number of elements to perform
	int ParallelizeSize() const override { return UsedFeatures.Size(); }
	// Run the process in a separate thread
	void Run( int threadIndex, int startIndex, int count ) override;

	const CGradientBoostFastHistTreeBuilderParams& Params;
	const CArray<int>& IdPos;
	TNode& Node;
	const CArray<int>& UsedFeatures;
	const CArray<int>& FeaturePos;
	const T* HistStats;
	const int PredictionSize;
	// Caching threads temporary memory in the builder
	CArray<int>& SplitIdsByThread;
	CArray<double>& SplitGainsByThread;
	CArray<T>& LeftCandidatesByThread;
	CArray<T>& RightCandidatesByThread;

	double BestValue{};
};

template<typename T>
CCalcSplitGainThreadTask<T>::CCalcSplitGainThreadTask(
		IThreadPool& threadPool,
		const CGradientBoostFastHistProblem& problem,
		const CGradientBoostFastHistTreeBuilderParams& params,
		const CArray<int>& idPos,
		TNode& node,
		const T* histStats,
		int predictSize,
		TThBuffers& tb ) :
	IGradientBoostThreadTask( threadPool ),
	Params( params ),
	IdPos( idPos ),
	Node( node ),
	UsedFeatures( problem.GetUsedFeatures() ),
	FeaturePos( problem.GetFeaturePos() ),
	HistStats( histStats ),
	PredictionSize( predictSize ),
	SplitIdsByThread( tb.SplitIdsBuffer ),
	SplitGainsByThread( tb.SplitGainsBuffer ),
	LeftCandidatesByThread( tb.LeftCandidates ),
	RightCandidatesByThread( tb.RightCandidates ),
	BestValue( Node.Statistics.CalcCriterion( Params.L1RegFactor, Params.L2RegFactor ) )
{
	const int threadCount = ThreadPool.Size();
	// Initializing the search results for each thread
	// The default bestValue is the parent's Gain (the node is not split by default)
	SplitGainsByThread.DeleteAll();
	SplitGainsByThread.Add( BestValue, threadCount );
	SplitIdsByThread.DeleteAll();
	SplitIdsByThread.Add( NotFound, threadCount );

	if( LeftCandidatesByThread.Size() == 0 ) {
		LeftCandidatesByThread.Add( T( PredictionSize ), threadCount );
		RightCandidatesByThread.Add( T( PredictionSize ), threadCount );
	}

}

template<typename T>
void CCalcSplitGainThreadTask<T>::Run( int threadIndex, int startIndex, int count )
{
	T LeftCandidate( PredictionSize );
	T RightCandidate( PredictionSize );
	// Iterate through features (a separate subset for each thread)
	const int endIndex = startIndex + count;
	for( int index = startIndex; index < endIndex; ++index ) {
		const int usedIndex = UsedFeatures[index];
		T left( PredictionSize ); // the gain for the left node after the split
		T right( PredictionSize ); // for the right node after the split (calculated as the complement to the parent)
		const int firstFeatureIndex = FeaturePos[usedIndex];
		const int lastFeatureIndex = FeaturePos[usedIndex + 1];
		// Iterate through feature values (sorted ascending) looking for the split position
		for( int j = firstFeatureIndex; j < lastFeatureIndex; ++j ) {
			const T& featureStats = HistStats[IdPos[j]];
			left.Add( featureStats );
			right = Node.Statistics;
			right.Sub( left );
			LeftCandidate = left;
			RightCandidate = right;

			// Calculating the gain: if the node is split at this position, 
			// the criterion loses the parent node (bestValue) and replaces it by left.CalcCriterion and right.CalcCriterion
			// In the reference paper, a gamma coefficient is also needed for a new node, but we take that into account while pruning
			double criterion = 0.;
			if( !T::CalcCriterion( criterion, LeftCandidate, RightCandidate, Node.Statistics,
				Params.L1RegFactor, Params.L2RegFactor, Params.MinSubsetHessian, Params.MinSubsetWeight, Params.DenseTreeBoostCoefficient ) )
			{
				continue;
			}

			if( SplitGainsByThread[threadIndex] < criterion ) {
				SplitGainsByThread[threadIndex] = criterion;
				SplitIdsByThread[threadIndex] = j;  // this number refers both to the feature and its value
				// save statistics for childs for case when class if not splitting further
				LeftCandidatesByThread[threadIndex] = LeftCandidate;
				RightCandidatesByThread[threadIndex] = RightCandidate;
			}
		}
	}
}

template<typename T>
int CCalcSplitGainThreadTask<T>::Reduction()
{
	// Choose the best result over all threads
	int result = NotFound;
	for( int t = 0; t < SplitGainsByThread.Size(); ++t ) {
		const double& threadBestGain = SplitGainsByThread[t];
		const int threadBestFeature = SplitIdsByThread[t]; // the coordinate in the array witha all values of all features
		if( BestValue < threadBestGain || ( BestValue == threadBestGain && threadBestFeature < result ) ) {
			BestValue = threadBestGain;
			result = threadBestFeature;
			Node.LeftStatistics = LeftCandidatesByThread[t];
			Node.RightStatistics = RightCandidatesByThread[t];
		}
	}
	return result;
}

//-------------------------------------------------------------------------------------------------------------

// Determining to which subtree each vector belongs
template<typename T>
class CDetermineSubTreeThreadTask : public IGradientBoostThreadTask {
public:
	using TNode = typename CGradientBoostFastHistTreeBuilder<T>::CNode;
	// Create a task
	CDetermineSubTreeThreadTask( IThreadPool& threadPool,
			const CGradientBoostFastHistProblem& problem,
			CArray<int>& vectorSet, const TNode& node ) :
		IGradientBoostThreadTask( threadPool ),
		Problem( problem ),
		VectorSet( vectorSet ),
		Node( node ),
		FeatureIndexes( Problem.GetFeatureIndexes() ),
		FeatureNullValueId( Problem.GetFeatureNullValueId() ),
		FeatureIndex( FeatureIndexes[Node.SplitFeatureId] ),
		VectorPtr( Node.VectorSetPtr ),
		NextId( Problem.GetFeaturePos()[FeatureIndex + 1] - 1 )
	{}
protected:
	// The size of parallelization, max number of elements to perform
	int ParallelizeSize() const override { return Node.VectorSetSize; }
	// Run the process in a separate thread
	void Run( int threadIndex, int startIndex, int count ) override;

	const CGradientBoostFastHistProblem& Problem;
	CArray<int>& VectorSet;
	const TNode& Node;
	const CArray<int>& FeatureIndexes;
	const CArray<int>& FeatureNullValueId;
	const int FeatureIndex;
	const int VectorPtr;
	const int NextId;
};

template<typename T>
void CDetermineSubTreeThreadTask<T>::Run( int /*threadIndex*/, int startIndex, int count )
{
	const int endIndex = startIndex + count;
	for( int index = startIndex; index < endIndex; ++index ) {
		const int* vectorDataPtr = Problem.GetUsedVectorDataPtr( VectorSet[VectorPtr + index] );
		const int vectorDataSize = Problem.GetUsedVectorDataSize( VectorSet[VectorPtr + index] );

		const int pos = FindInsertionPoint<int, Ascending<int>, int>( NextId, vectorDataPtr, vectorDataSize );
		int vectorFeatureId = NotFound; // the ID of the feature value used for split for this vector
		if( pos == 0 || ( FeatureIndexes[vectorDataPtr[pos - 1]] != FeatureIndex ) ) {
			// The vector contains no feature value for the split, therefore this value is 0
			vectorFeatureId = FeatureNullValueId[FeatureIndex];
		} else {
			vectorFeatureId = vectorDataPtr[pos - 1];
		}

		if( vectorFeatureId <= Node.SplitFeatureId ) { // the value is smaller for the smaller ID
			// The vector belongs to the left subtree
			VectorSet[VectorPtr + index] = -( VectorSet[VectorPtr + index] + 1 );
		} // To the right subtree otherwise (no action needed)
	}
}

} // namespace

//-------------------------------------------------------------------------------------------------------------

template<class T>
CGradientBoostFastHistTreeBuilder<T>::CGradientBoostFastHistTreeBuilder(
		const CGradientBoostFastHistTreeBuilderParams& _params, CTextStream* _logStream, int _predictionSize ) :
	threadPool( CreateThreadPool( _params.ThreadCount ) ),
	params( _params, threadPool->Size() ),
	logStream( _logStream ),
	predictionSize( _predictionSize  ),
	histSize( NotFound )
{
	NeoAssert( threadPool != nullptr );
	NeoAssert( params.ThreadCount > 0 );

	NeoAssert( params.MaxTreeDepth > 0 );
	NeoAssert( params.MaxNodesCount > 0 || params.MaxNodesCount == NotFound );
	NeoAssert( abs( params.MinSubsetHessian ) > 0 );
	NeoAssert( params.MaxBins > 1 );
	NeoAssert( params.MinSubsetWeight >= 0 );
}

template<class T>
CGradientBoostFastHistTreeBuilder<T>::~CGradientBoostFastHistTreeBuilder()
{
	delete threadPool;
}

template<class T>
CPtr<CRegressionTree> CGradientBoostFastHistTreeBuilder<T>::Build( const CGradientBoostFastHistProblem& problem,
	const CArray<typename T::Type>& gradients, const CArray<typename T::Type>& hessians, const CArray<double>& weights )
{
	NeoAssert( gradients.Size() == hessians.Size() );

	if( logStream != nullptr ) {
		*logStream << L"\nGradient boost float problem tree building started:\n";
	}

	// Initialization
	initVectorSet( problem.GetUsedVectorCount() );
	initHistData( problem );

	// Creating the tree root
	CNode root( /*level*/0, /*vectorSetPtr*/0, vectorSet.Size() );
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
			if( logStream != nullptr ) {
				*logStream << L"Split result: index = " << featureIndexes[nodes[node].SplitFeatureId]
					<< L" threshold = " << cuts[nodes[node].SplitFeatureId]
					<< L", criterion = " << nodes[node].Statistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor )
					<< L" \n";
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
			nodes[leftNode].Statistics.NullifyLeafClasses( nodes[node].LeftStatistics );
			nodes[rightNode].Statistics.NullifyLeafClasses( nodes[node].RightStatistics );
		} else {
			// The node could not be split
			if( logStream != nullptr ) {
				*logStream << L"Split result: created const node.\t\t"
					<< L"criterion = " << nodes[node].Statistics.CalcCriterion( params.L1RegFactor, params.L2RegFactor )
					<< L" \n";
			}
			freeHist( nodes[node].HistPtr );
			nodes[node].HistPtr = NotFound;
		}
		
	}

	if( logStream != nullptr ) {
		*logStream << L"\nGradient boost float problem tree building finished:\n";
	}

	// Pruning
	if( params.PruneCriterionValue != 0 ) {
		prune( /*node*/0 );
	}

	return buildTree( /*node*/0, featureIndexes, cuts ).Ptr();
}

// Initializes the array of node vector sets
template<class T>
void CGradientBoostFastHistTreeBuilder<T>::initVectorSet( int size )
{
	// For a start, all vectors are assigned to the root node
	vectorSet.SetSize( size );
	for( int i = 0; i < size; ++i ) {
		vectorSet[i] = i;
	}
}

// Initializes the array storing the histograms
template<class T>
void CGradientBoostFastHistTreeBuilder<T>::initHistData( const CGradientBoostFastHistProblem& problem )
{
	// Only the features that are used will be present in the histograms
	const CArray<int>& usedFeatures = problem.GetUsedFeatures();
	const CArray<int>& featurePos = problem.GetFeaturePos();

	idPos.Empty();
	idPos.Add( NotFound, featurePos.Last() );
	histSize = 0;
	for( int i = 0; i < usedFeatures.Size(); ++i ) {
		const int featureIndex = usedFeatures[i];
		for( int j = featurePos[featureIndex]; j < featurePos[featureIndex + 1]; ++j ) {
			idPos[j] = histSize;
			++histSize;
		}
	}

	// The histogram size of tree depth + 1 is sufficient
	histStats.Add( T( predictionSize ), histSize * ( params.MaxTreeDepth + 1 ) );
	freeHists.Empty();
	for( int i = 0; i <= params.MaxTreeDepth; ++i ) {
		freeHists.Add( i * histSize ); // a histogram is identified by the pointer to its start in the histData array
	}
}

// Gets a free histogram 
// A histogram is identified by the pointer to its start in the histData array
template<class T>
int CGradientBoostFastHistTreeBuilder<T>::allocHist()
{
	NeoAssert( !freeHists.IsEmpty() );

	const int result = freeHists.Last();
	freeHists.DeleteLast();
	return result;
}

// Subtract histograms
template<class T>
void CGradientBoostFastHistTreeBuilder<T>::subHist( int firstPtr, int secondPtr )
{
	for( int i = 0; i < histSize; ++i ) {
		histStats[firstPtr + i].Sub( histStats[secondPtr + i] );
	}
}

// Build a histogram on the vectors of the given node
template<class T>
void CGradientBoostFastHistTreeBuilder<T>::buildHist( const CGradientBoostFastHistProblem& problem, const CNode& node,
	const CArray<typename T::Type>& gradients, const CArray<typename T::Type>& hessians, const CArray<double>& weights,
	T& totalStats )
{
	T* const histStatsPtr = histStats.GetPtr() + node.HistPtr;

	// Check if a multithreading task makes sense
	const bool isMultiThreads = ( node.VectorSetSize > ( 4 * params.ThreadCount ) );
	// Building the histogram task in single or parallel threads
	CBuildHistogramThreadTask<T> task( *threadPool, problem, vectorSet, idPos, histStatsPtr,
		node, gradients, hessians, weights, tempHistStats, histSize, predictionSize, totalStats, isMultiThreads );
	if( isMultiThreads ) {
		task.ParallelRun();
		task.Reduction();

		CMergeHistogramsThreadTask<T>( *threadPool, tempHistStats, histStatsPtr, histSize ).ParallelRun();
	} else {
		task.RunInOneThread();
	}
	// Adding zero values
	CAddNullStatsThreadTask<T>( *threadPool, problem, idPos, histStatsPtr, totalStats ).ParallelRun();
}

// Calculates the optimal feature value for splitting the node
// Returns NotFound if splitting is impossible
template<class T>
int CGradientBoostFastHistTreeBuilder<T>::evaluateSplit( const CGradientBoostFastHistProblem& problem, CNode& node ) const
{
	if( ( params.MaxNodesCount != NotFound && ( nodes.Size() + 2 ) > params.MaxNodesCount )
		|| ( node.Level >= params.MaxTreeDepth ) )
	{
		// The nodes limit has been reached
		return NotFound;
	}

	CCalcSplitGainThreadTask<T> task( *threadPool, problem, params, idPos,
		node, ( histStats.GetPtr() + node.HistPtr ), predictionSize, tb );
	task.ParallelRun();
	return task.Reduction();
}

// Splits a node
template<class T>
void CGradientBoostFastHistTreeBuilder<T>::applySplit( const CGradientBoostFastHistProblem& problem, int node,
	int& leftNode, int& rightNode )
{
	NeoAssert( node >= 0 );

	// Determining to which subtree each vector belongs
	CDetermineSubTreeThreadTask<T>( *threadPool, problem, vectorSet, nodes[node] ).ParallelRun();

	const int vectorPtr = nodes[node].VectorSetPtr;
	const int vectorCount = nodes[node].VectorSetSize;

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
template<class T>
bool CGradientBoostFastHistTreeBuilder<T>::prune( int node )
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
template<class T>
CPtr<CLinkedRegressionTree> CGradientBoostFastHistTreeBuilder<T>::buildTree( int node, const CArray<int>& featureIndexes,
	const CArray<float>& cuts ) const
{
	CPtr<CLinkedRegressionTree> result = FINE_DEBUG_NEW CLinkedRegressionTree();

	if( nodes[node].SplitFeatureId == NotFound ) {
		typename T::Type values;
		nodes[node].Statistics.LeafValue( values );
		result->InitLeafNode( values );
	} else {
		CPtr<CLinkedRegressionTree> left = buildTree( nodes[node].Left, featureIndexes, cuts );
		CPtr<CLinkedRegressionTree> right = buildTree( nodes[node].Right, featureIndexes, cuts );
		result->InitSplitNode( *left, *right, featureIndexes[nodes[node].SplitFeatureId], cuts[nodes[node].SplitFeatureId] );
	}

	return result;
}

template class CGradientBoostFastHistTreeBuilder<CGradientBoostStatisticsSingle>;
template class CGradientBoostFastHistTreeBuilder<CGradientBoostStatisticsMulti>;

} // namespace NeoML
