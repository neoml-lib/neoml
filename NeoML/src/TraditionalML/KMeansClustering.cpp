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

#include <NeoML/TraditionalML/KMeansClustering.h>
#include <NeoML/TraditionalML/CommonCluster.h>
#include <NeoML/TraditionalML/VariableMatrix.h>
#include <NeoML/Random.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoMathEngine/ThreadPool.h>
#include <NeoML/Dnn/DnnBlob.h>
#include <NeoML/Dnn/Dnn.h>
#include <float.h>
#include <memory>

namespace NeoML {

static CPtr<CDnnBlob> createDataBlob( IMathEngine& mathEngine, const CFloatMatrixDesc& data )
{
	NeoAssert( data.Columns == nullptr );
	const int vectorCount = data.Height;
	const int featureCount = data.Width;
	CPtr<CDnnBlob> result = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, vectorCount, featureCount ); // no threads
	CFloatHandle currData = result->GetData();
	for( int row = 0; row < data.Height; ++row ) {
		mathEngine.DataExchangeTyped( currData, data.Values + data.PointerB[row], featureCount ); // no threads
		currData += featureCount;
	}
	return result;
}

static CPtr<CDnnBlob> createWeightBlob( IMathEngine& mathEngine, const IClusteringData* data )
{
	const int vectorCount = data->GetVectorCount();
	CPtr<CDnnBlob> weight = CDnnBlob::CreateVector( mathEngine, CT_Float, vectorCount ); // no threads
	CDnnBlobBuffer<float> buffer( *weight, TDnnBlobBufferAccess::Write );
	for( int vectorIndex = 0; vectorIndex < vectorCount; ++vectorIndex ) {
		buffer[vectorIndex] = static_cast<float>( data->GetVectorWeight( vectorIndex ) );
	}
	buffer.Close();
	return weight;
}

//-------------------------------------------------------------------------------------------------------------

namespace {

struct CInertia final {
	explicit CInertia( int count ) { inertia.Add( 0., count ); }
	const double& Get( int index ) const { return inertia[index]; }
	double& Get( int index ) { return inertia[index]; }
	double GetSum() const;

private:
	CArray<double> inertia{};
};

double CInertia::GetSum() const
{
	double sum = 0;
	for( int i = 0; i < inertia.Size(); ++i ) {
		sum += inertia[i];
	}
	return sum;
}

//-------------------------------------------------------------------------------------------------------------

// Function to decide run the task in one thread or in parallel
static inline bool isThreadTaskRelevant( int64_t operationCount )
{
	constexpr int64_t MinOmpOperationCount = 32768;
	return operationCount >= MinOmpOperationCount;
}

// Task which processes a set of elements on a single thread
struct IThreadTask {
public:
	virtual ~IThreadTask() {}
	// The number of separate executors
	int ThreadCount() const { return ThreadPool.Size(); }

	// MAIN METHOD of the task
	// Execute task in 3 steps:
	// 1. PrepareRun  -- make everithing to prepare Run in parallel in ont main thread.
	//                   Run in parallel may be rejected for some tasks, if the Complexity() of that task is too small.
	//                   Returns false, if the Run in parallel is rejected, else true.
	// 2. Run         -- Split into sub-tasks by threads and start parallel execution.
	// 3. Reduction   -- Combine results of each thread in 1 answer.
	void ParallelRun();

protected:
	// Number of dimensions of this task
	enum TSplitDims { T_1D = 0, T_2D = 1, T_Size__ };

	// Create a task 1D or 2D
	IThreadTask( IThreadPool& threadPool, std::initializer_list<int> sizeToParallelize, int alignment = 1 ) :
		ThreadPool( threadPool ),
		SplitDim( static_cast<TSplitDims>( sizeToParallelize.size() - 1 ) ),
		SizeToParallelize( *( sizeToParallelize.begin() ) ),
		SizeToParallelize2D( ( SplitDim == T_2D ) ? *( sizeToParallelize.begin() + 1 ) : 0 ),
		Align( alignment )
	{ NeoAssert( SplitDim == T_1D || SplitDim == T_2D ); }

	// Get way of split the task into sub-tasks
	void RunSplitedByThreads( int threadIndex )
	{ ( SplitDim == T_1D ) ? splitRun1D( threadIndex ) : splitRun2D( threadIndex ); }

	// The size of parallelization, max number of sub-tasks to perform
	virtual int ParallelizeSize() const { return SizeToParallelize; }
	// Total complexity to decide run this task in one thread or in parallel
	virtual int Complexity() const { return ParallelizeSize(); }

	// Step 1: Check the complexity of the task and try to perform in a single thread
	//         Resurns true, if successfully performed, else false
	//         Also may do some preparations for the run in parallel
	virtual bool TryRunOneThread() = 0;
	// Step 2: Run in parallel
	//         Arguments 'indeces' and 'counts' are arrays of size, corresponding to 1D or 2D task
	virtual void Run( int threadIndex, const int* startIndices, const int* counts ) = 0;
	// Step 3: Combine the answer
	virtual void Reduction() = 0;

	IThreadPool& ThreadPool; //executors
	const TSplitDims SplitDim; //defines the method of split run by threads
	const int SizeToParallelize; //max number of sub-tasks 1D
	const int SizeToParallelize2D; //max number of sub-tasks 2D
	const int Align; //defines positions in spliting
private:
	// 1 dimensional split
	void splitRun1D( int threadIndex );
	// 2 dimensional split
	void splitRun2D( int threadIndex );
};

void IThreadTask::ParallelRun()
{
	// Step 1: Try to run in a single thread
	if( TryRunOneThread() ) {
		return;
	}
	// Step 2: Run in parallel
	NEOML_NUM_THREADS( ThreadPool, this, []( int threadIndex, void* ptr ) {
		( ( IThreadTask* )ptr )->RunSplitedByThreads( threadIndex );
	} );
	// Step 3: Combine the answer
	Reduction();
}

void IThreadTask::splitRun1D( int threadIndex )
{
	int index = 0;
	int count = 0;
	// 1 dimensional split
	if( GetTaskIndexAndCount( ThreadCount(), threadIndex, ParallelizeSize(), Align, index, count ) ) {
		Run( threadIndex, &index, &count );
	}
}

void IThreadTask::splitRun2D( int threadIndex )
{
	int heightStarts[2]{};
	int heightCounts[2]{};
	// 2 dimensional split
	if( OmpGetTaskIndexAndCount2D( SizeToParallelize2D, /*alignX*/1, ParallelizeSize(), Align,
		heightStarts[1], heightCounts[1], heightStarts[0], heightCounts[0], ThreadPool.Size(), threadIndex ) )
	{
		Run( threadIndex, heightStarts, heightCounts );
	}
}

//-------------------------------------------------------------------------------------------------------------

// Task which processes a set of elements one-by-one on a single thread
struct IThreadSubTask : public IThreadTask {
public:
	const CFloatMatrixDesc* const Matrix;

protected:
	// Create strictly 1 dimensional task
	IThreadSubTask( IThreadPool& threadPool, int sizeToParallelize ) :
		IThreadTask( threadPool, /*1D*/{ sizeToParallelize } ),
		Matrix( nullptr )
	{}
	// Create strictly 1 dimensional task
	IThreadSubTask( IThreadPool& threadPool, const CFloatMatrixDesc* matrix ) :
		IThreadTask( threadPool, /*1D*/{ 0 } ),
		Matrix( matrix )
	{}

	// Max number of sub-tasks to perform
	int ParallelizeSize() const override final
	{ return Matrix ? Matrix->Height : IThreadTask::ParallelizeSize(); }

	// step 2: special way of run in parallel: perform each element separately
	void Run( int threadIndex, const int* index, const int* count ) override final;
	// Spesial step 2: run in parallel for each element separately
	virtual void RunOnElement( int threadIndex, int index ) = 0;
};

void IThreadSubTask::Run( int threadIndex, const int* startIndex, const int* count )
{
	const int lastIndex = *startIndex + *count - 1;
	for( int i = *startIndex; i <= lastIndex; ++i ) {
		RunOnElement( threadIndex, i );
	}
}

//-------------------------------------------------------------------------------------------------------------

// Distributes all elements over the existing clusters
struct CClassifyAllThreadTask : public IThreadSubTask {
	CClassifyAllThreadTask( IThreadPool& threadPool, const CFloatMatrixDesc& matrix,
			const CObjectArray<CCommonCluster>& clusters, TDistanceFunc distanceFunc ) :
		IThreadSubTask( threadPool, &matrix ),
		Clusters( clusters ),
		DistanceFunc( distanceFunc ),
		ThreadInertia( ThreadCount() )
	{ DataCluster.SetBufferSize( ParallelizeSize() ); }

	CArray<int> DataCluster{}; //the cluster for this element
	double Inertia = 0; //result
protected:
	// Prepare data for each run in parallel only
	bool TryRunOneThread() override;
	// Finds the nearest cluster for the element
	void RunOnElement( int threadIndex, int index ) override;
	void Reduction() override { Inertia = ThreadInertia.GetSum(); }

	const CObjectArray<CCommonCluster>& Clusters;
	const TDistanceFunc DistanceFunc;
	CInertia ThreadInertia;
};

bool CClassifyAllThreadTask::TryRunOneThread()
{
	DataCluster.SetSize( ParallelizeSize() );
	return false;
}

void CClassifyAllThreadTask::RunOnElement( int threadIndex, int dataIndex )
{
	double bestDistance = DBL_MAX;
	int res = NotFound;

	for( int i = 0; i < Clusters.Size(); ++i ) {
		CFloatVectorDesc desc;
		Matrix->GetRow( dataIndex, desc );
		const double distance = Clusters[i]->CalcDistance( desc, DistanceFunc );
		if( distance < bestDistance ) {
			bestDistance = distance;
			res = i;
		}
	}
	NeoAssert( res != NotFound );
	ThreadInertia.Get( threadIndex ) += bestDistance;
	// Each element is assigned to the nearest cluster
	DataCluster[dataIndex] = res;
}

//-------------------------------------------------------------------------------------------------------------

// Updates the clusters and returns true if the clusters were changed, false if they stayed the same
struct CUpdateClustersThreadTask final : public IThreadTask {
	CUpdateClustersThreadTask( IThreadPool& threadPool, const CFloatMatrixDesc& matrix,
		const CObjectArray<CCommonCluster>& clusters, const CArray<double>& weights, const CArray<int>& data );

	CArray<CClusterCenter> OldCenters{};
	bool Result = false;
protected:
	// Decision to run in parallel
	int Complexity() const override { return DataClusters.Size() * Matrix.Width; }
	// If number of clusters is too small, this task may be performed in one thread
	bool TryRunOneThread() override;
	// Recalculate cluster centers
	void Run( int threadIndex, const int* startIndices, const int* counts ) override;
	// Field Result will contain the full answer
	void Reduction() override;

	const CFloatMatrixDesc& Matrix;
	const CObjectArray<CCommonCluster>& Clusters;
	const CArray<double>& Weights;
	const CArray<int>& DataClusters;
	CArray<bool> IsChanged{};
};

CUpdateClustersThreadTask::CUpdateClustersThreadTask( IThreadPool& threadPool, const CFloatMatrixDesc& matrix,
		const CObjectArray<CCommonCluster>& clusters, const CArray<double>& weights, const CArray<int>& data ) :
	IThreadTask( threadPool, /*1D*/{ clusters.Size() } ),
	Matrix( matrix ),
	Clusters( clusters ),
	Weights( weights ),
	DataClusters( data )
{
	IsChanged.SetSize( ThreadCount() );
	OldCenters.SetSize( ParallelizeSize() );
}

bool CUpdateClustersThreadTask::TryRunOneThread()
{
	// Store the old cluster centers
	for( int clusterId = 0; clusterId < ParallelizeSize(); ++clusterId ) {
		OldCenters[clusterId] = Clusters[clusterId]->GetCenter();
		Clusters[clusterId]->Reset( DataClusters.Size() ); //avoid reallocations in threads
	}

	if( !isThreadTaskRelevant( Complexity() ) ) {
		const int firstClusterId = 0;
		const int clustersCount = ParallelizeSize();
		Run( /*threadIndex*/0, &firstClusterId, &clustersCount );
		Reduction();
		return true;
	}
	return false;
}

void CUpdateClustersThreadTask::Run( int threadIndex, const int* firstId, const int* count )
{
	const int lastId = *firstId + *count - 1;
	// Update the cluster contents
	for( int i = 0; i < DataClusters.Size(); ++i ) {
		const int clusterId = DataClusters[i];
		if( *firstId <= clusterId && clusterId <= lastId ) {
			CFloatVectorDesc desc;
			Matrix.GetRow( i, desc );
			Clusters[clusterId]->Add( i, desc, Weights[i] );
		}
	}
	// Update the cluster centers
	IsChanged[threadIndex] = false;
	for( int clusterId = *firstId; clusterId <= lastId; ++clusterId ) {
		if( Clusters[clusterId]->GetElementsCount() > 0 ) {
			Clusters[clusterId]->RecalcCenter();
		}
		// Compare the new cluster centers with the old
		IsChanged[threadIndex] |= ( OldCenters[clusterId].Mean != Clusters[clusterId]->GetCenter().Mean );
	}
}

void CUpdateClustersThreadTask::Reduction()
{
	for( int t = 0; t < ThreadCount(); ++t ) {
		if ( IsChanged[t] == true ) {
			Result = true;
			return;
		}
	}
}

//-------------------------------------------------------------------------------------------------------------

struct CAssignVectorsThreadTask : public IThreadSubTask {
	CAssignVectorsThreadTask( IThreadPool& threadPool, const CFloatMatrixDesc& matrix,
		const CKMeansClustering::CParam& params, const CObjectArray<CCommonCluster>& clusters );

	// Element assignments (objectCount)
	CArray<int> Assignments{};
	// Distances bounds
	CArray<float> UpperBounds{}; // upper bounds for every object (objectCount)
	CVariableMatrix<float> LowerBounds{}; // lower bounds for every object and every cluster (clusterCount x objectCount)

	// Distances between clusters (clusterCount x clusteCount)
	CVariableMatrix<float> ClusterDists{};
	// Distance to the closest center of another cluster (clusterCount)
	CArray<float> ClosestClusterDist{};
	// Distances between old and updated centers of each cluster (clusterCount)
	CArray<float> MoveDistance{};

	const CObjectArray<CCommonCluster>& Clusters;
	const TDistanceFunc DistanceFunc;

protected:
	// Check values for each run in parallel only
	bool TryRunOneThread() override;
	void RunOnElement( int threadIndex, int index ) override;
	void Reduction() override { /*empty*/ }

	// Checks if operation can be omitted
	// If it's true element 'id' can't be reassigned to the cluster 'clusterToProcess'
	// and all of the calculations related to this case can be skipped
	bool isPruned( int clusterToProcess, int id ) const;

	const int InitialClustersCount;
};

// Initializes all required statistics for Elkan algorithm
CAssignVectorsThreadTask::CAssignVectorsThreadTask( IThreadPool& threadPool, const CFloatMatrixDesc& matrix,
		const CKMeansClustering::CParam& params, const CObjectArray<CCommonCluster>& clusters ) :
	IThreadSubTask( threadPool, &matrix ),
	Clusters( clusters ),
	DistanceFunc( params.DistanceFunc ),
	InitialClustersCount( params.InitialClustersCount )
{
	// initialize mapping between elements and cluster index
	Assignments.DeleteAll();
	Assignments.Add( 0, Matrix->Height );

	// initialize bounds
	UpperBounds.DeleteAll();
	UpperBounds.Add( FLT_MAX, Matrix->Height );
	LowerBounds.SetSize( params.InitialClustersCount, Matrix->Height );
	LowerBounds.Set( 0.f );

	// initialize pairwise cluster distances
	ClusterDists.SetSize( params.InitialClustersCount, params.InitialClustersCount );
	ClusterDists.Set( FLT_MAX );

	// initialize closest cluster distances
	ClosestClusterDist.DeleteAll();
	ClosestClusterDist.Add( FLT_MAX, params.InitialClustersCount );

	// initialize move distances
	MoveDistance.DeleteAll();
	MoveDistance.Add( 0, params.InitialClustersCount );
}

bool CAssignVectorsThreadTask::TryRunOneThread()
{
	NeoAssert( Assignments.Size() == Matrix->Height );
	NeoAssert( LowerBounds.SizeX() == InitialClustersCount );
	NeoAssert( UpperBounds.Size() == Assignments.Size() );

	return false;
}

bool CAssignVectorsThreadTask::isPruned( int clusterToProcess, int id ) const
{
	return ( Assignments[id] == clusterToProcess ) ||
		( UpperBounds[id] <= LowerBounds( clusterToProcess, id ) ) ||
		( UpperBounds[id] <= 0.5 * ClusterDists( Assignments[id], clusterToProcess ) );
}

void CAssignVectorsThreadTask::RunOnElement( int /*threadIndex*/, int index )
{
	if( UpperBounds[index] > ClosestClusterDist[Assignments[index]] ) {
		bool recalculate = true;
		for( int c = 0; c < Clusters.Size(); ++c ) {
			if( isPruned( c, index ) ) {
				continue;
			}
			float dist = UpperBounds[index];
			if( recalculate ) {
				recalculate = false;
				const auto& cluster = *Clusters[Assignments[index]];
				const auto rowDesc = Matrix->GetRow( index );
				dist = static_cast<float>( sqrt( cluster.CalcDistance( rowDesc, DistanceFunc ) ) );
				LowerBounds( Assignments[index], index ) = dist;
				UpperBounds[index] = dist;
			}
			if( dist > LowerBounds( c, index ) || dist > 0.5 * ClusterDists( Assignments[index], c ) ) {
				const auto& cluster = *Clusters[c];
				const auto rowDesc = Matrix->GetRow( index );
				const float pointDist = static_cast<float>( sqrt( cluster.CalcDistance( rowDesc, DistanceFunc ) ) );
				LowerBounds( c, index ) = pointDist;
				if( pointDist < dist ) {
					UpperBounds[index] = pointDist;
					Assignments[index] = c;
				}
			}
		}
	}
}

//-------------------------------------------------------------------------------------------------------------

struct CUpdateULBoundsThreadTask : public IThreadSubTask {
	CUpdateULBoundsThreadTask( IThreadPool& threadPool, const CFloatMatrixDesc& matrix,
			CAssignVectorsThreadTask& assigns ) :
		IThreadSubTask( threadPool, &matrix ),
		Assigns( assigns ),
		ThreadInertia( ThreadCount() )
	{}

	double Inertia = 0;
protected:
	void RunOnElement( int threadIndex, int index ) override;
	bool TryRunOneThread() override { /*empty*/ return false; }
	void Reduction() override { Inertia = ThreadInertia.GetSum(); }

	CAssignVectorsThreadTask& Assigns;
	CInertia ThreadInertia;
};

void CUpdateULBoundsThreadTask::RunOnElement( int threadIndex, int index )
{
	for( int c = 0; c < Assigns.Clusters.Size(); ++c ) {
		Assigns.LowerBounds( c, index ) = max( Assigns.LowerBounds( c, index ) - Assigns.MoveDistance[c], 0.f );
	}
	const int c = Assigns.Assignments[index];
	Assigns.UpperBounds[index] += Assigns.MoveDistance[c];
	const auto rowDesc = Matrix->GetRow( index );
	ThreadInertia.Get( threadIndex ) += Assigns.Clusters[c]->CalcDistance( rowDesc, Assigns.DistanceFunc );
}

//-------------------------------------------------------------------------------------------------------------

struct IMathEngineThreadTask : public IThreadTask {
	static constexpr int FloatTaskAlignment = 16;
	// Create a task 1D or 2D
	IMathEngineThreadTask( IThreadPool& threadPool, std::initializer_list<int> sizeToParallelize,
			int align, IMathEngine& mathEngine, const CFloatHandle& result ) :
		IThreadTask( threadPool, sizeToParallelize, align ),
		MathEngine( mathEngine ),
		Result( result )
	{}
protected:
	bool TryRunOneThread() override;
	void Reduction() override final { /*empty*/ }

	IMathEngine& MathEngine;
	const CFloatHandle& Result;
};

bool IMathEngineThreadTask::TryRunOneThread()
{
	if( !isThreadTaskRelevant( Complexity() ) ) {
		const int starts[]{ 0, 0 };
		const int counts[]{ ParallelizeSize(), SizeToParallelize2D };
		Run( /*threadIndex*/0, starts, counts );
		Reduction();
		return true;
	}
	return false;
}

//-------------------------------------------------------------------------------------------------------------

struct IBinaryMathEngineThreadTask : public IMathEngineThreadTask {
	// Create a task 1D or 2D
	IBinaryMathEngineThreadTask( IThreadPool& threadPool, std::initializer_list<int> sizeToParallelize,
			int align, IMathEngine& mathEngine, const CConstFloatHandle& first, const CConstFloatHandle& second,
			const CFloatHandle& result ) :
		IMathEngineThreadTask( threadPool, sizeToParallelize, align, mathEngine, result ),
		First( first ),
		Second( second )
	{}
protected:
	const CConstFloatHandle& First;
	const CConstFloatHandle& Second;
};

//-------------------------------------------------------------------------------------------------------------

struct CVectorFillThreadTask : public IMathEngineThreadTask {
	CVectorFillThreadTask( IThreadPool& threadPool, IMathEngine& mathEngine, int vectorSize,
			const CFloatHandle& result, float value ) :
		IMathEngineThreadTask( threadPool, /*1D*/{ vectorSize }, FloatTaskAlignment, mathEngine, result ),
		Value( value )
	{}
protected:
	void Run( int /*threadIndex*/, const int* index, const int* count ) override
	{ MathEngine.VectorFill( Result + *index, Value, *count ); }

	const float Value;
};

//-------------------------------------------------------------------------------------------------------------

struct CVectorCopyThreadTask : public IMathEngineThreadTask {
	CVectorCopyThreadTask( IThreadPool& threadPool, IMathEngine& mathEngine, int vectorSize,
			const CFloatHandle& dest, const CConstFloatHandle& src ) :
		IMathEngineThreadTask( threadPool, /*1D*/{ vectorSize }, FloatTaskAlignment, mathEngine, dest ),
		Source( src )
	{}
protected:
	void Run( int /*threadIndex*/, const int* index, const int* count ) override
	{ MathEngine.VectorCopy( Result + *index, Source + *index, *count ); }

	const CConstFloatHandle& Source;
};

//-------------------------------------------------------------------------------------------------------------

struct CVectorSubThreadTask : public IBinaryMathEngineThreadTask {
	CVectorSubThreadTask( IThreadPool& threadPool, IMathEngine& mathEngine, int vectorSize,
			const CConstFloatHandle& first, const CConstFloatHandle& second, const CFloatHandle& result ) :
		IBinaryMathEngineThreadTask( threadPool, /*1D*/{ vectorSize }, FloatTaskAlignment,
			mathEngine, first, second, result )
	{}
protected:
	void Run( int /*threadIndex*/, const int* index, const int* count ) override
	{ MathEngine.VectorSub( First + *index, Second + *index, Result + *index, *count ); }
};

//-------------------------------------------------------------------------------------------------------------

struct CVectorMultiplyThreadTask : public IBinaryMathEngineThreadTask {
	CVectorMultiplyThreadTask( IThreadPool& threadPool, IMathEngine& mathEngine, int vectorSize,
			const CConstFloatHandle& first, const CFloatHandle& result, const CConstFloatHandle& multiplier ) :
		IBinaryMathEngineThreadTask( threadPool, /*1D*/{ vectorSize }, /*align*/1,
			mathEngine, first, multiplier, result )
	{}
protected:
	void Run( int /*threadIndex*/, const int* index, const int* count ) override
	{ MathEngine.VectorMultiply( First + *index, Result + *index, *count, Second ); }
};

//-------------------------------------------------------------------------------------------------------------

struct CVectorEltwiseMultiplyThreadTask : public IBinaryMathEngineThreadTask {
	CVectorEltwiseMultiplyThreadTask( IThreadPool& threadPool, IMathEngine& mathEngine, int vectorSize,
			const CConstFloatHandle& first, const CConstFloatHandle& second, const CFloatHandle& result ) :
		IBinaryMathEngineThreadTask( threadPool, /*1D*/{ vectorSize }, FloatTaskAlignment,
			mathEngine, first, second, result )
	{}
protected:
	void Run( int /*threadIndex*/, const int* index, const int* count ) override
	{ MathEngine.VectorEltwiseMultiply( First + *index, Second + *index, Result + *index, *count ); }
};

//-------------------------------------------------------------------------------------------------------------

struct CDiagMxMThreadTask : public IBinaryMathEngineThreadTask {
	CDiagMxMThreadTask( IThreadPool& threadPool, IMathEngine& mathEngine, const CConstFloatHandle& first, int firstSize,
			const CConstFloatHandle& second, int secondWidth, const CFloatHandle& result, int resultBufferSize ) :
		IBinaryMathEngineThreadTask( threadPool, /*1D*/{ firstSize }, /*align*/1,
			mathEngine, second, first, result ),
		SecondWidth( secondWidth )
	{ NeoAssert( resultBufferSize >= firstSize * secondWidth ); }
protected:
	int Complexity() const override { return ParallelizeSize() * SecondWidth; }
	void Run( int /*threadIndex*/, const int* index, const int* /*count*/ ) override
	{ MathEngine.VectorMultiply( First + *index * SecondWidth, Result + *index * SecondWidth, SecondWidth, Second + *index ); }

	const int SecondWidth;
};

//-------------------------------------------------------------------------------------------------------------

struct CVectorMultichannelLookupAndAddToTableThreadTask : public IThreadSubTask {
	CVectorMultichannelLookupAndAddToTableThreadTask( IThreadPool& threadPool,
			IMathEngine& mathEngine, int vectorSize, int channelCount, const CConstIntHandle& input,
			const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
			const CConstFloatHandle& mult, const CConstFloatHandle& matrix, int /*outputChannels*/ ) :
		IThreadSubTask( threadPool, vectorSize ),
		MathEngine( mathEngine ),
		LookupChannels( 1 ),
		LookupInput( input ),
		LookupHandles( lookupHandles ),
		LookupDimensions( lookupDimensions ),
		LookupCount( lookupCount ),
		LookupMult( mult ),
		LookupMatrix( matrix )
	{ NeoAssert( channelCount == 1 ); }
protected:
	bool TryRunOneThread() override { /*empty*/ return false; }
	void RunOnElement( int threadIndex, int index ) override;
	void Reduction() override { /*empty*/ }

	IMathEngine& MathEngine;
	const int LookupChannels;
	const CConstIntHandle& LookupInput;
	const CFloatHandle* const LookupHandles;
	const CLookupDimension* const LookupDimensions;
	const int LookupCount;
	const CConstFloatHandle& LookupMult;
	const CConstFloatHandle& LookupMatrix;
};

void CVectorMultichannelLookupAndAddToTableThreadTask::RunOnElement( int /*threadIndex*/, int index )
{
	const int remained = LookupChannels - LookupCount;
	const int vectorSize = LookupDimensions[0].VectorSize;
	// skip unmapped updates
	CConstIntHandle input = LookupInput + index * ( remained + 1 );
	CConstFloatHandle matrix = LookupMatrix + index * ( remained + vectorSize );
	// only if j=0 < LookupChannels=1
	const int indexInput = input.GetValue();
	NeoPresume( 0 <= indexInput && indexInput < LookupDimensions[0].VectorCount );
	const CFloatHandle pos = LookupHandles[0] + indexInput * vectorSize;
	MathEngine.VectorMultiplyAndAdd( pos, matrix, pos, vectorSize, LookupMult );
}

//-------------------------------------------------------------------------------------------------------------

struct CVectorAddToMatrixRowsThreadTask : public IBinaryMathEngineThreadTask {
	CVectorAddToMatrixRowsThreadTask( IThreadPool& threadPool, IMathEngine& mathEngine, int batchSize,
			const CConstFloatHandle& matrix, const CFloatHandle& result, int matrixHeight, int matrixWidth,
			const CConstFloatHandle& vector ) :
		IBinaryMathEngineThreadTask( threadPool, /*2D*/{ matrixWidth, matrixHeight }, /*align*/1,
			mathEngine, matrix, vector, result),
		MatrixHeight( matrixHeight ),
		MatrixWidth( matrixWidth )
	{ NeoAssert( batchSize == 1 ); }
protected:
	int Complexity() const override { return MatrixHeight * MatrixWidth; }
	void Run( int threadIndex, const int* startIndices, const int* counts ) override;

	enum { TMatrixWidth, TMatrixHeight };
	const int MatrixHeight;
	const int MatrixWidth;
};

void CVectorAddToMatrixRowsThreadTask::Run( int /*threadIndex*/, const int* startIndices, const int* counts )
{
	const int offset = startIndices[TMatrixHeight] * MatrixWidth + startIndices[TMatrixWidth];
	auto matrix = First + offset;
	auto vector = Second + offset;
	auto result = Result + offset;

	for( int h = 0; h < counts[TMatrixHeight]; ++h ) {
		MathEngine.VectorAdd( matrix, vector, result, counts[TMatrixWidth] );
		matrix += MatrixWidth;
		result += MatrixWidth;
	}
}

//-------------------------------------------------------------------------------------------------------------

struct CMxMTThreadTask : public IBinaryMathEngineThreadTask {
	CMxMTThreadTask( IThreadPool& threadPool, IMathEngine& mathEngine, int batchSize,
			const CConstFloatHandle& first, int firstHeight, int firstWidth,
			const CConstFloatHandle& second, int secondHeight,
			const CFloatHandle& result, int resultBufferSize );
protected:
	int Complexity() const override { return FirstWidth * FirstHeight * SecondHeight; }
	void Run( int threadIndex, const int* start, const int* count ) override;

	enum { TSecondHeight, TFirstHeight };
	const int FirstHeight;
	const int FirstWidth;
	const int SecondHeight;
};

CMxMTThreadTask::CMxMTThreadTask( IThreadPool& threadPool, IMathEngine& mathEngine, int batchSize,
		const CConstFloatHandle& first, int firstHeight, int firstWidth,
		const CConstFloatHandle& second, int secondHeight, const CFloatHandle& result, int resultBufferSize ) :
	IBinaryMathEngineThreadTask( threadPool, /*2D*/{ secondHeight, firstHeight }, NeoML::FloatAlignment,
		mathEngine, first, second, result ),
	FirstHeight( firstHeight ),
	FirstWidth( firstWidth ),
	SecondHeight( secondHeight )
{
	NeoAssert( batchSize == 1 );
	NeoAssert( resultBufferSize >= firstHeight * secondHeight );
}

void CMxMTThreadTask::Run( int /*threadIndex*/, const int* startIndices, const int* counts )
{
	const CConstFloatHandle& first = First + startIndices[TFirstHeight] * FirstWidth;
	const CConstFloatHandle& second = Second + startIndices[TSecondHeight] * FirstWidth;
	const CFloatHandle& result = Result + startIndices[TFirstHeight] * SecondHeight + startIndices[TSecondHeight];

	MathEngine.MultiplyMatrixByTransposedMatrix( first, counts[TFirstHeight], FirstWidth, FirstWidth/*RowSize*/,
		second, counts[TSecondHeight], FirstWidth/*RowSize*/, result, SecondHeight/*RowSize*/, 0/*resBufferSize*/ );
}

} // namespace

//-------------------------------------------------------------------------------------------------------------

CKMeansClustering::CKMeansClustering( const CArray<CClusterCenter>& _clusters, const CParam& _params ) :
	CKMeansClustering( _params )
{
	NeoAssert( !_clusters.IsEmpty() );
	NeoAssert( _clusters.Size() == params.InitialClustersCount );

	_clusters.CopyTo( initialClusterCenters );
}

CKMeansClustering::CKMeansClustering( const CParam& _params ) :
	threadPool( CreateThreadPool( _params.ThreadCount ) ),
	params( _params, threadPool->Size() )
{
	NeoAssert( threadPool != nullptr );
}

CKMeansClustering::~CKMeansClustering()
{
	delete threadPool;
}

bool CKMeansClustering::Clusterize( IClusteringData* input, CClusteringResult& result )
{
	double inertia;
	// Run first clusterization with params.Seed as initial seed
	bool succeeded = runClusterization( input, params.Seed, result, inertia );

	if( params.RunCount == 1 ) {
		return succeeded;
	}

	// Run other clusterization with seeds generated from CRandom( params.Seed )
	CRandom random( params.Seed );
	for( int runIndex = 1; runIndex < params.RunCount; ++runIndex ) {
		CClusteringResult newResult;
		double newInertia;
		bool runSucceeded = runClusterization( input, static_cast<int>( random.Next() ), newResult, newInertia );
		// Update result if current run is better (== has less inertia)
		if( newInertia < inertia ) {
			inertia = newInertia;
			succeeded = runSucceeded;
			newResult.CopyTo( result );
		}
	}
	return succeeded;
}

bool CKMeansClustering::runClusterization( IClusteringData* input, int seed, CClusteringResult& result, double& inertia )
{
	NeoAssert( input != 0 );

	CFloatMatrixDesc matrix = input->GetMatrix();
	NeoAssert( matrix.Height == input->GetVectorCount() );
	NeoAssert( matrix.Width == input->GetFeaturesCount() );

	if( log != 0 ) {
		*log << "\nK-means clustering started:\n";
	}

	// Specific optimized case (uses MathEngine)
	if( matrix.Columns == nullptr && params.DistanceFunc == DF_Euclid && params.Algo == KMA_Lloyd ) {
		return denseLloydL2Clusterize( input, seed, result, inertia );
	}

	CArray<double> weights;
	for( int i = 0; i < input->GetVectorCount(); ++i ) {
		weights.Add( input->GetVectorWeight( i ) );
	}

	selectInitialClusters( matrix, seed );

	if( log != 0 ) {
		*log << "Initial clusters:\n";

		for( int i = 0; i < clusters.Size(); ++i ) {
			*log << *clusters[i] << "\n";
		}
	}

	bool success = clusterize( matrix, weights, inertia );

	result.ClusterCount = clusters.Size();
	result.Data.SetSize( matrix.Height );
	result.Clusters.SetBufferSize( clusters.Size() );

	for( int i = 0; i < clusters.Size(); i++ ) {
		CArray<int> elements;
		clusters[i]->GetAllElements( elements );
		for( int j = 0; j < elements.Size(); ++j ) {
			result.Data[elements[j]] = i;
		}
		result.Clusters.Add( clusters[i]->GetCenter() );
	}

	if( log != 0 ) {
		if( success ) {
			*log << "\nSuccessful!\n";
		} else {
			*log << "\nNeed more iterations!\n";
		}
	}
	return success;
}

bool CKMeansClustering::denseLloydL2Clusterize( IClusteringData* rawData, int seed, CClusteringResult& result, double& inertia )
{
	NeoAssert( params.DistanceFunc == DF_Euclid );
	NeoAssert( params.Algo == KMA_Lloyd );
	NeoAssert( rawData->GetVectorCount() > params.InitialClustersCount );
	const int vectorCount = rawData->GetVectorCount();
	const int featureCount = rawData->GetFeaturesCount();
	const int clusterCount = params.InitialClustersCount;

	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( /*params.ThreadCount*/1, /*memoryLimit*/0 ) );

	CPtr<CDnnBlob> data = createDataBlob( *mathEngine, rawData->GetMatrix() ); // no threads
	CPtr<CDnnBlob> weight = createWeightBlob( *mathEngine, rawData ); // no threads
	CPtr<CDnnBlob> centers = CDnnBlob::CreateDataBlob( *mathEngine, CT_Float, 1, clusterCount, featureCount ); // no threads

	selectInitialClusters( *data, seed, *centers );

	bool success = false;

	CPtr<CDnnBlob> sizes = CDnnBlob::CreateVector( *mathEngine, CT_Float, clusterCount ); // no threads
	CPtr<CDnnBlob> labels = CDnnBlob::CreateVector( *mathEngine, CT_Int, vectorCount ); // no threads

	static_assert( KMA_Count == 2, "KMA_Count != 2" );
	switch( params.Algo ) {
		case KMA_Lloyd:
			success = lloydBlobClusterization( *data, *weight, *centers, *sizes, *labels, inertia );
			break;
		case KMA_Elkan:
			// Only Lloyd algorithm is supported for dense data
		default:
			NeoAssert( false );
	}

	// finalizing results
	result.ClusterCount = clusterCount;
	result.Data.SetSize( vectorCount );
	labels->CopyTo( result.Data.GetPtr() );

	CPtr<CDnnBlob> variances = CDnnBlob::CreateDataBlob( *mathEngine, CT_Float, 1, clusterCount, featureCount ); // no threads
	calcClusterVariances( *data, *labels, *centers, *sizes, *variances );

	CConstFloatHandle rawCentersPtr = centers->GetData();
	CConstFloatHandle rawVariancesPtr = variances->GetData();

	result.Clusters.SetBufferSize( result.ClusterCount );

	for( int i = 0; i < clusterCount; ++i ) {
		CFloatVector center( featureCount );
		CFloatVector variance( featureCount );

		mathEngine->DataExchangeTyped( center.CopyOnWrite(), rawCentersPtr, featureCount ); // no threads
		mathEngine->DataExchangeTyped( variance.CopyOnWrite(), rawVariancesPtr, featureCount ); // no threads

		CClusterCenter& currentCenter = result.Clusters.Append();
		currentCenter.Mean = center;
		currentCenter.Disp = variance;
		currentCenter.Norm = DotProduct( currentCenter.Mean, currentCenter.Mean );
		currentCenter.Weight = 0;

		rawCentersPtr += featureCount;
		rawVariancesPtr += featureCount;
	}

	result.ClusterCount = result.Clusters.Size();
	NeoAssert( result.Clusters.Size() > 0 );
	return success;
}

// Selects the initial clusters
void CKMeansClustering::selectInitialClusters( const CFloatMatrixDesc& matrix, int seed )
{
	if( !clusters.IsEmpty() ) {
		// The initial clusters have been set already
		return;
	}

	// If the initial cluster centers have been specified, create the clusters from that
	if( !initialClusterCenters.IsEmpty() ) {
		clusters.SetBufferSize( params.InitialClustersCount );
		for( int i = 0; i < initialClusterCenters.Size(); ++i ) {
			clusters.Add( FINE_DEBUG_NEW CCommonCluster( initialClusterCenters[i] ) );
		}
		return;
	}

	if( params.Initialization == KMI_Default ) {
		defaultInitialization( matrix, seed );
	} else if( params.Initialization == KMI_KMeansPlusPlus ) {
		kMeansPlusPlusInitialization( matrix, seed );
	} else {
		NeoAssert( false );
	}
}

void CKMeansClustering::defaultInitialization( const CFloatMatrixDesc& matrix, int seed )
{
	const int vectorCount = matrix.Height;
	CCommonCluster::CParams clusterParam;
	clusterParam.MinElementCountForVariance = 1;
	clusters.SetBufferSize( params.InitialClustersCount );

	if( seed == 0xCEA ) {
		// !Backward compatibility!
		// If the cluster centers have not been specified, use some elements of the input data
		const int step = max( vectorCount / params.InitialClustersCount, 1 );
		NeoAssert( step > 0 );
		for( int i = 0; i < params.InitialClustersCount; ++i ) {
			CFloatVectorDesc desc;
			matrix.GetRow( ( i * step ) % vectorCount, desc );
			CFloatVector mean( matrix.Width, desc );
			clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( mean ), clusterParam ) );
		}
	} else {
		CArray<int> perm;
		perm.SetSize( matrix.Height );
		for( int i = 0; i < perm.Size(); ++i ) {
			perm[i] = i;
		}
		CRandom random( seed );
		for( int i = 0; i < perm.Size(); ++i ) {
			const int j = random.UniformInt( 0, matrix.Height - 1 );
			if( i != j ) {
				swap( perm[i], perm[j] );
			}
		}
		for( int i = 0; i < params.InitialClustersCount; ++i ) {
			CFloatVectorDesc desc;
			matrix.GetRow( perm[i] );
			CFloatVector mean( matrix.Width, desc );
			clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( mean ), clusterParam ) );
		}
	}
}

void CKMeansClustering::kMeansPlusPlusInitialization( const CFloatMatrixDesc& matrix, int seed )
{
	const int vectorCount = matrix.Height;
	NeoAssert( params.InitialClustersCount <= vectorCount );
	CCommonCluster::CParams clusterParam;
	clusterParam.MinElementCountForVariance = 1;

	// Use random element as the first center
	CRandom random( seed );
	const int firstCenterIndex = random.UniformInt( 0, vectorCount - 1 );
	CFloatVector firstCenter( matrix.Width, matrix.GetRow( firstCenterIndex ) );
	clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( firstCenter ), clusterParam ) );

	CArray<double> dists;
	dists.Add( HUGE_VAL, vectorCount );

	while( clusters.Size() < params.InitialClustersCount ) {
		double distSum = 0;
		for( int i = 0; i < vectorCount; ++i ) {
			dists[i] = min( dists[i], clusters.Last()->CalcDistance( matrix.GetRow( i ), params.DistanceFunc ) );
			if( params.DistanceFunc == DF_Cosine ) {
				dists[i] *= dists[i];
			}
			distSum += dists[i];
		}

		double selectedSum = random.Uniform( 0, distSum );
		double prefixSum = 0;
		int nextCenterIndex = 0;

		while( prefixSum + dists[nextCenterIndex] < selectedSum ) {
			prefixSum += dists[nextCenterIndex++];
		}

		CFloatVector nextCenter( matrix.Width, matrix.GetRow( nextCenterIndex ) );
		clusters.Add( new CCommonCluster( CClusterCenter( nextCenter ) ) );
	}
}

bool CKMeansClustering::clusterize( const CFloatMatrixDesc& matrix, const CArray<double>& weights, double& inertia )
{
	if( params.Algo == KMA_Lloyd ) {
		return lloydClusterization( matrix, weights, inertia );
	} else {
		return elkanClusterization( matrix, weights, inertia );
	}
}

bool CKMeansClustering::lloydClusterization( const CFloatMatrixDesc& matrix, const CArray<double>& weights, double& inertia )
{
	bool success = false;

	CClassifyAllThreadTask classifyTask( *GetThreadPool(), matrix, clusters, params.DistanceFunc );
	for( int i = 0; i < params.MaxIterations; ++i ) {
		classifyTask.ParallelRun();
		inertia = classifyTask.Inertia;

		if( log != 0 ) {
			*log << "\n[Step " << i << "]\nData classification result:\n";
			for( int j = 0; j < clusters.Size(); ++j ) {
				*log << "Cluster " << j << ": \n";
				*log << *clusters[j];
			}
		}

		CUpdateClustersThreadTask updateClustersTask( *GetThreadPool(), matrix, clusters, weights, classifyTask.DataCluster );
		updateClustersTask.ParallelRun();
		if( !updateClustersTask.Result ) {
			// Cluster centers stay the same, no need to continue
			success = true;
			break;
		}
	}
	return success;
}

bool CKMeansClustering::elkanClusterization( const CFloatMatrixDesc& matrix, const CArray<double>& weights, double& inertia )
{
	// Metric must support triangle inequality
	NeoAssert( params.DistanceFunc == DF_Euclid );

	CAssignVectorsThreadTask assignVectorsTask( *GetThreadPool(), matrix, params, clusters );
	double lastResidual = DBL_MAX;
	for( int i = 0; i < params.MaxIterations; ++i ) {
		// Calculaate pairwise and closest cluster distances
		computeClustersDists( assignVectorsTask.ClusterDists, assignVectorsTask.ClosestClusterDist );
		// Reassign vectors
		assignVectorsTask.ParallelRun();
		// Recalculate centers
		CUpdateClustersThreadTask updateClustersTask( *GetThreadPool(), matrix, clusters, weights, assignVectorsTask.Assignments );
		updateClustersTask.ParallelRun();
		// Update move distances
		updateMoveDistance( updateClustersTask.OldCenters, assignVectorsTask.MoveDistance );
		// Update bounds based on move distance
		CUpdateULBoundsThreadTask updateBoundsTask( *GetThreadPool(), matrix, assignVectorsTask );
		updateBoundsTask.ParallelRun();
		inertia = updateBoundsTask.Inertia;
		// Check stop criteria
		if( abs( inertia - lastResidual ) <= params.Tolerance ) {
			return true;
		}
		lastResidual = inertia;
		if( log != 0 ) {
			*log << L"Step " << i << L"Itertia: " << inertia << L"\n";
		}
	}
	return false;
}

void CKMeansClustering::computeClustersDists( CVariableMatrix<float>& dists, CArray<float>& closestCluster ) const
{
	for( int i = 0; i < clusters.Size(); ++i ) {
		closestCluster[i] = FLT_MAX;
	}
	for( int i = 0; i < clusters.Size() - 1; ++i ) {
		dists( i, i ) = FLT_MAX;
		for( int j = i + 1; j < clusters.Size(); ++j ) {
			const float dist = static_cast<float>(
				sqrt( clusters[i]->CalcDistance( *clusters[j], params.DistanceFunc ) ) );
			dists( i, j ) = dist;
			dists( j, i ) = dist;
			closestCluster[i] = min( 0.5f * dist, closestCluster[i] );
			closestCluster[j] = min( 0.5f * dist, closestCluster[j] );
		}
	}
}

void CKMeansClustering::updateMoveDistance( const CArray<CClusterCenter>& oldCenters, CArray<float>& moveDistance ) const
{
	for( int i = 0; i < clusters.Size(); ++i ) {
		const double moveNorm = sqrt( clusters[i]->CalcDistance( oldCenters[i].Mean, params.DistanceFunc ) );
		moveDistance[i] = static_cast<float>( moveNorm );
	}
}

// Selects initial centers from dense data
void CKMeansClustering::selectInitialClusters( const CDnnBlob& data, int seed, CDnnBlob& centers )
{
	const int featureCount = data.GetObjectSize();
	if( !initialClusterCenters.IsEmpty() ) {
		CDnnBlobBuffer<float> buffer( centers, TDnnBlobBufferAccess::Write );
		float* currPtr = buffer;
		for( int i = 0; i < params.InitialClustersCount; ++i ) {
			::memcpy( currPtr, initialClusterCenters[i].Mean.GetPtr(), featureCount * sizeof( float ) );
			currPtr += featureCount;
		}
		buffer.Close();
		return;
	}

	static_assert( KMI_Count == 2, "KMI_Count != 2" );
	switch( params.Initialization ) {
		case KMI_Default:
			defaultInitialization( data, seed, centers );
			break;
		case KMI_KMeansPlusPlus:
			kMeansPlusPlusInitialization( data, seed, centers );
			break;
		default:
			NeoAssert( false );
	}
}

// Selects initial centers by using default algo from dense data
void CKMeansClustering::defaultInitialization( const CDnnBlob& data, int seed, CDnnBlob& centers )
{
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();
	NeoAssert( centers.GetObjectCount() == params.InitialClustersCount );
	NeoAssert( centers.GetObjectSize() == featureCount );
	IMathEngine& mathEngine = data.GetMathEngine();

	if( seed == 0xCEA ) {
		// !Backward compatibility!
		const int step = max( vectorCount / params.InitialClustersCount, 1 );
		NeoAssert( step > 0 );

		for( int i = 0; i < params.InitialClustersCount; ++i ) {
			const int pos = ( i * step ) % vectorCount;
			CVectorCopyThreadTask( *GetThreadPool(), mathEngine, featureCount,
				centers.GetObjectData( i ), data.GetObjectData( pos ) ).ParallelRun();
		}
	} else {
		CArray<int> perm;
		perm.SetSize( vectorCount );
		for( int i = 0; i < perm.Size(); ++i ) {
			perm[i] = i;
		}
		CRandom random( seed );
		for( int i = 0; i < perm.Size(); ++i ) {
			const int j = random.UniformInt( 0, vectorCount - 1 );
			if( i != j ) {
				swap( perm[i], perm[j] );
			}
		}

		for( int i = 0; i < params.InitialClustersCount; ++i ) {
			CVectorCopyThreadTask( *GetThreadPool(), mathEngine, featureCount,
				centers.GetObjectData( i ), data.GetObjectData( perm[i] ) ).ParallelRun();
		}
	}
}

// Selects initial centers by using K-Means++ algo from dense data
void CKMeansClustering::kMeansPlusPlusInitialization( const CDnnBlob& data, int seed, CDnnBlob& centers )
{
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();
	// Select random vector
	CRandom random( seed );
	const int firstChoice = random.UniformInt( 0, vectorCount - 1 );
	// Copy first vector
	IMathEngine& mathEngine = centers.GetMathEngine();
	CVectorCopyThreadTask( *GetThreadPool(), mathEngine, data.GetObjectSize(),
		centers.GetData(), data.GetObjectData( firstChoice ) ).ParallelRun();

	CFloatHandleStackVar stackBuff( mathEngine, vectorCount + 1 );
	CFloatHandle sumBlob = stackBuff;
	CFloatHandle currDists = stackBuff + 1;

	CHashTable<int> usedVectors;
	CPtr<CDnnBlob> prevDists = CDnnBlob::CreateVector( mathEngine, CT_Float, vectorCount ); // no threads
	mathEngine.MatrixRowsToVectorSquaredL2Distance( data.GetData(), vectorCount, featureCount, // no threads
		centers.GetData(), prevDists->GetData() );
	for( int k = 1; k < params.InitialClustersCount; ++k ) {
		CConstFloatHandle currentVector = centers.GetObjectData( k - 1 );
		mathEngine.MatrixRowsToVectorSquaredL2Distance( data.GetData(), vectorCount, featureCount, // no threads
			currentVector, currDists );
		mathEngine.VectorEltwiseMin( currDists, prevDists->GetData(), prevDists->GetData(), prevDists->GetDataSize() ); // no threads
		mathEngine.VectorSum( prevDists->GetData(), prevDists->GetDataSize(), sumBlob ); // no threads

		const double sumValue = static_cast<double>( sumBlob.GetValue() );
		const double scaledSum = random.Uniform( 0, 1 ) * sumValue;

		CArray<float> squares;
		squares.SetSize( prevDists->GetDataSize() );
		prevDists->CopyTo( squares.GetPtr() );

		double prefixSum = 0;
		int nextCoice = -1;
		for( int i = 0; i < vectorCount; ++i ) {
			prefixSum += squares[i];
			if( prefixSum > scaledSum ) {
				nextCoice = i;
				break;
			}
		}

		NeoAssert( nextCoice != -1 );
		NeoAssert( !usedVectors.Has( nextCoice ) );
		usedVectors.Add( nextCoice );
		CVectorCopyThreadTask( *GetThreadPool(), mathEngine, featureCount,
			centers.GetObjectData( k ), data.GetObjectData( nextCoice ) ).ParallelRun();
	}
}

// Clusterizes dense data by using Lloyd algorithm
bool CKMeansClustering::lloydBlobClusterization( const CDnnBlob& data, const CDnnBlob& weight,
	CDnnBlob& centers, CDnnBlob& sizes, CDnnBlob& labels, double& inertia )
{
	double prevInertia = FLT_MAX;
	const float eps = 1e-3f;

	IMathEngine& mathEngine = data.GetMathEngine();
	// pre-calculate l2-norm of input data
	CPtr<CDnnBlob> squaredData = CDnnBlob::CreateVector( mathEngine, CT_Float, data.GetObjectCount() ); // no threads
	mathEngine.RowMultiplyMatrixByMatrix( data.GetData(), data.GetData(), data.GetObjectCount(), // no threads
		data.GetObjectSize(), squaredData->GetData() );
	for( int iter = 0; iter < params.MaxIterations; ++iter ) {
		inertia = assignClosest( data, *squaredData, weight, centers, labels );
		recalcCenters( data, weight, labels, centers, sizes );
		if( abs( prevInertia - inertia ) < eps ) {
			return true;
		}
		prevInertia = inertia;
	}
	return false;
}

static const int DistanceBufferSize = 2 * 1024 * 1024;

// Calculates distances between every point and the closest cluster
static void calcClosestDistances( IThreadPool& threadPool, const CDnnBlob& data, const CDnnBlob& squaredData,
	const CDnnBlob& centers, CFloatHandle& closestDist, CIntHandle& labels )
{
	IMathEngine& mathEngine = data.GetMathEngine();
	const int clusterCount = centers.GetObjectCount();
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();

	int batchSize = min( vectorCount, max( 1, static_cast<int>( DistanceBufferSize / ( sizeof( float ) * clusterCount ) ) ) );

	CFloatHandleStackVar stackBuff( mathEngine, batchSize * clusterCount + clusterCount + 1 );
	CFloatHandle distances = stackBuff.GetHandle();
	CFloatHandle squaredCenters = stackBuff.GetHandle() + batchSize * clusterCount;
	CFloatHandle minusTwo = stackBuff.GetHandle() + batchSize * clusterCount + clusterCount;

	minusTwo.SetValue( -2.f );
	// pre-calculate l2-norm of current cluster centers
	mathEngine.RowMultiplyMatrixByMatrix( centers.GetData(), centers.GetData(), clusterCount, featureCount, squaredCenters ); // no threads

	int batchStart = 0;
	CConstFloatHandle currData = data.GetData();
	CConstFloatHandle currSquaredData = squaredData.GetData();
	CFloatHandle currClosesDist = closestDist;
	CIntHandle currLabels = labels;
	while( batchStart < vectorCount ) {
		batchSize = min( batchSize, vectorCount - batchStart );

		// (a - b)^2 = a^2 + b^2 - 2*a*b
		CMxMTThreadTask( threadPool, mathEngine, /*batchSize*/1,
			/*first*/centers.GetData(), clusterCount, featureCount,
			/*second*/currData, batchSize, /*result*/distances, clusterCount * batchSize ).ParallelRun();
		CVectorMultiplyThreadTask( threadPool, mathEngine, clusterCount * batchSize,
			distances, distances, minusTwo ).ParallelRun( );
		CVectorAddToMatrixRowsThreadTask( threadPool, mathEngine, /*batchSize*/1,
			distances, distances, clusterCount, batchSize, currSquaredData ).ParallelRun();
		mathEngine.AddVectorToMatrixColumns( distances, distances, clusterCount, batchSize, squaredCenters ); // no threads
		mathEngine.FindMinValueInColumns( distances, clusterCount, batchSize, currClosesDist, currLabels ); // no threads

		batchStart += batchSize;
		currData += batchSize * featureCount;
		currSquaredData += batchSize;
		currClosesDist += batchSize;
		currLabels += batchSize;
	}
}

// Assigns every point to its closest cluster
double CKMeansClustering::assignClosest( const CDnnBlob& data, const CDnnBlob& squaredData, const CDnnBlob& weight,
	const CDnnBlob& centers, CDnnBlob& labels )
{
	IMathEngine& mathEngine = data.GetMathEngine();
	const int vectorCount = data.GetObjectCount();
	CFloatHandleStackVar stackBuff( mathEngine, vectorCount + 1 );
	CFloatHandle closestDist = stackBuff.GetHandle();
	CFloatHandle totalDist = stackBuff.GetHandle() + vectorCount;
	CIntHandle labelsHandle = labels.GetData<int>();
	calcClosestDistances( *GetThreadPool(), data, squaredData, centers, closestDist, labelsHandle );
	CVectorEltwiseMultiplyThreadTask( *GetThreadPool(), mathEngine, vectorCount, closestDist,
		weight.GetData(), closestDist ).ParallelRun();
	mathEngine.VectorSum( closestDist, vectorCount, totalDist ); // no threads
	const double result = static_cast<double>( totalDist.GetValue() );
	return result;
}

// Recalculates cluster centers
void CKMeansClustering::recalcCenters( const CDnnBlob& data, const CDnnBlob& weight, const CDnnBlob& labels,
	CDnnBlob& centers, CDnnBlob& sizes )
{
	IMathEngine& mathEngine = data.GetMathEngine();
	const int clusterCount = params.InitialClustersCount;
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();

	CFloatHandleStackVar stackBuff( mathEngine, centers.GetDataSize() + 1 );
	CFloatHandle newCenter = stackBuff;
	mathEngine.LookupAndAddToTable( labels.GetData<int>(), vectorCount, 1, data.GetData(), // no threads
		data.GetObjectSize(), newCenter, clusterCount );
	mathEngine.LookupAndAddToTable( labels.GetData<int>(), vectorCount, 1, weight.GetData(), // no threads
		1, sizes.GetData(), clusterCount );

	CFloatHandle invertedSize = stackBuff + centers.GetDataSize();
	CDnnBlobBuffer<float> rawSizes( sizes, TDnnBlobBufferAccess::Write );
	for( int i = 0; i < clusterCount; ++i ) {
		// Ignore empty clusters
		if( rawSizes[i] > 0 ) {
			// Update centers for non-emtpy clusters
			invertedSize.SetValue( 1.f / rawSizes[i] );
			CVectorMultiplyThreadTask( *GetThreadPool(), mathEngine, featureCount, newCenter,
				centers.GetObjectData( i ), invertedSize ).ParallelRun();
		}
		newCenter += featureCount;
	}
	rawSizes.Close();
}

// Calculates clusters' variances
void CKMeansClustering::calcClusterVariances( const CDnnBlob& data, const CDnnBlob& labels,
	const CDnnBlob& centers, const CDnnBlob& sizes, CDnnBlob& variances )
{
	IMathEngine& mathEngine = data.GetMathEngine();
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();
	const int clusterCount = sizes.GetDataSize();

	// 1 / *cluster size*
	CPtr<CDnnBlob> sizeInv = CDnnBlob::CreateVector( mathEngine, CT_Float, clusterCount ); // no threads
	{
		CDnnBlobBuffer<float> sizeBuff( const_cast<CDnnBlob&>( sizes ), TDnnBlobBufferAccess::Read );
		CDnnBlobBuffer<float> sizeInvBuff( *sizeInv, TDnnBlobBufferAccess::Write );
		for( int i = 0; i < clusterCount; ++i ) {
			sizeInvBuff[i] = sizeBuff[i] > 0 ? 1.f / sizeBuff[i] : 1.f;
		}
	}

	CFloatHandleStackVar stackBuff( mathEngine, vectorCount * featureCount + clusterCount * featureCount + 1 );
	{
		// Calculate sum of squares of objects in each cluster
		CFloatHandle squaredData = stackBuff.GetHandle();
		CFloatHandle sumOfSquares = stackBuff.GetHandle() + vectorCount * featureCount;
		CFloatHandle one = stackBuff.GetHandle() + vectorCount * featureCount + clusterCount * featureCount;
		CVectorEltwiseMultiplyThreadTask( *threadPool, mathEngine, data.GetDataSize(), data.GetData(), data.GetData(), squaredData ).ParallelRun();
		CVectorFillThreadTask( *threadPool, mathEngine, clusterCount * featureCount, sumOfSquares, /*value*/0.f ).ParallelRun();
		one.SetValue( 1.f );
		CLookupDimension dim;
		dim.VectorCount = params.InitialClustersCount;
		dim.VectorSize = featureCount;
		variances.Clear();
		CVectorMultichannelLookupAndAddToTableThreadTask( *GetThreadPool(), mathEngine, vectorCount, /*chennels*/1,
			labels.GetData<int>(), &sumOfSquares, &dim, 1, one, squaredData, featureCount ).ParallelRun();
		// Divide sum of squares by cluster size
		CDiagMxMThreadTask( *GetThreadPool(), mathEngine, sizeInv->GetData(), clusterCount, sumOfSquares, featureCount,
			variances.GetData(), variances.GetDataSize() ).ParallelRun();
	}

	{
		// Calculate squared centers
		CFloatHandle squaredMean = stackBuff;
		CVectorEltwiseMultiplyThreadTask( *GetThreadPool(), mathEngine, clusterCount * featureCount,
			centers.GetData(), centers.GetData(), squaredMean ).ParallelRun();
		// Subtract squares from average in order to get variance
		CVectorSubThreadTask( *GetThreadPool(), mathEngine, clusterCount * featureCount,
			variances.GetData(), squaredMean, variances.GetData() ).ParallelRun();
	}
}

} // namespace NeoML
