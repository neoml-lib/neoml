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

struct CInertia final {
	explicit CInertia( int threadCount ) { localInertia.Add( 0., threadCount ); }
	double& Get( int threadIndex ) { return localInertia[threadIndex]; }
	double GetSum() const;

private:
	CArray<double> localInertia{};
};

double CInertia::GetSum() const
{
	double inertia = 0;
	for( int i = 0; i < localInertia.Size(); ++i ) {
		inertia += localInertia[i];
	}
	return inertia;
}

//-------------------------------------------------------------------------------------------------------------

// Task which processes a set of elements on a single thread
struct CKMeansClustering::IThreadTask {
	virtual ~IThreadTask() {}
	virtual int Size() const { return Matrix ? Matrix->Height : VectorSize; }

	int ThreadCount() const { return Owner.params.ThreadCount; }
	void CallRun( int threadIndex, int align = 1 );

	const CFloatMatrixDesc* const Matrix;
protected:
	IThreadTask( const CKMeansClustering& owner, const CFloatMatrixDesc* matrix ) :
		Matrix( matrix ),
		VectorSize( 0 ),
		Owner( owner )
	{}
	IThreadTask( const CKMeansClustering& owner, int size ) :
		Matrix( nullptr ),
		VectorSize( size ),
		Owner( owner )
	{}

	virtual void Run( int threadIndex, int index, int count );
	virtual void RunOnElement( int threadIndex, int index ) = 0;

	const int VectorSize;
	const CKMeansClustering& Owner;
};

void CKMeansClustering::IThreadTask::CallRun( int threadIndex, int align )
{
	int index = 0;
	int count = 0;
	if( GetTaskIndexAndCount( ThreadCount(), threadIndex, Size(), align, index, count ) ) {
		Run( threadIndex, index, count );
	}
}

void CKMeansClustering::IThreadTask::Run( int threadIndex, int index, int count )
{
	const int last = index + count;
	for( int i = index; i < last; ++i ) {
		RunOnElement( threadIndex, i );
	}
}

//-------------------------------------------------------------------------------------------------------------

// Finds the nearest cluster for the element
struct CKMeansClustering::CClassifyAllThreadTask final : public CKMeansClustering::IThreadTask {
	CClassifyAllThreadTask( const CKMeansClustering& owner, const CFloatMatrixDesc& matrix ) :
		IThreadTask( owner, &matrix ),
		Inertia( ThreadCount() )
	{ DataCluster.SetBufferSize( Size() ); }

	CArray<int> DataCluster{}; // the cluster for this element
	CInertia Inertia;
protected:
	void RunOnElement( int threadIndex, int index ) override;
};

void CKMeansClustering::CClassifyAllThreadTask::RunOnElement( int threadIndex, int dataIndex )
{
	const auto distanceFunc = Owner.params.DistanceFunc;
	double bestDistance = DBL_MAX;
	int res = NotFound;

	for( int i = 0; i < Owner.clusters.Size(); ++i ) {
		CFloatVectorDesc desc;
		Matrix->GetRow( dataIndex, desc );
		const double distance = Owner.clusters[i]->CalcDistance( desc, distanceFunc );
		if( distance < bestDistance ) {
			bestDistance = distance;
			res = i;
		}
	}

	NeoAssert( res != NotFound );
	Inertia.Get( threadIndex ) += bestDistance;
	DataCluster[dataIndex] = res;
}

//-------------------------------------------------------------------------------------------------------------

struct CKMeansClustering::CAssignVectorsThreadTask final : public CKMeansClustering::IThreadTask {
	CAssignVectorsThreadTask( const CKMeansClustering& owner, const CFloatMatrixDesc& matrix ) :
		IThreadTask( owner, &matrix )
	{}

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

protected:
	void RunOnElement( int threadIndex, int index ) override;
	// Checks if operation can be omitted
	// If it's true element 'id' can't be reassigned to the cluster 'clusterToProcess'
	// and all of the calculations related to this case can be skipped
	bool isPruned( int clusterToProcess, int id ) const;
};

bool CKMeansClustering::CAssignVectorsThreadTask::isPruned( int clusterToProcess, int id ) const
{
	return ( Assignments[id] == clusterToProcess ) ||
		( UpperBounds[id] <= LowerBounds( clusterToProcess, id ) ) ||
		( UpperBounds[id] <= 0.5 * ClusterDists( Assignments[id], clusterToProcess ) );
}

void CKMeansClustering::CAssignVectorsThreadTask::RunOnElement( int /*threadIndex*/, int index )
{
	const auto distanceFunc = Owner.params.DistanceFunc;
	if( UpperBounds[index] > ClosestClusterDist[Assignments[index]] ) {
		bool recalculate = true;
		for( int c = 0; c < Owner.clusters.Size(); ++c ) {
			if( isPruned( c, index ) ) {
				continue;
			}
			float dist = UpperBounds[index];
			if( recalculate ) {
				recalculate = false;
				auto& cluster = *Owner.clusters[Assignments[index]];
				auto rowDesc = Matrix->GetRow( index );
				dist = static_cast<float>( sqrt( cluster.CalcDistance( rowDesc, distanceFunc ) ) );
				LowerBounds( Assignments[index], index ) = dist;
				UpperBounds[index] = dist;
			}
			if( dist > LowerBounds( c, index ) || dist > 0.5 * ClusterDists( Assignments[index], c ) ) {
				auto& cluster = *Owner.clusters[c];
				auto rowDesc = Matrix->GetRow( index );
				const float pointDist = static_cast<float>( sqrt( cluster.CalcDistance( rowDesc, distanceFunc ) ) );
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

struct CKMeansClustering::CUpdateULBoundsThreadTask final : public CKMeansClustering::IThreadTask {
	CUpdateULBoundsThreadTask( const CKMeansClustering& owner, const CFloatMatrixDesc& matrix,
			CAssignVectorsThreadTask& assigns ) :
		IThreadTask( owner, &matrix ),
		Inertia( ThreadCount() ),
		Assigns( assigns )
	{}

	CInertia Inertia;
protected:
	void RunOnElement( int threadIndex, int index ) override;

	CAssignVectorsThreadTask& Assigns;
};

void CKMeansClustering::CUpdateULBoundsThreadTask::RunOnElement( int threadIndex, int index )
{
	for( int c = 0; c < Owner.clusters.Size(); ++c ) {
		Assigns.LowerBounds( c, index ) = max( Assigns.LowerBounds( c, index ) - Assigns.MoveDistance[c], 0.f );
	}
	Assigns.UpperBounds[index] += Assigns.MoveDistance[Assigns.Assignments[index]];
	Inertia.Get( threadIndex ) += Owner.clusters[Assigns.Assignments[index]]->CalcDistance(
		Matrix->GetRow( index ), Owner.params.DistanceFunc );
}

//-------------------------------------------------------------------------------------------------------------

namespace {
struct CThreadTaskMxMT final {
	CThreadTaskMxMT( int threadCount, IMathEngine& mathEngine,
			const CConstFloatHandle& first, int firstHeight, int firstWidth,
			const CConstFloatHandle& second, int secondHeight,
			const CFloatHandle& result ) :
		ThreadCount( threadCount ),
		MathEngine( mathEngine ),
		First( first ),
		FirstHeight( firstHeight ),
		FirstWidth( firstWidth ),
		Second( second ),
		SecondHeight( secondHeight ),
		Result( result )
	{}

	void CallRun( int threadIndex );
protected:
	const int ThreadCount;
	IMathEngine& MathEngine;
	const CConstFloatHandle& First;
	const int FirstHeight;
	const int FirstWidth;
	const CConstFloatHandle& Second;
	const int SecondHeight;
	const CFloatHandle& Result;
};

void CThreadTaskMxMT::CallRun( int threadIndex )
{
	int firstHeightStart;
	int firstHeightCount;
	int secondHeightStart;
	int secondHeightCount;
	if( OmpGetTaskIndexAndCount2D( FirstHeight, /*alignX*/1, SecondHeight, FloatAlignment,
		firstHeightStart, firstHeightCount, secondHeightStart, secondHeightCount,
		ThreadCount, threadIndex ) )
	{
		const CConstFloatHandle& first = First + firstHeightStart * FirstWidth;
		const CConstFloatHandle& second = Second + secondHeightStart * FirstWidth;
		const CFloatHandle& result = Result + firstHeightStart * SecondHeight + secondHeightStart;

		MathEngine.MultiplyMatrixByTransposedMatrix(
			first, firstHeightCount, FirstWidth, FirstWidth/*RowSize*/,
			second, secondHeightCount, FirstWidth/*RowSize*/,
			result, SecondHeight/*RowSize*/, 0/*resBufferSize*/ );
	}
}
} // namespace

//-------------------------------------------------------------------------------------------------------------

static inline bool isThreadTaskRelevant( int64_t operationCount )
{
	constexpr int64_t MinOmpOperationCount = 32768;
	return operationCount >= MinOmpOperationCount;
}

static void matrixMultiplyByTransposed( const CKMeansClustering& owner, IMathEngine& mathEngine, int batchSize,
	const CConstFloatHandle& first, int firstHeight, int firstWidth, const CConstFloatHandle& second, int secondHeight,
	const CFloatHandle& result, int resultBufferSize )
{
	NeoAssert( batchSize == 1 );
	NeoAssert( resultBufferSize >= firstHeight * secondHeight );

	if( !isThreadTaskRelevant( firstWidth * firstHeight * secondHeight ) )
	{
		mathEngine.MultiplyMatrixByTransposedMatrix(
			first, firstHeight, firstWidth, firstWidth/*firstRowSize*/,
			second, secondHeight, firstWidth/*secondRowSize*/,
			result, secondHeight/*resultRowSize*/, 0/*resultBufferSize*/ );
		return;
	}

	CThreadTaskMxMT task( owner.GetThreadPool()->Size(), mathEngine,
		first, firstHeight, firstWidth, second, secondHeight, result );

	NEOML_NUM_THREADS( *owner.GetThreadPool(), &task, []( int threadIndex, void* ptr ) {
		( ( CThreadTaskMxMT* )ptr )->CallRun( threadIndex );
	} );
}

//-------------------------------------------------------------------------------------------------------------

CKMeansClustering::CKMeansClustering( const CArray<CClusterCenter>& _clusters, const CParam& _params ) :
	CKMeansClustering( _params )
{
	NeoAssert( !_clusters.IsEmpty() );
	NeoAssert( _clusters.Size() == params.InitialClustersCount );

	_clusters.CopyTo( initialClusterCenters );
}

CKMeansClustering::CKMeansClustering( const CParam& _params ) :
	params( _params ),
	threadPool( CreateThreadPool( params.ThreadCount ) )
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

	CClassifyAllThreadTask task( *this, matrix );
	for( int i = 0; i < params.MaxIterations; ++i ) {
		inertia = classifyAllData( task );

		if( log != 0 ) {
			*log << "\n[Step " << i << "]\nData classification result:\n";
			for( int j = 0; j < clusters.Size(); ++j ) {
				*log << "Cluster " << j << ": \n";
				*log << *clusters[j];
			}
		}

		CArray<CClusterCenter> oldCenters;
		storeClusterCenters( oldCenters );
		if( !updateClusters( matrix, weights, task.DataCluster, oldCenters ) ) {
			// Cluster centers stay the same, no need to continue
			success = true;
			break;
		}
	}
	return success;
}

// Distributes all elements over the existing clusters
double CKMeansClustering::classifyAllData( CClassifyAllThreadTask& task )
{
	task.DataCluster.SetSize( task.Size() ); // Each element is assigned to the nearest cluster

	NEOML_NUM_THREADS( *threadPool, &task, []( int threadIndex, void* ptr ) {
		( ( IThreadTask* )ptr )->CallRun( threadIndex );
	} );

	return task.Inertia.GetSum();
}

void CKMeansClustering::storeClusterCenters( CArray<CClusterCenter>& result )
{
	result.SetBufferSize( clusters.Size() );
	for( int i = 0; i < clusters.Size(); ++i ) {
		result.Add( clusters[i]->GetCenter() );
	}
}

// Updates the clusters and returns true if the clusters were changed, false if they stayed the same
bool CKMeansClustering::updateClusters( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
	const CArray<int>& dataCluster, const CArray<CClusterCenter>& oldCenters )
{
	// Store the old cluster centers
	for( int i = 0; i < clusters.Size(); ++i ) {
		clusters[i]->Reset();
	}

	// Update the cluster contents
	for( int i = 0; i < dataCluster.Size(); ++i ) {
		CFloatVectorDesc desc;
		matrix.GetRow( i, desc );
		clusters[dataCluster[i]]->Add( i, desc, weights[i] );
	}

	// Update the cluster centers
	for( int i = 0; i < clusters.Size(); ++i ) {
		if( clusters[i]->GetElementsCount() > 0 ) {
			clusters[i]->RecalcCenter();
		}
	}

	// Compare the new cluster centers with the old
	for( int i = 0; i < clusters.Size(); ++i ) {
		if( oldCenters[i].Mean != clusters[i]->GetCenter().Mean ) {
			return true;
		}
	}
	return false;
}

bool CKMeansClustering::elkanClusterization( const CFloatMatrixDesc& matrix, const CArray<double>& weights, double& inertia )
{
	// Metric must support triangle inequality
	NeoAssert( params.DistanceFunc == DF_Euclid );

	CAssignVectorsThreadTask assignTask( *this, matrix );
	initializeElkanStatistics( assignTask );

	double lastResidual = DBL_MAX;
	for( int i = 0; i < params.MaxIterations; ++i ) {
		// Calculaate pairwise and closest cluster distances
		computeClustersDists( assignTask.ClusterDists, assignTask.ClosestClusterDist );
		// Reassign vectors
		assignVectors( assignTask );
		// Recalculate centers
		CArray<CClusterCenter> oldCenters;
		storeClusterCenters( oldCenters );
		updateClusters( matrix, weights, assignTask.Assignments, oldCenters );
		// Update move distances
		updateMoveDistance( oldCenters, assignTask.MoveDistance );
		// Update bounds based on move distance
		CUpdateULBoundsThreadTask updateTask( *this, matrix, assignTask );
		inertia = updateUpperAndLowerBounds( updateTask );
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

// Initializes all required statistics for Elkan algorithm
void CKMeansClustering::initializeElkanStatistics( CAssignVectorsThreadTask& task )
{
	// initialize mapping between elements and cluster index
	task.Assignments.DeleteAll();
	task.Assignments.Add( 0, task.Matrix->Height );

	// initialize bounds
	task.UpperBounds.DeleteAll();
	task.UpperBounds.Add( FLT_MAX, task.Matrix->Height );
	task.LowerBounds.SetSize( params.InitialClustersCount, task.Matrix->Height );
	task.LowerBounds.Set( 0.f );

	// initialize pairwise cluster distances
	task.ClusterDists.SetSize( params.InitialClustersCount, params.InitialClustersCount );
	task.ClusterDists.Set( FLT_MAX );

	// initialize closest cluster distances
	task.ClosestClusterDist.DeleteAll();
	task.ClosestClusterDist.Add( FLT_MAX, params.InitialClustersCount );

	// initialize move distances
	task.MoveDistance.DeleteAll();
	task.MoveDistance.Add( 0, params.InitialClustersCount );
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

void CKMeansClustering::assignVectors( CAssignVectorsThreadTask& task ) const
{
	NeoAssert( task.Assignments.Size() == task.Matrix->Height );
	NeoAssert( task.LowerBounds.SizeX() == params.InitialClustersCount );
	NeoAssert( task.UpperBounds.Size() == task.Assignments.Size() );

	NEOML_NUM_THREADS( *threadPool, &task, []( int threadIndex, void* ptr )  {
		( ( IThreadTask* )ptr )->CallRun( threadIndex );
	} );
}

void CKMeansClustering::updateMoveDistance( const CArray<CClusterCenter>& oldCenters, CArray<float>& moveDistance ) const
{
	for( int i = 0; i < clusters.Size(); ++i ) {
		const double moveNorm = sqrt( clusters[i]->CalcDistance( oldCenters[i].Mean, params.DistanceFunc ) );
		moveDistance[i] = static_cast<float>( moveNorm );
	}
}

double CKMeansClustering::updateUpperAndLowerBounds( CUpdateULBoundsThreadTask& task ) const
{
	NEOML_NUM_THREADS( *threadPool, &task, []( int threadIndex, void* ptr ) {
		( ( IThreadTask* )ptr )->CallRun( threadIndex );
	} );

	return task.Inertia.GetSum();
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
			mathEngine.VectorCopy( centers.GetObjectData( i ), data.GetObjectData( pos ), featureCount ); // TODO! threads
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
			mathEngine.VectorCopy( centers.GetObjectData( i ), data.GetObjectData( perm[i] ), featureCount ); // TODO! threads
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
	mathEngine.VectorCopy( centers.GetData(), data.GetObjectData( firstChoice ), data.GetObjectSize() ); // TODO! threads

	CFloatHandleStackVar stackBuff( mathEngine, vectorCount + 1 );
	CFloatHandle sumBlob = stackBuff;
	CFloatHandle currDists = stackBuff + 1;

	CHashTable<int> usedVectors;
	CPtr<CDnnBlob> prevDists = CDnnBlob::CreateVector( mathEngine, CT_Float, vectorCount ); // no threads
	mathEngine.MatrixRowsToVectorSquaredL2Distance( data.GetData(), vectorCount, featureCount, // TODO! threads
		centers.GetData(), prevDists->GetData() );
	for( int k = 1; k < params.InitialClustersCount; ++k ) {
		CConstFloatHandle currentVector = centers.GetObjectData( k - 1 );
		mathEngine.MatrixRowsToVectorSquaredL2Distance( data.GetData(), vectorCount, featureCount, // TODO! threads
			currentVector, currDists );
		mathEngine.VectorEltwiseMin( currDists, prevDists->GetData(), prevDists->GetData(), prevDists->GetDataSize() ); // TODO! threads
		mathEngine.VectorSum( prevDists->GetData(), prevDists->GetDataSize(), sumBlob ); // TODO! threads

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
		mathEngine.VectorCopy( centers.GetObjectData( k ), data.GetObjectData( nextCoice ), featureCount ); // TODO! threads
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
	mathEngine.RowMultiplyMatrixByMatrix( data.GetData(), data.GetData(), data.GetObjectCount(), // TODO! threads
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
static void calcClosestDistances( const CKMeansClustering& owner, const CDnnBlob& data, const CDnnBlob& squaredData,
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
	mathEngine.RowMultiplyMatrixByMatrix( centers.GetData(), centers.GetData(), clusterCount, featureCount, squaredCenters ); // TODO! threads

	int batchStart = 0;
	CConstFloatHandle currData = data.GetData();
	CConstFloatHandle currSquaredData = squaredData.GetData();
	CFloatHandle currClosesDist = closestDist;
	CIntHandle currLabels = labels;
	while( batchStart < vectorCount ) {
		batchSize = min( batchSize, vectorCount - batchStart );

		// (a - b)^2 = a^2 + b^2 - 2*a*b
		matrixMultiplyByTransposed( owner, mathEngine, /*batchSize*/1, /*first*/centers.GetData(), clusterCount, featureCount,
			/*second*/currData, batchSize, /*result*/distances, clusterCount * batchSize );
		mathEngine.VectorMultiply( distances, distances, clusterCount * batchSize, minusTwo ); // TODO! threads
		mathEngine.AddVectorToMatrixRows( 1, distances, distances, clusterCount, batchSize, currSquaredData ); // TODO! threads
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
	calcClosestDistances( *this, data, squaredData, centers, closestDist, labelsHandle );
	mathEngine.VectorEltwiseMultiply( closestDist, weight.GetData(), closestDist, vectorCount );
	mathEngine.VectorSum( closestDist, vectorCount, totalDist );
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
			mathEngine.VectorMultiply( newCenter, centers.GetObjectData( i ), // TODO! threads
				featureCount, invertedSize );
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
	CPtr<CDnnBlob> sizeInv = CDnnBlob::CreateVector( mathEngine, CT_Float, clusterCount );
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
		mathEngine.VectorEltwiseMultiply( data.GetData(), data.GetData(), squaredData, // TODO! threads
			data.GetDataSize() );
		mathEngine.VectorFill( sumOfSquares, 0, clusterCount * featureCount ); // TODO! threads
		one.SetValue( 1.f );
		CLookupDimension dim;
		dim.VectorCount = params.InitialClustersCount;
		dim.VectorSize = featureCount;
		variances.Clear();
		mathEngine.VectorMultichannelLookupAndAddToTable( vectorCount, 1, labels.GetData<int>(), // TODO! threads
			&sumOfSquares, &dim, 1, one, squaredData, featureCount );
		// Divide sum of squares by cluster size
		mathEngine.MultiplyDiagMatrixByMatrix( sizeInv->GetData(), clusterCount, sumOfSquares, featureCount, // TODO! threads
			variances.GetData(), variances.GetDataSize() );
	}

	{
		// Calculate squared centers
		CFloatHandle squaredMean = stackBuff;
		mathEngine.VectorEltwiseMultiply( centers.GetData(), centers.GetData(), // TODO! threads
			squaredMean, clusterCount * featureCount );
		// Subtract squares from average in order to get variance
		mathEngine.VectorSub( variances.GetData(), squaredMean, variances.GetData(), clusterCount * featureCount ); // TODO! threads
	}
}

} // namespace NeoML
