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

#include <NeoML/TraditionalML/KMeansClustering.h>
#include <NeoML/TraditionalML/CommonCluster.h>
#include <NeoML/TraditionalML/VariableMatrix.h>
#include <NeoML/Random.h>
#include <NeoMathEngine/OpenMP.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/DnnBlob.h>
#include <NeoML/Dnn/Dnn.h>
#include <float.h>
#include <memory>

namespace NeoML {

static CPtr<CDnnBlob> createDataBlob( IMathEngine& mathEngine, IDenseClusteringData* data )
{
	const int vectorCount = data->GetVectorCount();
	const int featureCount = data->GetFeaturesCount();
	CPtr<CDnnBlob> result = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, vectorCount, featureCount );
	result->CopyFrom( data->GetMatrix().Values );
	return result;
}

static CPtr<CDnnBlob> createWeightBlob( IMathEngine& mathEngine, IDenseClusteringData* data )
{
	const int vectorCount = data->GetVectorCount();
	CPtr<CDnnBlob> weight = CDnnBlob::CreateVector( mathEngine, CT_Float, vectorCount );
	float* buffer = weight->GetBuffer<float>( 0, vectorCount );
	for( int vectorIndex = 0; vectorIndex < vectorCount; ++vectorIndex ) {
		buffer[vectorIndex] = static_cast<float>( data->GetVectorWeight( vectorIndex ) );
	}
	weight->ReleaseBuffer( buffer, true );
	return weight;
}

CKMeansClustering::CKMeansClustering( const CArray<CClusterCenter>& _clusters, const CParam& _params ) :
	params( _params ),
	log( 0 )
{
	NeoAssert( !_clusters.IsEmpty() );
	NeoAssert( _clusters.Size() == params.InitialClustersCount );

	_clusters.CopyTo( initialClusterCenters );
}

CKMeansClustering::CKMeansClustering( const CParam& _params ) :
	params( _params ),
	log( 0 )
{
}

bool CKMeansClustering::Clusterize( ISparseClusteringData* input, CClusteringResult& result )
{
	NeoAssert( input != 0 );

	CSparseFloatMatrixDesc matrix = input->GetMatrix();
	NeoAssert( matrix.Height == input->GetVectorCount() );
	NeoAssert( matrix.Width == input->GetFeaturesCount() );

	CArray<double> weights;
	for( int i = 0; i < input->GetVectorCount(); i++ ) {
		weights.Add( input->GetVectorWeight( i ) );
	}

	if( log != 0 ) {
		*log << "\nK-means clustering started:\n";
	}

	selectInitialClusters( matrix );

	if( log != 0 ) {
		*log << "Initial clusters:\n";

		for( int i = 0; i < clusters.Size(); i++ ) {
			*log << *clusters[i] << "\n";
		}
	}

	bool success = clusterize( matrix, weights );

	result.ClusterCount = clusters.Size();
	result.Data.SetSize( matrix.Height );
	result.Clusters.SetBufferSize( clusters.Size() );

	for( int i = 0; i < clusters.Size(); i++ ) {
		CArray<int> elements;
		clusters[i]->GetAllElements( elements );
		for(int j = 0; j < elements.Size(); j++ ) {
			result.Data[elements[j]]=i;
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

bool CKMeansClustering::Clusterize( IDenseClusteringData* rawData, CClusteringResult& result )
{
	NeoAssert( params.DistanceFunc == DF_Euclid );
	NeoAssert( params.Algo == KMA_Lloyd );
	NeoAssert( rawData->GetVectorCount() > params.InitialClustersCount );
	const int vectorCount = rawData->GetVectorCount();
	const int featureCount = rawData->GetFeaturesCount();
	const int clusterCount = params.InitialClustersCount;

	if( log != 0 ) {
		*log << L"\nK-means clustering started:\n";
	}

	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( params.ThreadCount, 0 ) );

	CPtr<CDnnBlob> data = createDataBlob( *mathEngine, rawData );
	CPtr<CDnnBlob> weight = createWeightBlob( *mathEngine, rawData );
	CPtr<CDnnBlob> centers = CDnnBlob::CreateDataBlob( *mathEngine, CT_Float, 1, clusterCount, featureCount );

	selectInitialClusters( *data, *centers );

	if( log != 0 ) {
		*log << L"Initial clusters:\n";

		for( int i = 0; i < clusters.Size(); i++ ) {
			*log << *clusters[i] << L"\n";
		}
	}

	bool success = false;

	CPtr<CDnnBlob> sizes = CDnnBlob::CreateVector( *mathEngine, CT_Float, clusterCount );
	CPtr<CDnnBlob> labels = CDnnBlob::CreateVector( *mathEngine, CT_Int, vectorCount );

	static_assert( KMA_Count == 2, "KMA_Count != 2" );
	switch( params.Algo ) {
		case KMA_Lloyd:
			success = lloydBlobClusterization( *data, *weight, *centers, *sizes, *labels );
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

	CPtr<CDnnBlob> variances = CDnnBlob::CreateDataBlob( *mathEngine, CT_Float, 1, clusterCount, featureCount );
	calcClusterVariances( *data, *labels, *centers, *sizes, *variances );

	CConstFloatHandle rawCentersPtr = centers->GetData();
	CConstFloatHandle rawVariancesPtr = variances->GetData();

	result.Clusters.SetBufferSize( result.ClusterCount );

	for( int i = 0; i < clusterCount; i++ ) {
		CFloatVector center( featureCount );
		CFloatVector variance( featureCount );

		mathEngine->DataExchangeTyped( center.CopyOnWrite(), rawCentersPtr, featureCount );
		mathEngine->DataExchangeTyped( variance.CopyOnWrite(), rawVariancesPtr, featureCount );

		CClusterCenter& currentCenter = result.Clusters.Append();
		currentCenter.Mean = center;
		currentCenter.Disp = variance;
		currentCenter.Norm = DotProduct( currentCenter.Mean, currentCenter.Mean );

		rawCentersPtr += featureCount;
		rawVariancesPtr += featureCount;
	}

	result.ClusterCount = result.Clusters.Size();
	assert( result.Clusters.Size() > 0 );
	return success;
}

// Selects the initial clusters
void CKMeansClustering::selectInitialClusters( const CSparseFloatMatrixDesc& matrix )
{
	if( !clusters.IsEmpty() ) {
		// The initial clusters have been set already
		return;
	}
	
	// If the initial cluster centers have been specified, create the clusters from that
	if( !initialClusterCenters.IsEmpty() ) {
		clusters.SetBufferSize( params.InitialClustersCount );
		for( int i = 0; i < initialClusterCenters.Size(); i++ ) {
			clusters.Add( FINE_DEBUG_NEW CCommonCluster( initialClusterCenters[i] ) );
		}
		return;
	}

	if( params.Initialization == KMI_Default ) {
		defaultInitialization( matrix );
	} else if( params.Initialization == KMI_KMeansPlusPlus ) {
		kMeansPlusPlusInitialization( matrix );
	} else {
		NeoAssert( false );
	}
}

void CKMeansClustering::defaultInitialization( const CSparseFloatMatrixDesc& matrix )
{
	const int vectorCount = matrix.Height;
	// If the cluster centers have not been specified, use some elements of the input data
	const int step = max( vectorCount / params.InitialClustersCount, 1 );
	NeoAssert( step > 0 );
	clusters.SetBufferSize( params.InitialClustersCount );
	for( int i = 0; i < params.InitialClustersCount; i++ ) {
		CSparseFloatVectorDesc desc;
		matrix.GetRow( ( i * step ) % vectorCount, desc );
		CFloatVector mean( matrix.Width, desc );
		clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( mean ) ) );
	}
}

void CKMeansClustering::kMeansPlusPlusInitialization( const CSparseFloatMatrixDesc& matrix )
{
	const int vectorCount = matrix.Height;
	NeoAssert( params.InitialClustersCount <= vectorCount );

	// Use random element as the first center
	CRandom random( 0xCEA );
	const int firstCenterIndex = random.UniformInt( 0, vectorCount - 1 );
	CFloatVector firstCenter( matrix.Width, matrix.GetRow( firstCenterIndex ) );
	clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( firstCenter ) ) );

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

bool CKMeansClustering::clusterize( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights )
{
	if( params.Algo == KMA_Lloyd ) {
		return lloydClusterization( matrix, weights );
	} else {
		return elkanClusterization( matrix, weights );
	}
}

bool CKMeansClustering::lloydClusterization( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights )
{
	CArray<int> dataCluster; // the cluster for this element
	dataCluster.SetBufferSize( matrix.Height );
	bool success = false;
	for( int i = 0; i < params.MaxIterations; i++ ) {
		classifyAllData( matrix, dataCluster );

		if( log != 0 ) {
			*log << "\n[Step " << i << "]\nData classification result:\n";
			for( int j = 0; j < clusters.Size(); j++ ) {
				*log << "Cluster " << j << ": \n";
				*log << *clusters[j];
			}
		}

		CArray<CClusterCenter> oldCenters;
		storeClusterCenters( oldCenters );
		if( !updateClusters( matrix, weights, dataCluster, oldCenters ) ) {
			// Cluster centers stay the same, no need to continue
			success = true;
			break;
		}
	}

	return success;
}

// Distributes all elements over the existing clusters
void CKMeansClustering::classifyAllData( const CSparseFloatMatrixDesc& matrix, CArray<int>& dataCluster )
{
	// Each element is assigned to the nearest cluster
	dataCluster.SetSize( matrix.Height );
	NEOML_OMP_NUM_THREADS( params.ThreadCount ) {
		int firstVector = 0;
		int vectorCount = 0;
		if( OmpGetTaskIndexAndCount( matrix.Height, firstVector, vectorCount ) ) {
			const int lastVector = firstVector + vectorCount;
			for( int i = firstVector; i < lastVector; i++ ) {
				dataCluster[i] = findNearestCluster( matrix, i );
			}
		}
	}
}

// Finds the nearest cluster for the element
int CKMeansClustering::findNearestCluster( const CSparseFloatMatrixDesc& matrix, int dataIndex ) const
{
	double bestDistance = DBL_MAX;
	int res = NotFound;
	
	for( int i = 0; i < clusters.Size(); i++ ) {
		CSparseFloatVectorDesc desc;
		matrix.GetRow( dataIndex, desc );
		const double distance = clusters[i]->CalcDistance( desc, params.DistanceFunc );
		if( distance < bestDistance ) {
			bestDistance = distance;
			res = i;
		}
	}

	NeoAssert( res != NotFound );
	return res;
}

void CKMeansClustering::storeClusterCenters( CArray<CClusterCenter>& result )
{
	result.SetBufferSize( clusters.Size() );
	for( int i = 0; i < clusters.Size(); ++i ) {
		result.Add( clusters[i]->GetCenter() );
	}
}

// Updates the clusters and returns true if the clusters were changed, false if they stayed the same
bool CKMeansClustering::updateClusters( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights,
	const CArray<int>& dataCluster, const CArray<CClusterCenter>& oldCenters )
{
	// Store the old cluster centers
	for( int i = 0; i < clusters.Size(); i++ ) {
		clusters[i]->Reset();
	}

	// Update the cluster contents
	for( int i = 0; i < dataCluster.Size(); i++ ) {
		CSparseFloatVectorDesc desc;
		matrix.GetRow( i, desc );
		clusters[dataCluster[i]]->Add( i, desc, weights[i] );
	}

	// Update the cluster centers
	for( int i = 0; i < clusters.Size(); i++ ) {
		if( clusters[i]->GetElementsCount() > 0 ) {
			clusters[i]->RecalcCenter();
		}
	}

	// Compare the new cluster centers with the old
	for( int i = 0; i < clusters.Size(); i++ ) {
		if( oldCenters[i].Mean != clusters[i]->GetCenter().Mean ) {
			return true;
		}
	}

	return false;
}

bool CKMeansClustering::elkanClusterization( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights )
{
	// Metric must support triangle inequality
	NeoAssert( params.DistanceFunc == DF_Euclid );

	// Distances bounds
	CArray<float> upperBounds; // upper bounds for every object (objectCount)
	CVariableMatrix<float> lowerBounds; // lower bounds for every object and every cluster (clusterCount x objectCount)
	// Distances between old and updated centers of each cluster (clusterCount)
	CArray<float> moveDistance;
	// Distances between clusters (clusterCount x clusteCount)
	CVariableMatrix<float> clusterDists;
	// Distance to the closest center of another cluster (clusterCount)
	CArray<float> closestClusterDist;
	// Element assignments (objectCount)
	CArray<int> assignments;

	initializeElkanStatistics( matrix, assignments, upperBounds, lowerBounds, clusterDists,
		closestClusterDist, moveDistance );

	double lastResidual = DBL_MAX;
	double inertia = DBL_MAX;
	for( int i = 0; i < params.MaxIterations; i++ ) {
		// Calculaate pairwise and closest cluster distances
		computeClustersDists( clusterDists, closestClusterDist );
		// Reassign vectors
		assignVectors( matrix, clusterDists, closestClusterDist, assignments,
			upperBounds, lowerBounds );
		// Recalculate centers
		CArray<CClusterCenter> oldCenters;
		storeClusterCenters( oldCenters );
		updateClusters( matrix, weights, assignments, oldCenters );
		// Update move distances
		updateMoveDistance( oldCenters, moveDistance );
		// Update bounds based on move distance
		inertia = updateUpperAndLowerBounds( matrix, moveDistance, assignments, upperBounds, lowerBounds );
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
void CKMeansClustering::initializeElkanStatistics( const CSparseFloatMatrixDesc& matrix,
	CArray<int>& assignments, CArray<float>& upperBounds, CVariableMatrix<float>& lowerBounds,
	CVariableMatrix<float>& clusterDists, CArray<float>& closestClusterDist, CArray<float>& moveDistance )
{
	// initialize mapping between elements and cluster index
	assignments.DeleteAll();
	assignments.Add( 0, matrix.Height );

	// initialize bounds
	upperBounds.DeleteAll();
	upperBounds.Add( FLT_MAX, matrix.Height );
	lowerBounds.SetSize( params.InitialClustersCount, matrix.Height );
	lowerBounds.Set( 0.f );

	// initialize pairwise cluster distances
	clusterDists.SetSize( params.InitialClustersCount, params.InitialClustersCount );
	clusterDists.Set( FLT_MAX );

	// initialize closest cluster distances
	closestClusterDist.DeleteAll();
	closestClusterDist.Add( FLT_MAX, params.InitialClustersCount );

	// initialize move distances
	moveDistance.DeleteAll();
	moveDistance.Add( 0, params.InitialClustersCount );
}

void CKMeansClustering::computeClustersDists( CVariableMatrix<float>& dists, CArray<float>& closestCluster ) const
{
	for( int i = 0; i < clusters.Size(); i++ ) {
		closestCluster[i] = FLT_MAX;
	}
	for( int i = 0; i < clusters.Size() - 1; i++ ) {
		dists( i, i ) = FLT_MAX;
		for( int j = i + 1; j < clusters.Size(); j++ ) {
			const float dist = static_cast<float>(
				sqrt( clusters[i]->CalcDistance( *clusters[j], params.DistanceFunc ) ) );
			dists( i, j ) = dist;
			dists( j, i ) = dist;
			closestCluster[i] = min( 0.5f * dist, closestCluster[i] );
			closestCluster[j] = min( 0.5f * dist, closestCluster[j] );
		}
	}
}

void CKMeansClustering::assignVectors( const CSparseFloatMatrixDesc& matrix, const CVariableMatrix<float>& clusterDists,
	const CArray<float>& closestClusterDist, CArray<int>& assignments, CArray<float>& upperBounds,
	CVariableMatrix<float>& lowerBounds ) const
{
	NeoAssert( assignments.Size() == matrix.Height );
	NeoAssert( lowerBounds.SizeX() == params.InitialClustersCount );
	NeoAssert( upperBounds.Size() == assignments.Size() );
	NEOML_OMP_NUM_THREADS( params.ThreadCount ) {
		int firstVector = 0;
		int vectorCount = 0;
		if( OmpGetTaskIndexAndCount( matrix.Height, firstVector, vectorCount ) ) {
			const int lastVector = firstVector + vectorCount;
			for( int i = firstVector; i < lastVector; i++ ) {
				bool mustRecalculate = true;
				if( upperBounds[i] <= closestClusterDist[assignments[i]] ) {
					continue;
				} else {
					for( int c = 0; c < clusters.Size(); c++ ) {
						if( isPruned( upperBounds, lowerBounds, clusterDists, assignments[i], c, i ) ) {
							continue;
						}
						float dist = upperBounds[i];
						if( mustRecalculate ) {
							mustRecalculate = false;
							dist = static_cast<float>(
								sqrt( clusters[assignments[i]]->CalcDistance( matrix.GetRow( i ),
									params.DistanceFunc ) ) );
							lowerBounds( assignments[i], i ) = dist;
							upperBounds[i] = dist;
						}
						if( dist > lowerBounds( c, i ) || dist > 0.5 * clusterDists( assignments[i], c ) ) {
							const float pointDist = static_cast<float>(
								sqrt( clusters[c]->CalcDistance( matrix.GetRow( i ),
									params.DistanceFunc ) ) );
							lowerBounds( c, i ) = pointDist;
							if( pointDist < dist ) {
								upperBounds[i] = pointDist;
								assignments[i] = c;
							}
						}
					}
				}
			}
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

double CKMeansClustering::updateUpperAndLowerBounds( const CSparseFloatMatrixDesc& matrix,
	const CArray<float>& moveDistance, const CArray<int>& assignments,
	CArray<float>& upperBounds, CVariableMatrix<float>& lowerBounds ) const
{
	CFastArray<double, 16> localInertia;
	localInertia.Add( 0., params.ThreadCount );

	NEOML_OMP_NUM_THREADS( params.ThreadCount ) {
		const int threadIndex = OmpGetThreadNum();
		int firstVector = 0;
		int vectorCount = 0;
		if( OmpGetTaskIndexAndCount( matrix.Height, firstVector, vectorCount ) ) {
			const int lastVector = firstVector + vectorCount;
			for( int j = firstVector; j < lastVector; j++ ) {
				for( int c = 0; c < clusters.Size(); c++ ) {
					lowerBounds( c, j ) = max( lowerBounds( c, j ) - moveDistance[c], 0.f );
				}
				upperBounds[j] += moveDistance[assignments[j]];
				localInertia[threadIndex] += clusters[assignments[j]]->CalcDistance( matrix.GetRow( j ), params.DistanceFunc );
			}
		}
	}

	double inertia = 0;
	for( int i = 0; i < localInertia.Size(); ++i ) {
		inertia += localInertia[i];
	}

	return inertia;
}

// Checks if operation can be omitted
// If it's true element 'id' can't be reassigned to the cluster 'clusterToProcess'
// and all of the calculations related to this case can be skipped
bool CKMeansClustering::isPruned( const CArray<float>& upperBounds, const CVariableMatrix<float>& lowerBounds,
	const CVariableMatrix<float>& clusterDists, int currentCluster, int clusterToProcess, int id ) const
{
	return ( currentCluster == clusterToProcess ) ||
		( upperBounds[id] <= lowerBounds( clusterToProcess, id ) ) ||
		( upperBounds[id] <= 0.5 * clusterDists( currentCluster, clusterToProcess ) );
}

// Selects initial centers from dense data
void CKMeansClustering::selectInitialClusters( const CDnnBlob& data, CDnnBlob& centers )
{
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();
	if( !initialClusterCenters.IsEmpty() ) {
		float* buffer = centers.GetBuffer<float>( 0, params.InitialClustersCount * featureCount );
		float* currPtr = buffer;
		for( int i = 0; i < params.InitialClustersCount; ++i ) {
			::memcpy( currPtr, initialClusterCenters[i].Mean.GetPtr(), featureCount * sizeof( float ) );
			currPtr += featureCount;
		}
		centers.ReleaseBuffer( buffer, true );
		return;
	}

	static_assert( KMI_Count == 2, "KMI_Count != 2" );
	switch( params.Initialization ) {
		case KMI_Default:
			defaultInitialization( data, centers );
			break;
		case KMI_KMeansPlusPlus:
			kMeansPlusPlusInitialization( data, centers );
			break;
		default:
			NeoAssert( false );
	}
}

// Selects initial centers by using default algo from dense data
void CKMeansClustering::defaultInitialization( const CDnnBlob& data, CDnnBlob& centers )
{
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();
	NeoAssert( centers.GetObjectCount() == params.InitialClustersCount );
	NeoAssert( centers.GetObjectSize() == featureCount );

	const int step = max( vectorCount / params.InitialClustersCount, 1 );
	NeoAssert( step > 0 );
	IMathEngine& mathEngine = data.GetMathEngine();

	for( int i = 0; i < params.InitialClustersCount; i++ ) {
		const int pos = ( i * step ) % vectorCount;
		mathEngine.VectorCopy( centers.GetObjectData( i ), data.GetObjectData( pos ), featureCount );
	}
}

// Selects initial centers by using K-Means++ algo from dense data
void CKMeansClustering::kMeansPlusPlusInitialization( const CDnnBlob& data, CDnnBlob& centers )
{
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();
	// Select random vector
	CRandom random( 0xCEA );
	const int firstChoice = random.UniformInt( 0, vectorCount - 1 );
	// Copy first vector
	IMathEngine& mathEngine = centers.GetMathEngine();
	mathEngine.VectorCopy( centers.GetData(), data.GetObjectData( firstChoice ), data.GetObjectSize() );

	CFloatHandleStackVar stackBuff( mathEngine, vectorCount + 1 );
	CFloatHandle sumBlob = stackBuff;
	CFloatHandle currDists = stackBuff + 1;

	CHashTable<int> usedVectors;
	CPtr<CDnnBlob> prevDists = CDnnBlob::CreateVector( mathEngine, CT_Float, vectorCount );
	mathEngine.MatrixRowsToVectorSquaredL2Distance( data.GetData(), vectorCount, featureCount,
		centers.GetData(), prevDists->GetData() );
	for( int k = 1; k < params.InitialClustersCount; k++ ) {
		CConstFloatHandle currentVector = centers.GetObjectData( k - 1 );
		mathEngine.MatrixRowsToVectorSquaredL2Distance( data.GetData(), vectorCount, featureCount,
			currentVector, currDists );
		mathEngine.VectorEltwiseMin( currDists, prevDists->GetData(), prevDists->GetData(), prevDists->GetDataSize() );
		mathEngine.VectorSum( prevDists->GetData(), prevDists->GetDataSize(), sumBlob );

		const double sumValue = static_cast<double>( sumBlob.GetValue() );
		const double scaledSum = random.Uniform( 0, 1 ) * sumValue;

		CArray<float> squares;
		squares.SetSize( prevDists->GetDataSize() );
		prevDists->CopyTo( squares.GetPtr() );
	
		double prefixSum = 0;
		int nextCoice = -1;
		for( int i = 0; i < vectorCount; i++ ) {
			prefixSum += squares[i];
			if( prefixSum > scaledSum ) {
				nextCoice = i;
				break;
			}
		}

		assert( nextCoice != -1 );
		assert( !usedVectors.Has( nextCoice ) );
		usedVectors.Add( nextCoice );
		mathEngine.VectorCopy( centers.GetObjectData( k ), data.GetObjectData( nextCoice ), featureCount );
	}
}

// Clusterizes dense data by using Lloyd algorithm
bool CKMeansClustering::lloydBlobClusterization( const CDnnBlob& data, const CDnnBlob& weight,
	CDnnBlob& centers, CDnnBlob& sizes, CDnnBlob& labels )
{
	double prevDist = FLT_MAX;
	double totalDist = 0;
	const float eps = 1e-3f;

	IMathEngine& mathEngine = data.GetMathEngine();
	// pre-calculate l2-norm of input data
	CPtr<CDnnBlob> squaredData = CDnnBlob::CreateVector( mathEngine, CT_Float, data.GetObjectCount() );
	mathEngine.RowMultiplyMatrixByMatrix( data.GetData(), data.GetData(), data.GetObjectCount(),
		data.GetObjectSize(), squaredData->GetData() );
	for( int iter = 0; iter < params.MaxIterations; iter++ ) {
		totalDist = assignClosest( data, *squaredData, weight, centers, labels );
		recalcCenters( data, weight, labels, centers, sizes );
		if( abs( prevDist - totalDist ) < eps ) {
			return true;
		}
		prevDist = totalDist;
	}

	return false;
}

static const int DistanceBufferSize = 2 * 1024 * 1024;

// Calculates distances between every point and the closest cluster
static void calcClosestDistances( const CDnnBlob& data, const CDnnBlob& squaredData, const CDnnBlob& centers,
	CFloatHandle& closestDist, CIntHandle& labels )
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
	mathEngine.RowMultiplyMatrixByMatrix( centers.GetData(), centers.GetData(), clusterCount, featureCount, squaredCenters );

	int batchStart = 0;
	CConstFloatHandle currData = data.GetData();
	CConstFloatHandle currSquaredData = squaredData.GetData();
	CFloatHandle currClosesDist = closestDist;
	CIntHandle currLabels = labels;
	while( batchStart < vectorCount ) {
		if( batchStart + batchSize > vectorCount ) {
			batchSize = vectorCount - batchStart;
		}

		// (a - b)^2 = a^2 + b^2 - 2*a*b
		mathEngine.MultiplyMatrixByTransposedMatrix( 1, centers.GetData(), clusterCount, featureCount, currData,
			batchSize, distances, clusterCount * batchSize );
		mathEngine.VectorMultiply( distances, distances, clusterCount * batchSize, minusTwo );
		mathEngine.AddVectorToMatrixRows( 1, distances, distances, clusterCount, batchSize, currSquaredData );
		mathEngine.AddVectorToMatrixColumns( distances, distances, clusterCount, batchSize, squaredCenters );
		mathEngine.FindMinValueInColumns( distances, clusterCount, batchSize, currClosesDist, currLabels );

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
	const int featureCount = data.GetObjectSize();
	const int clusterCount = centers.GetObjectCount();
	CFloatHandleStackVar stackBuff( mathEngine, vectorCount + 1 );
	CFloatHandle closestDist = stackBuff.GetHandle();
	CFloatHandle totalDist = stackBuff.GetHandle() + vectorCount;
	CIntHandle labelsHandle = labels.GetData<int>();
	calcClosestDistances( data, squaredData, centers, closestDist, labelsHandle, params.ResultChunkSize );
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
	mathEngine.LookupAndAddToTable( labels.GetData<int>(), vectorCount, 1, data.GetData(),
		data.GetObjectSize(), newCenter, clusterCount );
	mathEngine.LookupAndAddToTable( labels.GetData<int>(), vectorCount, 1, weight.GetData(),
		1, sizes.GetData(), clusterCount );

	CFloatHandle invertedSize = stackBuff + centers.GetDataSize();
	float* rawSizes = sizes.GetBuffer<float>( 0, clusterCount );
	for( int i = 0; i < clusterCount; i++ ) {
		// Ignore empty clusters
		if( rawSizes[i] > 0 ) {
			// Update centers for non-emtpy clusters
			invertedSize.SetValue( 1.f / rawSizes[i] );
			mathEngine.VectorMultiply( newCenter, centers.GetObjectData( i ),
				featureCount, invertedSize );
		}
		newCenter += featureCount;
	}
	sizes.ReleaseBuffer( rawSizes, false );
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
		float* sizeBuff = const_cast<CDnnBlob&>( sizes ).GetBuffer<float>( 0, clusterCount );
		float* sizeInvBuff = sizeInv->GetBuffer<float>( 0, clusterCount );
		for( int i = 0; i < clusterCount; ++i ) {
			sizeInvBuff[i] = sizeBuff[i] > 0 ? 1.f / sizeBuff[i] : 1.f;
		}
		sizeInv->ReleaseBuffer( sizeInvBuff, true );
		const_cast<CDnnBlob&>( sizes ).ReleaseBuffer( sizeBuff, false );
	}

	{
		// Calculate sum of squares of objects in each cluster
		CFloatHandleStackVar stackBuff( mathEngine, vectorCount * featureCount + clusterCount * featureCount + 1 );
		CFloatHandle squaredData = stackBuff.GetHandle();
		mathEngine.VectorEltwiseMultiply( data.GetData(), data.GetData(), squaredData,
			data.GetDataSize() );
		CFloatHandle sumOfSquares = stackBuff.GetHandle() + vectorCount * featureCount;
		mathEngine.VectorFill( sumOfSquares, 0, clusterCount * featureCount );
		CFloatHandle one = stackBuff.GetHandle() + vectorCount * featureCount + clusterCount * featureCount;
		one.SetValue( 1.f );
		CLookupDimension dim;
		dim.VectorCount = params.InitialClustersCount;
		dim.VectorSize = featureCount;
		variances.Clear();
		mathEngine.VectorMultichannelLookupAndAddToTable( vectorCount, 1, labels.GetData<int>(),
			&sumOfSquares, &dim, 1, one, squaredData, featureCount );
		// Divide sum of squares by cluster size
		mathEngine.MultiplyDiagMatrixByMatrix( sizeInv->GetData(), clusterCount, sumOfSquares, featureCount,
			variances.GetData(), variances.GetDataSize() );
	}

	{
		// Calculate squared centers
		CFloatHandleStackVar squaredMean( mathEngine, clusterCount * featureCount );
		mathEngine.VectorEltwiseMultiply( centers.GetData(), centers.GetData(),
			squaredMean, clusterCount * featureCount );
		// Subtract squares from average in order to get variance
		mathEngine.VectorSub( variances.GetData(), squaredMean, variances.GetData(), clusterCount * featureCount );
	}
}

} // namespace NeoML
