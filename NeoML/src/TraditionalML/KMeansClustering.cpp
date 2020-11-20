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
#include <NeoML/Random.h>
#include <float.h>

namespace NeoML {

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

bool CKMeansClustering::Clusterize( IClusteringData* input, CClusteringResult& result )
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
	dataCluster.DeleteAll();
	const int vectorCount = matrix.Height;
	for( int i = 0; i < vectorCount; i++ ) {
		dataCluster.Add( findNearestCluster( matrix, i ) );
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
	for( int i = 0; i < matrix.Height; i++ ) {
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
	double inertia = 0;
	for( int j = 0; j < matrix.Height; j++ ) {
		for( int c = 0; c < clusters.Size(); c++ ) {
			lowerBounds( c, j ) = max( lowerBounds( c, j ) - moveDistance[c], 0.f );
		}
		upperBounds[j] += moveDistance[assignments[j]];
		inertia += clusters[assignments[j]]->CalcDistance( matrix.GetRow( j ), params.DistanceFunc );
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

} // namespace NeoML
