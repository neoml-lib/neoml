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

	CArray<int> dataCluster; // the cluster for this element
	dataCluster.SetBufferSize( input->GetVectorCount() );
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
		
		if( !updateClusters( matrix, weights, dataCluster ) ) {
			// Cluster centers stay the same, no need to continue
			success = true;
			break;
		}
	}

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

	// If the cluster centers have not been specified, use some elements of the input data
	const int vectorsCount = matrix.Height;
	const int step = max( vectorsCount / params.InitialClustersCount, 1 );
	NeoAssert( step > 0 );
	clusters.SetBufferSize( params.InitialClustersCount );
	for( int i = 0; i < params.InitialClustersCount; i++ ) {
		CSparseFloatVectorDesc desc;
		matrix.GetRow( ( i * step ) % vectorsCount, desc );
		CFloatVector mean( matrix.Width, desc );
		clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( mean ) ) );
	}
}

// Distributes all elements over the existing clusters
void CKMeansClustering::classifyAllData( const CSparseFloatMatrixDesc& matrix, CArray<int>& dataCluster )
{
	// Each element is assigned to the nearest cluster
	dataCluster.DeleteAll();
	const int vectorsCount = matrix.Height;
	for( int i = 0; i < vectorsCount; i++ ) {
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

// Updates the clusters and returns true if the clusters were changed, false if they stayed the same
bool CKMeansClustering::updateClusters( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights,
	const CArray<int>& dataCluster )
{
	// Store the old cluster centers
	CArray<CClusterCenter> oldCenters;
	oldCenters.SetBufferSize( clusters.Size() );
	for( int i = 0; i < clusters.Size(); i++ ) {
		oldCenters.Add( clusters[i]->GetCenter() );
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

} // namespace NeoML
