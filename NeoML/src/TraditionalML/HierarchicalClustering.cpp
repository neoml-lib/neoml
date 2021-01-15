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

#include <NeoML/TraditionalML/HierarchicalClustering.h>
#include <float.h>

namespace NeoML {

CHierarchicalClustering::CHierarchicalClustering( const CArray<CClusterCenter>& clustersCenters, const CParam& _params ) :
	params( _params ),
	log( 0 )
{
	NeoAssert( params.MinClustersCount > 0 );
	clustersCenters.CopyTo( initialClusters );
}

CHierarchicalClustering::CHierarchicalClustering( const CParam& _params ) :
	params( _params ),
	log( 0 )
{
	NeoAssert( params.MinClustersCount > 0 );
}

bool CHierarchicalClustering::Clusterize( IClusteringData* data, CClusteringResult& result )
{
	NeoAssert( data != 0 );

	if( log != 0 ) {
		*log << "\nHierarchical clustering started:\n";
	}

	CFloatMatrixDesc matrix = data->GetMatrix();
	NeoAssert( matrix.Height == data->GetVectorCount() );
	NeoAssert( matrix.Width == data->GetFeaturesCount() );

	CArray<double> weights;
	for( int i = 0; i < data->GetVectorCount(); i++ ) {
		weights.Add( data->GetVectorWeight( i ) );
	}

	initialize( matrix, weights );

	if( log != 0 ) {
		*log << "Initial clusters:\n";

		for( int i = 0; i < clusters.Size(); i++ ) {
			*log << *clusters[i] << "\n";
		}
	}

	bool success = false;
	const int initialClustersCount = clusters.Size();
	while( true ) {
		if( log != 0 ) {
			*log << "\n[Step " << initialClustersCount - clusters.Size() << "]\n";
		}

		int first = NotFound;
		int second = NotFound;
		findNearestClusters( first, second );

		if( log != 0 ) {
			*log << "Distance: " << distances[first][second] << "\n";
		}

		if( distances[first][second] > params.MaxClustersDistance ) {
			success = true;
			break;
		}

		if( clusters.Size() <= params.MinClustersCount ) {
			break;
		}

		if( log != 0 ) {
			*log << "Merge clusters (" << first << ") and (" << second << ") distance - " << distances[first][second] << "\n";
		}

		mergeClusters( matrix, weights, first, second );
	}

	result.ClusterCount = clusters.Size();
	result.Data.SetSize( data->GetVectorCount() );
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
			*log << "\nMaxClustersDistance is too small!\n";
		}
	}

	return success;
}

// Initializes the algorithm
void CHierarchicalClustering::initialize( const CFloatMatrixDesc& matrix, const CArray<double>& weights )
{
	const int vectorsCount = matrix.Height;

	// Define the initial cluster set
	if( initialClusters.IsEmpty() ) {
		// Each element is a cluster
		clusters.SetBufferSize( vectorsCount );
		for( int i = 0; i < vectorsCount; i++ ) {
			CFloatVectorDesc desc;
			matrix.GetRow( i, desc );
			CFloatVector mean( matrix.Width, desc );
			clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( mean ) ) );
			clusters.Last()->Add( i, desc, weights[i] );
		}
	} else {
		// The initial cluster centers have been specified directly
		clusters.SetBufferSize( initialClusters.Size() );
		for( int i = 0; i < initialClusters.Size(); i++ ) {
			clusters.Add( FINE_DEBUG_NEW CCommonCluster( initialClusters[i] ) );
		}

		// Each element of the original data set is put into the nearest cluster
		for( int i = 0; i < vectorsCount; i++ ) {
			int nearestCluster = 0;
			CFloatVectorDesc desc;
			matrix.GetRow( i, desc );
			double minDistance = clusters[nearestCluster]->CalcDistance( desc, params.DistanceType );

			for( int j = 0; j < clusters.Size(); j++ ) {
				const double distance = clusters[j]->CalcDistance( desc, params.DistanceType );
				if( distance < minDistance ) {
					minDistance = distance;
					nearestCluster = j;
				}
			}

			NeoAssert( nearestCluster == i );
			clusters[nearestCluster]->Add( i, desc, weights[i] );
		}

		for( int i = 0; i < clusters.Size(); i++ ) {
			clusters[i]->RecalcCenter();
		}
	}

	NeoAssert( !clusters.IsEmpty() );

	// Initialize the cluster distance matrix
	distances.DeleteAll();
	distances.Add( CFloatVector( clusters.Size() ), clusters.Size() );
	for( int i = 0; i < clusters.Size(); i++ ) {
		for( int j = i + 1; j < clusters.Size(); j++ ) {
			distances[i].SetAt( j, static_cast<float>( clusters[i]->CalcDistance( *clusters[j], params.DistanceType ) ) );
		}
	}
}

// Finds the two closest clusters
void CHierarchicalClustering::findNearestClusters( int& first, int& second ) const
{
	NeoAssert( clusters.Size() > 1 );

	first = 0;
	second = 1;
	for( int i = 0; i < clusters.Size(); i++ ) {
		for( int j = i + 1; j < clusters.Size(); j++ ) {
			if( distances[i][j] < distances[first][second] ) {
				first = i;
				second = j;
			}
		}
	}
}

// Merges two clusters
void CHierarchicalClustering::mergeClusters( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
	int first, int second )
{
	NeoAssert( first < second );

	if( log != 0 ) {
		*log << "Cluster " << first << "\n";
		*log << *clusters[first];
		*log << "Cluster " << second << "\n";
		*log << *clusters[second];
	}

	// Move all elements of the second cluster into the first
	CArray<int> secondClustersElements;
	clusters[second]->GetAllElements( secondClustersElements );
	for( int i = 0; i < secondClustersElements.Size(); i++ ) {
		CFloatVectorDesc desc;
		matrix.GetRow( secondClustersElements[i], desc );
		clusters[first]->Add( secondClustersElements[i], desc, weights[secondClustersElements[i]] );
	}
	clusters[first]->RecalcCenter();

	// Switch the second cluster with the last; now we can calculate the cluster distance matrix in linear time
	const int last = clusters.Size() - 1;
	clusters[second] = clusters[last];
	distances[second] = distances[last];
	for( int i = 0; i < second; i++ ) {
		distances[i].SetAt( second, distances[i][last] );
	}
	for( int i = second + 1; i < last; i++ ) {
		distances[second].SetAt( i, distances[i][last] );
	}
	for( int i = 0; i < last; i++ ) {
		const double distance = clusters[first]->CalcDistance( *clusters[i], params.DistanceType );
		if( i < first ) {
			distances[i].SetAt( first, static_cast<float>( distance ) );
		} else {
			distances[first].SetAt( i, static_cast<float>( distance ) );
		}
	}
	clusters.SetSize( last );

	if( log != 0 ) {
		*log << "Result:\n";
		*log << *clusters[first];
	}
}

} // namespace NeoML
