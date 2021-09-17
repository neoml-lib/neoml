/* Copyright © 2017-2021 ABBYY Production LLC

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

#include <NaiveHierarchicalClustering.h>

namespace NeoML {

bool CNaiveHierarchicalClustering::Clusterize( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
	CClusteringResult& result, CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices )
{
	initialize( matrix, weights );

	if( log != 0 ) {
		*log << "Initial clusters:\n";

		for( int i = 0; i < clusters.Size(); i++ ) {
			*log << *clusters[i] << "\n";
		}
	}

	bool success = false;
	const int initialClustersCount = clusters.Size();
	int step = 0;
	while( true ) {
		if( log != 0 ) {
			*log << "\n[Step " << step << "]\n";
		}

		if( initialClustersCount - step <= params.MinClustersCount ) {
			break;
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

		if( log != 0 ) {
			*log << "Merge clusters (" << first << ") and (" << second << ") distance - " << distances[first][second] << "\n";
		}

		mergeClusters( first, second, initialClustersCount + step, dendrogram );

		step += 1;
	}

	fillResult( matrix, result, dendrogramIndices );

	return success;
}

// Initializes the algorithm
void CNaiveHierarchicalClustering::initialize( const CFloatMatrixDesc& matrix, const CArray<double>& weights )
{
	const int vectorsCount = matrix.Height;

	// Define the initial cluster set
	if( initialClusters.IsEmpty() ) {
		// Each element is a cluster
		clusters.SetBufferSize( vectorsCount );
		clusterIndices.SetBufferSize( vectorsCount );
		for( int i = 0; i < vectorsCount; i++ ) {
			CFloatVectorDesc desc;
			matrix.GetRow( i, desc );
			CFloatVector mean( matrix.Width, desc );
			clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( mean ) ) );
			clusters.Last()->Add( i, desc, weights[i] );
			clusters.Last()->RecalcCenter();
			clusterIndices.Add( i );
		}
	} else {
		// The initial cluster centers have been specified directly
		clusters.SetBufferSize( initialClusters.Size() );
		clusterIndices.SetBufferSize( initialClusters.Size() );
		for( int i = 0; i < initialClusters.Size(); i++ ) {
			clusters.Add( FINE_DEBUG_NEW CCommonCluster( initialClusters[i] ) );
			clusterIndices.Add( i );
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
void CNaiveHierarchicalClustering::findNearestClusters( int& first, int& second ) const
{
	NeoAssert( clusters.Size() > 1 );

	first = 0;
	while( first < clusters.Size() && clusters[first] == nullptr ) {
		++first;
	}
	NeoAssert( first < clusters.Size() );

	second = first + 1;
	while( second < clusters.Size() && clusters[second] == nullptr ) {
		++second;
	}
	NeoAssert( second < clusters.Size() );

	for( int i = first; i < clusters.Size(); i++ ) {
		if( clusters[i] == nullptr ) {
			continue;
		}
		for( int j = i + 1; j < clusters.Size(); j++ ) {
			if( clusters[j] != nullptr && distances[i][j] < distances[first][second] ) {
				first = i;
				second = j;
			}
		}
	}
}

// Merges two clusters
void CNaiveHierarchicalClustering::mergeClusters( int first, int second, int newClusterIndex, CArray<CMergeInfo>* dendrogram )
{
	NeoAssert( first < second );

	if( log != 0 ) {
		*log << "Cluster " << first << "\n";
		*log << *clusters[first];
		*log << "Cluster " << second << "\n";
		*log << *clusters[second];
	}

	const int firstSize = clusters[first]->GetElementsCount();
	const int secondSize = clusters[second]->GetElementsCount();
	const float mergeDistance = distances[first][second];

	// Move all elements of the second cluster into the first
	clusters[first] = FINE_DEBUG_NEW CCommonCluster( *clusters[first], *clusters[second] );
	if( dendrogram != nullptr ) {
		CMergeInfo& mergeInfo = dendrogram->Append();
		mergeInfo.First = clusterIndices[first];
		mergeInfo.Second = clusterIndices[second];
		mergeInfo.Distance = mergeDistance;
		mergeInfo.Center = clusters[first]->GetCenter();
	}
	clusters[second] = nullptr;

	for( int i = 0; i < clusters.Size(); i++ ) {
		if( i == first || clusters[i] == nullptr ) {
			continue;
		}
		const float distance = recalcDistance( *clusters[i], *clusters[first], firstSize, secondSize,
			i < first ? distances[i][first] : distances[first][i],
			i < second ? distances[i][second] : distances[second][i],
			mergeDistance );
		if( i < first ) {
			distances[i].SetAt( first, distance );
		} else {
			distances[first].SetAt( i, distance );
		}
	}

	clusterIndices[first] = newClusterIndex;

	if( log != 0 ) {
		*log << "Result:\n";
		*log << *clusters[first];
	}
}

// Calculates distance between current cluster and merged cluster based on 
float CNaiveHierarchicalClustering::recalcDistance( const CCommonCluster& currCluster, const CCommonCluster& mergedCluster,
	int firstSize, int secondSize, float currToFirst, float currToSecond, float firstToSecond ) const
{
	static_assert( CHierarchicalClustering::L_Count == 5, "L_Count != 5" );

	switch( params.Linkage ) {
		case CHierarchicalClustering::L_Centroid:
			return static_cast<float>( currCluster.CalcDistance( mergedCluster, params.DistanceType ) );
		case CHierarchicalClustering::L_Single:
			return ::fminf( currToFirst, currToSecond );
		case CHierarchicalClustering::L_Average:
		{
			if( params.DistanceType == DF_Euclid || params.DistanceType == DF_Machalanobis ) {
				// NeoML works with squared Euclid and Machalanobis
				const float total = ::sqrtf( currToFirst ) * firstSize + ::sqrtf( currToSecond ) * secondSize;
				const float avg = total / ( firstSize + secondSize );
				return avg * avg;
			}
			return ( currToFirst * firstSize + currToSecond * secondSize ) / ( firstSize + secondSize );
		}
		case CHierarchicalClustering::L_Complete:
			return ::fmaxf( currToFirst, currToSecond );
		case CHierarchicalClustering::L_Ward:
		{
			const int mergedSize = firstSize + secondSize;
			return ( firstSize * currToFirst + secondSize * currToSecond
				- ( firstSize * secondSize * firstToSecond ) / mergedSize ) / mergedSize;
		}
		default:
			NeoAssert( false );
	}

	return 0.f;
}

// Fills CClusteringResult and dendrogramIndices
void CNaiveHierarchicalClustering::fillResult( const CFloatMatrixDesc& matrix,
	CClusteringResult& result, CArray<int>* dendrogramIndices ) const
{
	result.Data.SetSize( matrix.Height );
	result.Clusters.Empty();
	if( dendrogramIndices != nullptr ) {
		dendrogramIndices->Empty();
	}

	for( int i = 0; i < clusters.Size(); i++ ) {
		if( clusters[i] == nullptr ) {
			continue;
		}

		CArray<int> elements;
		clusters[i]->GetAllElements( elements );
		for( int j = 0; j < elements.Size(); j++ ) {
			result.Data[elements[j]] = i;
		}
		result.Clusters.Add( clusters[i]->GetCenter() );
		if( dendrogramIndices != nullptr ) {
			dendrogramIndices->Add( clusterIndices[i] );
		}
	}

	result.ClusterCount = result.Clusters.Size();
}

} // namespace NeoML
