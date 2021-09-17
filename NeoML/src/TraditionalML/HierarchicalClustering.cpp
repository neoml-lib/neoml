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
#include <math.h>

namespace NeoML {

CHierarchicalClustering::CHierarchicalClustering( const CArray<CClusterCenter>& clustersCenters, const CParam& _params ) :
	params( _params ),
	log( 0 )
{
	NeoAssert( params.MinClustersCount > 0 );
	// Initial cluster centers are supported only for centroid
	NeoAssert( params.Linkage == L_Centroid );
	clustersCenters.CopyTo( initialClusters );
}

CHierarchicalClustering::CHierarchicalClustering( const CParam& _params ) :
	params( _params ),
	log( 0 )
{
	NeoAssert( params.MinClustersCount > 0 );
}

bool CHierarchicalClustering::Clusterize( IClusteringData* input, CClusteringResult& result )
{
	return clusterizeImpl( input, result, nullptr, nullptr );
}

bool CHierarchicalClustering::ClusterizeEx( IClusteringData* input, CClusteringResult& result,
	CArray<CMergeInfo>& dendrogram, CArray<int>& dendrogramIndices )
{
	return clusterizeImpl( input, result, &dendrogram, &dendrogramIndices );
}

bool CHierarchicalClustering::clusterizeImpl( IClusteringData* data, CClusteringResult& result,
	CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices )
{
	NeoAssert( params.Linkage != L_Ward || params.DistanceType == DF_Euclid ); // Ward works only in L2
	NeoAssert( ( dendrogram != nullptr && dendrogramIndices != nullptr )
		|| ( dendrogram == nullptr && dendrogramIndices == nullptr ) );
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

		if( clusters.Size() <= params.MinClustersCount ) {
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

		mergeClusters( matrix, first, second, dendrogram );
	}

	result.ClusterCount = clusters.Size();
	result.Data.SetSize( data->GetVectorCount() );
	result.Clusters.SetBufferSize( clusters.Size() );

	if( dendrogramIndices != nullptr ) {
		dendrogramIndices->Empty();
		dendrogramIndices->SetBufferSize( clusters.Size() );
	}

	for( int i = 0; i < clusters.Size(); i++ ) {
		CArray<int> elements;
		clusters[i]->GetAllElements( elements );
		for(int j = 0; j < elements.Size(); j++ ) {
			result.Data[elements[j]]=i;
		}
		result.Clusters.Add( clusters[i]->GetCenter() );
		if( dendrogramIndices != nullptr ) {
			dendrogramIndices->Add( clusterIndices[i] );
		}
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
void CHierarchicalClustering::mergeClusters( const CFloatMatrixDesc& matrix, int first, int second, CArray<CMergeInfo>* dendrogram )
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
	const int last = clusters.Size() - 1;

	// Move all elements of the second cluster into the first
	const int initialClusterCount = initialClusters.IsEmpty() ? matrix.Height : initialClusters.Size();
	const int newClusterIndex = 2 * initialClusterCount - clusters.Size();
	clusters[first] = FINE_DEBUG_NEW CCommonCluster( *clusters[first], *clusters[second] );

	if( dendrogram != nullptr ) {
		CMergeInfo& mergeInfo = dendrogram->Append();
		mergeInfo.First = clusterIndices[first];
		mergeInfo.Second = clusterIndices[second];
		mergeInfo.Distance = mergeDistance;
		mergeInfo.Center = clusters[first]->GetCenter();
	}

	clusterIndices[first] = newClusterIndex;
	clusters[second] = clusters[last];
	clusterIndices[second] = clusterIndices[last];
	clusters.SetSize( last );

	// Switch the second cluster with the last; now we can calculate the cluster distance matrix in linear time
	CFloatVector secondDistances = distances[second];
	distances[second] = distances[last];
	for( int i = 0; i < second; i++ ) {
		secondDistances.SetAt( i, distances[i][second] );
		distances[i].SetAt( second, distances[i][last] );
	}
	for( int i = second + 1; i < last; i++ ) {
		// These were already copied during secondDistances construction
		distances[second].SetAt( i, distances[i][last] );
	}
	// distances[last] was moved to distances[second]
	secondDistances.SetAt( second, secondDistances[last] );
	for( int i = 0; i < last; i++ ) {
		if( i == first ) {
			distances[first].SetAt( first, 0.f );
			continue;
		}
		const float distance = recalcDistance( *clusters[i], *clusters[first], firstSize, secondSize,
			i < first ? distances[i][first] : distances[first][i], secondDistances[i], mergeDistance );
		if( i < first ) {
			distances[i].SetAt( first, distance );
		} else {
			distances[first].SetAt( i, distance );
		}
	}

	if( log != 0 ) {
		*log << "Result:\n";
		*log << *clusters[first];
	}
}

// Calculates distance between current cluster and merged cluster based on 
float CHierarchicalClustering::recalcDistance( const CCommonCluster& currCluster, const CCommonCluster& mergedCluster,
	int firstSize, int secondSize, float currToFirst, float currToSecond, float firstToSecond ) const
{
	static_assert( L_Count == 5, "L_Count != 5" );

	switch( params.Linkage ) {
		case L_Centroid:
			// No optimizations, just calculate distance between new centers
			return static_cast<float>( currCluster.CalcDistance( mergedCluster, params.DistanceType ) );
		case L_Single:
			return ::fminf( currToFirst, currToSecond );
		case L_Average:
		{
			if( params.DistanceType == DF_Euclid || params.DistanceType == DF_Machalanobis ) {
				// NeoML works with squared Euclid and Machalanobis
				const float total = ::sqrtf( currToFirst ) * firstSize + ::sqrtf( currToSecond ) * secondSize;
				const float avg = total / ( firstSize + secondSize );
				return avg * avg;
			}
			return ( currToFirst * firstSize + currToSecond * secondSize ) / ( firstSize + secondSize );
		}
		case L_Complete:
			return ::fmaxf( currToFirst, currToSecond );
		case L_Ward:
		{
			const int mergeSize = firstSize + secondSize;
			return ( firstSize * currToFirst + secondSize * currToSecond
				- ( firstSize * secondSize * firstToSecond ) / mergeSize ) / mergeSize;
		}
		default:
			NeoAssert( false );
	}

	return 0.f;
}

} // namespace NeoML
