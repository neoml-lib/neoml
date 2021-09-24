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

#include <NnChainHierarchicalClustering.h>

#include <cfloat>

namespace NeoML {

// Recalculates distance between current and the result of the merge of the first and second clusters
// Doesn't support centroid linkage!
static float recalcDistance( CHierarchicalClustering::TLinkage linkage, TDistanceFunc distance, int firstSize, int secondSize,
	float currToFirst, float currToSecond, float firstToSecond )
{
	switch( linkage ) {
		case CHierarchicalClustering::L_Single:
			return ::fminf( currToFirst, currToSecond );
		case CHierarchicalClustering::L_Average:
		{
			if( distance == DF_Euclid || distance == DF_Machalanobis ) {
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
		case CHierarchicalClustering::L_Centroid:
		default:
			NeoAssert( false );
	}

	return 0.f;
}

// --------------------------------------------------------------------------------------------------------------------

// Union-find data structure used for re-labeling clusters in sorted dendrogram during NnChain algo
class CUnionFind
{
public:
	explicit CUnionFind( int size );

	int Root( int index );
	void Merge( int first, int second, int newLabel );

private:
	CArray<int> root;
};

CUnionFind::CUnionFind( int size )
{
	root.SetBufferSize( size );
	for( int i = 0; i < size; ++i ) {
		root.Add( i );
	}
}

int CUnionFind::Root( int index )
{
	int actualRoot = index;
	while( root[actualRoot] != actualRoot ) {
		actualRoot = root[actualRoot];
	}

	while( index != actualRoot ) {
		int temp = index;
		index = root[index];
		root[temp] = actualRoot;
	}

	return actualRoot;
}

void CUnionFind::Merge( int first, int second, int newRoot )
{
	int firstRoot = Root( first );
	int secondRoot = Root( second );

	root[firstRoot] = newRoot;
	root[secondRoot] = newRoot;
}

// --------------------------------------------------------------------------------------------------------------------

bool CNnChainHierarchicalClustering::Clusterize( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
	CClusteringResult& result, CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices )
{
	initialize( matrix );
	buildFullDendrogram( matrix );
	sortDendrogram();
	return buildResult( matrix, weights, result, dendrogram, dendrogramIndices );
}

// Initializes NnChain algo algo
void CNnChainHierarchicalClustering::initialize( const CFloatMatrixDesc& matrix )
{
	const int vectorCount = matrix.Height;
	const int featureCount = matrix.Width;

	clusterSizes.Empty();
	clusterSizes.Add( 1, vectorCount );

	// Initialize the cluster distance matrix
	distances.DeleteAll();
	distances.Add( CFloatVector( vectorCount ), vectorCount );

	for( int i = 0; i < vectorCount; i++ ) {
		CClusterCenter currObject( CFloatVector( featureCount, matrix.GetRow( i ) ) );
		for( int j = i + 1; j < vectorCount; j++ ) {
			distances[i].SetAt( j, static_cast<float>( CalcDistance( currObject, matrix.GetRow( j ), params.DistanceType ) ) );
		}
	}
}

// Builds full dendrogram
// NnChain always builds full tree and cuts it later (if result contains more than 1 cluster)
void CNnChainHierarchicalClustering::buildFullDendrogram( const CFloatMatrixDesc& matrix )
{
	fullDendrogram.Empty();
	fullDendrogram.SetBufferSize( matrix.Height - 1 );

	CArray<int> chain;
	chain.SetSize( matrix.Height );
	int chainSize = 0;

	for( int step = 0; step < matrix.Height - 1; ++step ) {
		if( chainSize == 0 ) {
			int first = 0;
			while( clusterSizes[first] == 0 ) {
				++first;
			}
			chain[chainSize++] = first;
		}

		while( true ) {
			const int first = chain[chainSize - 1];
			int second = chainSize == 1 ? NotFound : chain[chainSize - 2];
			float minDistance = second == NotFound ? FLT_MAX
				: ( first < second ? distances[first][second] : distances[second][first] );

			for( int candidate = 0; candidate < clusterSizes.Size(); ++candidate ) {
				if( candidate == first || clusterSizes[candidate] == 0 ) {
					continue;
				}

				const float currDistance = candidate < first ? distances[candidate][first] : distances[first][candidate];
				if( currDistance < minDistance ) {
					minDistance = currDistance;
					second = candidate;
				}
			}

			if( chainSize > 1 && chain[chainSize - 2] == second ) {
				break;
			}

			chain[chainSize++] = second;
		}

		const int first = chain[--chainSize];
		const int second = chain[--chainSize];
		mergeClusters( first, second );
	}
}

// Merges 2 clusters during NnChain algorithm and adds merge result to the full dendrogram
void CNnChainHierarchicalClustering::mergeClusters( int first, int second )
{
	if( second < first ) {
		swap( first, second );
	}

	const int firstSize = clusterSizes[first];
	const int secondSize = clusterSizes[second];
	const float mergeDistance = distances[first][second];

	CMergeInfo& newMerge = fullDendrogram.Append();
	newMerge.First = first;
	newMerge.Second = second;
	newMerge.Distance = mergeDistance;

	clusterSizes[first] = 0;
	clusterSizes[second] = firstSize + secondSize;

	for( int i = 0; i < clusterSizes.Size(); i++ ) {
		if( i == second || clusterSizes[i] == 0 ) {
			continue;
		}
		// We can pass ref to any cluster here because linkage isn't centroid
		const float distance = recalcDistance( params.Linkage, params.DistanceType, firstSize, secondSize,
			i < first ? distances[i][first] : distances[first][i],
			i < second ? distances[i][second] : distances[second][i],
			mergeDistance );
		if( i < second ) {
			distances[i].SetAt( second, distance );
		} else {
			distances[second].SetAt( i, distance );
		}
	}
}

// Finds distance-sorted order of the current full dendrogram
void CNnChainHierarchicalClustering::sortDendrogram()
{
	sortedDendrogram.Empty();
	sortedDendrogram.SetBufferSize( fullDendrogram.Size() );
	for( int i = 0; i < fullDendrogram.Size(); ++i ) {
		sortedDendrogram.Add( i );
	}

	// Compares indices based on the distances in the data array
	class CMergeInfoIndexCompare
	{
	public:
		explicit CMergeInfoIndexCompare( const CArray<CMergeInfo>& data ) : data( data ) {}

		bool Predicate( const int& first, const int& second ) const { return data[first].Distance < data[second].Distance; }
		bool IsEqual( const int& first, const int& second ) const { return data[first].Distance == data[second].Distance; }
		void Swap( int& first, int& second ) const { swap( first, second ); }
	private:
		const CArray<CMergeInfo>& data;
	};

	CMergeInfoIndexCompare compare( fullDendrogram );
	sortedDendrogram.QuickSort<CMergeInfoIndexCompare>( &compare );
}

bool CNnChainHierarchicalClustering::buildResult( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
	CClusteringResult& result, CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices ) const
{
	bool success = false;
	int clusterCount = matrix.Height;

	NeoAssert( ( dendrogram == nullptr && dendrogramIndices == nullptr )
		|| ( dendrogram != nullptr && dendrogramIndices != nullptr ) );

	CObjectArray<CCommonCluster> clusters;
	clusters.Add( nullptr, matrix.Height * 2 - 1 );
	for( int i = 0; i < matrix.Height; ++i ) {
		clusters[i] = FINE_DEBUG_NEW CCommonCluster( CClusterCenter( CFloatVector( matrix.Width, 0.f ) ) );
		clusters[i]->Add( i, matrix.GetRow( i ), weights[i] );
		clusters[i]->RecalcCenter();
	}

	// Step 1: build shortened dendrogram and re-label clusters
	CUnionFind clusterIndex( 2 * matrix.Height - 1 );
	for( int step = 0; step < sortedDendrogram.Size(); ++step ) {
		if( log != 0 ) {
			*log << "\n[Step " << step << "]\n";
		}

		const int currClusterCount = matrix.Height - step;
		if( currClusterCount <= params.MinClustersCount ) {
			break;
		}

		CMergeInfo merge = fullDendrogram[sortedDendrogram[step]];
		if( log != 0 ) {
			*log << "Distance: " << merge.Distance << "\n";
		}

		if( merge.Distance > params.MaxClustersDistance ) {
			success = true;
			break;
		}

		merge.First = clusterIndex.Root( merge.First );
		merge.Second = clusterIndex.Root( merge.Second );
		if( merge.Second < merge.First ) {
			swap( merge.First, merge.Second );
		}
		const int newClusterIndex = matrix.Height + step;
		clusterIndex.Merge( merge.First, merge.Second, newClusterIndex );
		clusters[newClusterIndex] = FINE_DEBUG_NEW CCommonCluster( *clusters[merge.First], *clusters[merge.Second] );
		clusters[merge.First] = nullptr;
		clusters[merge.Second] = nullptr;
		clusterCount--;

		if( log != 0 ) {
			*log << "Merge clusters (" << merge.First << ") and (" << merge.Second << ") distance - " << merge.Distance << "\n";
		}

		if( dendrogram != nullptr ) {
			merge.Center = clusters[matrix.Height + step]->GetCenter();
			dendrogram->Add( merge );
		}
	}

	// Step 2: fill CClusteringResult
	result.ClusterCount = clusterCount;
	result.Data.SetSize( matrix.Height );
	if( dendrogramIndices != nullptr ) {
		dendrogramIndices->Empty();
		dendrogramIndices->SetBufferSize( clusterCount );
	}
	CArray<int> resultClusterIndex;
	resultClusterIndex.Add( NotFound, clusters.Size() );
	result.Clusters.Empty();
	result.Clusters.SetBufferSize( clusterCount );
	for( int i = 0; i < matrix.Height; ++i ) {
		const int dendrogramClusterIndex = clusterIndex.Root( i );
		if( clusters[dendrogramClusterIndex] != nullptr ) {
			NeoAssert( resultClusterIndex[dendrogramClusterIndex] == NotFound );
			resultClusterIndex[dendrogramClusterIndex] = result.Clusters.Size();
			result.Clusters.Add( clusters[dendrogramClusterIndex]->GetCenter() );
			clusters[dendrogramClusterIndex] = nullptr;
			if( dendrogramIndices != nullptr ) {
				dendrogramIndices->Add( dendrogramClusterIndex );
			}
		}
		NeoAssert( resultClusterIndex[dendrogramClusterIndex] != NotFound );
		result.Data[i] = resultClusterIndex[dendrogramClusterIndex];
	}
	NeoAssert( result.Clusters.Size() == clusterCount );
	NeoAssert( dendrogramIndices == nullptr || dendrogramIndices->Size() == clusterCount );

	return success;
}

} // namespace NeoML
