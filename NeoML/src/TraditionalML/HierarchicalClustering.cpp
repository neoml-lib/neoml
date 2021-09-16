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

// Matrix row which supports both random indexing and getting minimum
// FLT_MAX is used as "not set" value
class CDistanceMatrixRow {
public:
	CDistanceMatrixRow() {}
	CDistanceMatrixRow( const CDistanceMatrixRow& other );

	// Random access (CFloatVector-like)
	float operator[] ( int index ) const { return index < distances.Size() ? distances[index] : FLT_MAX; }
	// Sets index'th distance
	// newValue must not be equal to FLT_MAX
	void SetAt( int index, float newValue );
	// Resets index'th distance
	void ResetAt( int index );

	// Returns closest cluster information
	// It takes logN * X where X is a number of changes made since last Closest* method call
	// Returns NotFound if no distance is set in this row
	int ClosestCluster() const { synchronize(); return queue.IsEmpty() ? NotFound : queue.Peek().Index; }
	// Returns FLT_MAX if no distance is set in this row
	float ClosestDistance() const { synchronize(); return queue.IsEmpty() ? FLT_MAX : queue.Peek().Distance; }

private:
	struct CDistanceInfo {
		float Distance;
		int Index;

		CDistanceInfo() : Distance( FLT_MAX ), Index( NotFound ) {}
		CDistanceInfo( float distance, int index ) : Distance( distance ), Index( index ) {}
	};

	// Array with current distances
	CArray<float> distances;
	// Priority queue with minimum distance at the top
	// May contain outdated entries (see synchronize method)
	mutable CPriorityQueue<CArray<CDistanceInfo>,
		CompositeComparer<CDistanceInfo,
			DescendingByMember<CDistanceInfo, float, &CDistanceInfo::Distance>,
			DescendingByMember<CDistanceInfo, int, &CDistanceInfo::Index>>> queue;

	// Synchronizes the peak of the priority queue with current distances
	void synchronize() const
	{
		// queue.Peek().Distance != distances[queue.Peek().Index] means that distance was changed
		// (via SetAt or ResetAt methods)
		while( !queue.IsEmpty() && queue.Peek().Distance != distances[queue.Peek().Index] ) {
			queue.Pop();
		}
	}
};

CDistanceMatrixRow::CDistanceMatrixRow( const CDistanceMatrixRow& other )
{
	other.distances.CopyTo( distances );
	CArray<CDistanceInfo> queueArray;
	other.queue.CopyTo( queueArray );
	queue.Attach( queueArray );
}

void CDistanceMatrixRow::ResetAt( int index )
{
	if( index < distances.Size() ) {
		distances[index] = FLT_MAX;
	}
}

void CDistanceMatrixRow::SetAt( int index, float newValue )
{
	NeoAssert( newValue != FLT_MAX );
	if( index < distances.Size() && distances[index] == newValue ) {
		return;
	}

	if( index >= distances.Size() ) {
		distances.Add( FLT_MAX, index - distances.Size() + 1 );
	}
	distances[index] = newValue;
	queue.Push( CDistanceInfo( newValue, index ) );
}

// Naive hierarchical clustering algorithm (O(N^2 logN) time)
class CNaiveHierarchicalClustering {
	typedef CHierarchicalClustering::CParam CParam;
	typedef CHierarchicalClustering::CMergeInfo CMergeInfo;

public:
	CNaiveHierarchicalClustering( const CParam& params, const CArray<CClusterCenter>& initialClusters, CTextStream* log )
		: params( params ), initialClusters( initialClusters ), log( log ) {}

	bool Clusterize( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
		CClusteringResult& result, CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices );

private:
	const CParam& params; // the clustering parameters
	const CArray<CClusterCenter>& initialClusters; // the initial cluster centers
	CTextStream* log; // the logging stream
	CObjectArray<CCommonCluster> clusters; // the current clusters
	CArray<int> clusterIndices; // the current clusters indices in the dendrogram
	CArray<CDistanceMatrixRow> distances; // distances matrix stored in priority queues

	void initialize( const CFloatMatrixDesc& matrix, const CArray<double>& weights );
	void findNearestClusters( int& first ) const;
	void mergeClusters( int first, int newClusterIndex, CArray<CMergeInfo>* dendrogram );
	float recalcDistance( const CCommonCluster& currCluster, const CCommonCluster& mergedCluster,
		int firstSize, int secondSize, float currToFirst, float currToSecond, float firstToSecond ) const;
	void fillResult( const CFloatMatrixDesc& matrix, CClusteringResult& result, CArray<int>* dendrogramIndices ) const;
};

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
		findNearestClusters( first );
		const float mergeDistance = distances[first].ClosestDistance();

		if( log != 0 ) {
			*log << "Distance: " << mergeDistance << "\n";
		}

		if( mergeDistance > params.MaxClustersDistance ) {
			success = true;
			break;
		}

		if( log != 0 ) {
			*log << "Merge clusters (" << first << ") and (" << distances[first].ClosestCluster() << ") distance - " << mergeDistance << "\n";
		}

		mergeClusters( first, initialClustersCount + step, dendrogram );

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
	distances.SetSize( clusters.Size() );
	for( int i = 0; i < clusters.Size(); i++ ) {
		for( int j = i + 1; j < clusters.Size(); j++ ) {
			const float currDist = static_cast<float>( clusters[i]->CalcDistance( *clusters[j], params.DistanceType ) );
			distances[i].SetAt( j, currDist );
		}
	}
}

// Finds the two closest clusters
void CNaiveHierarchicalClustering::findNearestClusters( int& first ) const
{
	NeoAssert( clusters.Size() > 1 );

	first = 0;
	while( first < clusters.Size() && clusters[first] == nullptr ) {
		++first;
	}
	NeoAssert( first < clusters.Size() );

	for( int i = first + 1; i < clusters.Size(); i++ ) {
		if( clusters[i] == nullptr ) {
			continue;
		}
		if( distances[first].ClosestDistance() > distances[i].ClosestDistance() ) {
			first = i;
		}
	}
}

// Merges two clusters
void CNaiveHierarchicalClustering::mergeClusters( int first, int newClusterIndex, CArray<CMergeInfo>* dendrogram )
{
	const int second = distances[first].ClosestCluster();
	NeoAssert( first < second && clusters[second] != nullptr );

	if( log != 0 ) {
		*log << "Cluster " << first << "\n";
		*log << *clusters[first];
		*log << "Cluster " << second << "\n";
		*log << *clusters[second];
	}

	const int firstSize = clusters[first]->GetElementsCount();
	const int secondSize = clusters[second]->GetElementsCount();
	const float mergeDistance = distances[first].ClosestDistance();
	const int last = clusters.Size() - 1;

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
		if( i < second ) {
			distances[i].ResetAt( second );
		}
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

	if( params.Linkage == CHierarchicalClustering::L_Centroid ) {
		return static_cast<float>( currCluster.CalcDistance( mergedCluster, params.DistanceType ) );
	}

	return NeoML::recalcDistance(params.Linkage, params.DistanceType, firstSize, secondSize, currToFirst, currToSecond, firstToSecond);
}

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

// Nearest neighbor chain clustering algorithm (O(N^2) time, incompatible with centroid linkage)
class CNnChainHierarchicalClustering {
	typedef CHierarchicalClustering::CParam CParam;
	typedef CHierarchicalClustering::CMergeInfo CMergeInfo;

public:
	CNnChainHierarchicalClustering( const CParam& params, CTextStream* log ) : params( params ), log( log ) {}

	bool Clusterize( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
		CClusteringResult& result, CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices );

private:
	const CParam& params; // the clustering parameters
	CTextStream* log; // the logging stream
	CFloatVectorArray distances; // the matrix containing distances between clusters
	CArray<int> clusterSizes; // sizes of current clusters
	CArray<CMergeInfo> fullDendrogram; // dendrogram of the whole tree
	CArray<int> sortedDendrogram; // indices of full dendrogram nodes in distance-increasing order

	void initialize( const CFloatMatrixDesc& matrix );
	void buildFullDendrogram( const CFloatMatrixDesc& matrix );
	void mergeClusters( int first, int second );
	void sortDendrogram();
	bool buildResult( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
		CClusteringResult& result, CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices ) const;
};

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
		if( i == first || clusterSizes[i] == 0 ) {
			continue;
		}
		// We can pass ref to any cluster here because linkage isn't centroid
		const float distance = recalcDistance( params.Linkage, params.DistanceType, firstSize, secondSize,
			i < first ? distances[i][first] : distances[first][i],
			i < second ? distances[i][second] : distances[second][i],
			mergeDistance );
		if( i < first ) {
			distances[i].SetAt( first, distance );
		} else {
			distances[first].SetAt( i, distance );
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

// --------------------------------------------------------------------------------------------------------------------

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
	CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices ) const
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

	static_assert( L_Count == 5, "L_Count != 5" );
	bool success = false;
	switch( params.Linkage ) {
		case L_Centroid:
			success = naiveAlgo( matrix, weights, result, dendrogram, dendrogramIndices );
			break;
		case L_Single:
		case L_Average:
		case L_Complete:
		case L_Ward:
			if( initialClusters.IsEmpty() ) {
				success = nnChainAlgo( matrix, weights, result, dendrogram, dendrogramIndices );
			} else {
				success = naiveAlgo( matrix, weights, result, dendrogram, dendrogramIndices );
			}
			break;
		default:
			NeoAssert( false );
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

// Clusterizes data by using naive algorithm
bool CHierarchicalClustering::naiveAlgo( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
	CClusteringResult& result, CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices ) const
{
	CNaiveHierarchicalClustering naiveClustering( params, initialClusters, log );
	return naiveClustering.Clusterize( matrix, weights, result, dendrogram, dendrogramIndices );
}

// Clusterizes data by using nearest neighbor chain algorithm
bool CHierarchicalClustering::nnChainAlgo(const CFloatMatrixDesc& matrix, const CArray<double>& weights,
	CClusteringResult& result, CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices ) const
{
	NeoAssert( params.Linkage != L_Centroid );
	NeoAssert( initialClusters.IsEmpty() );
	CNnChainHierarchicalClustering nnChainClustering( params, log );
	return nnChainClustering.Clusterize( matrix, weights, result, dendrogram, dendrogramIndices );
}

} // namespace NeoML
