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

#pragma once

#include <NeoML/TraditionalML/HierarchicalClustering.h>

#include <cfloat>

namespace NeoML {

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
	// Resets all the distances
	void Reset();
	// Copies the content to the array
	void CopyTo( CArray<float>& buffer ) const { distances.CopyTo( buffer ); }

	// Returns closest cluster information
	// It takes logN * X where X is a number of outdated entries
	// Outdated entries appear when overwriting values which were already set
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
	void synchronize() const;
};

// --------------------------------------------------------------------------------------------------------------------

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

} // namespace NeoML
