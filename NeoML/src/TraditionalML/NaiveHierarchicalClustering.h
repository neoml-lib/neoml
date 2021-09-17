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

namespace NeoML {

// Naive hierarchical clustering algorithm (O(N^3) time)
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
	CFloatVectorArray distances; // the matrix containing distances between clusters

	void initialize( const CFloatMatrixDesc& matrix, const CArray<double>& weights );
	void findNearestClusters( int& first, int& second ) const;
	void mergeClusters( int first, int second, int newClusterIndex, CArray<CMergeInfo>* dendrogram );
	float recalcDistance( const CCommonCluster& currCluster, const CCommonCluster& mergedCluster,
		int firstSize, int secondSize, float currToFirst, float currToSecond, float firstToSecond ) const;
	void fillResult( const CFloatMatrixDesc& matrix, CClusteringResult& result, CArray<int>* dendrogramIndices ) const;
};

} // namespace NeoML
