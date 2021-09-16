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

} // namespace NeoML
