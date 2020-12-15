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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/Clustering.h>
#include <NeoML/TraditionalML/FloatVector.h>
#include <NeoML/TraditionalML/CommonCluster.h>

namespace NeoML {

// Hierarchical clustering algorithm
// Merges the two closest clusters on each step, 
// until the limit to the clusters number or the distance between them is reached
class NEOML_API CHierarchicalClustering : public IClustering {
public:
	// Algorithm settings
	struct CParam {
		TDistanceFunc DistanceType; // the distance function
		double MaxClustersDistance; // the maximum distance between two clusters that still may be merged
		int MinClustersCount; // the minimum number of clusters in the result
	};

	CHierarchicalClustering( const CArray<CClusterCenter>& clusters, const CParam& params );
	explicit CHierarchicalClustering( const CParam& clusteringParams );

	// Sets a text stream for logging processing
	// By default logging is off (set to null to turn off)
	void SetLog( CTextStream* newLog ) { log = newLog; }

	// IClustering interface methods:
	// Returns true if the specified distance between the clusters was reached AND 
	// there are more than MinClustersCount clusters
	bool Clusterize( ISparseClusteringData* input, CClusteringResult& result ) override;

	// Not implemented
	bool Clusterize( IDenseClusteringData* input, CClusteringResult& result ) override { NeoAssert( false ); return false; };

private:
	const CParam params; // the clustering parameters
	CTextStream* log; // the logging stream
	CArray<CClusterCenter> initialClusters; // the initial cluster centers
	CObjectArray<CCommonCluster> clusters; // the current clusters
	CFloatVectorArray distances; // the matrix containing distances between clusters

	void initialize( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights );
	void findNearestClusters( int& first, int& second ) const;
	void mergeClusters( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights, int first, int second );
};

} // namespace NeoML
