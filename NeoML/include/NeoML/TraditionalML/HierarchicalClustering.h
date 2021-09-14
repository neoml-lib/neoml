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
	// Hierarchical clustering linkage
	// Determines the approach used for distance calculation between clusters
	enum TLinkage {
		L_Centroid, // Distance between centroids (NeoML's default)
		L_Single, // Min distance between objects in clusters
		L_Average, // Average distance between objects in clusters
		L_Complete, // Max distance between objects in clusters
		L_Ward, // Ward's linkage

		L_Count
	};

	// Algorithm settings
	struct CParam {
		TDistanceFunc DistanceType; // the distance function
		double MaxClustersDistance; // the maximum distance between two clusters that still may be merged
		int MinClustersCount; // the minimum number of clusters in the result
		TLinkage Linkage; // the clustering linkage

		CParam() : DistanceType( DF_Euclid ), MaxClustersDistance( 1e32 ),
			MinClustersCount( 1 ), Linkage( L_Centroid ) {}
	};

	CHierarchicalClustering( const CArray<CClusterCenter>& clusters, const CParam& params );
	explicit CHierarchicalClustering( const CParam& clusteringParams );

	// Sets a text stream for logging processing
	// By default logging is off (set to null to turn off)
	void SetLog( CTextStream* newLog ) { log = newLog; }

	// IClustering interface methods:
	// Returns true if the specified distance between the clusters was reached AND 
	// there are more than MinClustersCount clusters
	bool Clusterize( IClusteringData* input, CClusteringResult& result ) override;

	// Information about one step of the clustering
	struct CMergeInfo {
		// Merged clusters indices
		// If initial centers were not provided to the constructor, then InitialClusters is equal to VectorCount
		// and each of initial clusters consists of the corresponding input vector
		// If index is in [0; InitialClusters - 1] then it's one of the initial clusters
		// If index >- InitialClusters then it's the result of (index - InitialClusters)'th merge
		int First;
		int Second;
		// The distance between clusters before merge
		double Distance;
	};

	// Returns true if the specified distance between the clusters was reached AND 
	// there are more than MinClustersCount clusters
	// Also fills the dendrogram: a sequence of (InitialClusters - ClusterCount) merges
	// dendrogramIndices[i] contains the index of result.Clusters[i] in the dendrogram
	bool ClusterizeEx( IClusteringData* input, CClusteringResult& result,
		CArray<CMergeInfo>& dendrogram, CArray<int>& dendrogramIndices );

private:
	const CParam params; // the clustering parameters
	CTextStream* log; // the logging stream
	CArray<CClusterCenter> initialClusters; // the initial cluster centers

	bool clusterizeImpl( IClusteringData* input, CClusteringResult& result,
		CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices ) const;
	bool naiveAlgo( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
		CClusteringResult& result, CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices ) const;
	bool nnChainAlgo( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
		CClusteringResult& result, CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices ) const;
};

} // namespace NeoML
