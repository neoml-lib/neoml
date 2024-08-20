/* Copyright © 2017-2023 ABBYY

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
#include <NeoML/TraditionalML/SparseFloatMatrix.h>
#include <NeoML/TraditionalML/CommonCluster.h>

namespace NeoML {

// Clustering result
class NEOML_API CClusteringResult final {
public:
	int ClusterCount = 0;               // the number of clusters
	CArray<int> Data{};                 // the array of cluster numbers for each of the input data elements 
	                                    // (the clusters are numbered from 0 to ClusterCount - 1)
	CArray<CClusterCenter> Clusters{};  // the cluster centers

	CClusteringResult() = default;
	CClusteringResult( const CClusteringResult& result ) { result.CopyTo( *this ); }

	void CopyTo( CClusteringResult& to ) const;
};

inline void CClusteringResult::CopyTo( CClusteringResult& to ) const
{
	to.ClusterCount = ClusterCount;
	Data.CopyTo( to.Data );
	Clusters.CopyTo( to.Clusters );
}

//---------------------------------------------------------------------------------------------------------

// The input data set for clustering
class IClusteringData : public virtual IObject {
public:
	// The number of vectors
	virtual int GetVectorCount() const = 0;

	// The number of features (vector length)
	virtual int GetFeaturesCount() const = 0;

	// Gets all input vectors as a matrix of size GetVectorCount() x GetFeaturesCount()
	virtual CFloatMatrixDesc GetMatrix() const = 0;

	// Gets the weight of the vector with the given index
	virtual double GetVectorWeight( int index ) const = 0;
};

// The clustering algorithm interface
class IClustering {
public:
	virtual ~IClustering() {}

	// Clusterizes the input data 
	// and returns true if successful with the given parameters
	virtual bool Clusterize( const IClusteringData*, CClusteringResult& ) = 0;
};

} // namespace NeoML
