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

namespace NeoML {

class CCommonCluster;

// K-means clustering algorithm
class NEOML_API CKMeansClustering : public IClustering {
public:
	// K-means clustering parameters
	struct CParam {
		// The distance function
		TDistanceFunc DistanceFunc;
		// The initial cluster count
		// Unless you set up the initial cluster centers when creating the object, 
		// this number of centers will be randomly selected from the input data set
		int InitialClustersCount;
		// The maximum number of iterations
		int MaxIterations;
	};

	// Constructors
	// The initialClusters parameter is the array of cluster centers (of params.InitialClustersCount size)
	// that will be used on the first step of the algorithm
	CKMeansClustering( const CArray<CClusterCenter>& initialClusters, const CParam& params );
	// If you do not specify the initial cluster centers, they will be selected randomly from the input data
	CKMeansClustering( const CParam& params );
	virtual ~CKMeansClustering() {}

	// Sets a text stream for logging processing
	// By default logging is off (set to null to turn off)
	void SetLog( CTextStream* newLog ) { log = newLog; }

	// IClustering inteface methods:
	// Clusterizes the input data and returns true if successful,
	// false if more iterations are needed
	virtual bool Clusterize( IClusteringData* data, CClusteringResult& result );

private:
	const CParam params; // clustering parameters
	CTextStream* log; // the logging stream
	CObjectArray<CCommonCluster> clusters; // the current clusters
	CArray<CClusterCenter> initialClusterCenters; // the initial cluster centers

	void selectInitialClusters( const CSparseFloatMatrixDesc& matrix );
	void classifyAllData( const CSparseFloatMatrixDesc& matrix, CArray<int>& dataCluster );
	int findNearestCluster( const CSparseFloatMatrixDesc& matrix, int dataIndex ) const;
	bool updateClusters( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights,
		const CArray<int>& dataCluster );
};

} // namespace NeoML
