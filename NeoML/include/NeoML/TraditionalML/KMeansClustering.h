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
template<class T>
class CVariableMatrix;
class CDnnBlob;

// K-means clustering algorithm
class NEOML_API CKMeansClustering : public IClustering {
public:
	// Algo used during clusterization
	enum TKMeansAlgo {
		// Lloyd algorithm ('naive')
		KMA_Lloyd = 0,
		// Elkan argorithm
		// If used then the distance func must support triangle inequality
		KMA_Elkan,

		KMA_Count
	};

	// Algo used during cluster initialization
	enum TKMeansInitialization {
		// Some elements of the data
		KMI_Default = 0,
		// KMeans++ initialization
		KMI_KMeansPlusPlus,

		KMI_Count
	};

	// K-means clustering parameters
	struct CParam {
		// Clusterization algorithm
		TKMeansAlgo Algo;
		// The distance function
		TDistanceFunc DistanceFunc;
		// The initial cluster count
		// Unless you set up the initial cluster centers when creating the object, 
		// this number of centers will be randomly selected from the input data set
		int InitialClustersCount;
		// Initialization algorithm
		// It's ignored if initial clusters were provided by user (initialClusters parameter of constructor)
		TKMeansInitialization Initialization;
		// The maximum number of iterations
		int MaxIterations;
		// Tolerance criterion for Elkan algorithm
		double Tolerance;
		// Number of threads used in KMeans
		int ThreadCount;
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

	bool Clusterize( const CFloatVectorArray& data, CClusteringResult& result );

private:
	const CParam params; // clustering parameters
	CTextStream* log; // the logging stream
	CObjectArray<CCommonCluster> clusters; // the current clusters
	CArray<CClusterCenter> initialClusterCenters; // the initial cluster centers

	void selectInitialClusters( const CSparseFloatMatrixDesc& matrix );
	void defaultInitialization( const CSparseFloatMatrixDesc& matrix );
	void kMeansPlusPlusInitialization( const CSparseFloatMatrixDesc& matrix );

	bool clusterize( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights );

	bool lloydClusterization( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights );
	void classifyAllData( const CSparseFloatMatrixDesc& matrix, CArray<int>& dataCluster );
	int findNearestCluster( const CSparseFloatMatrixDesc& matrix, int dataIndex ) const;

	bool elkanClusterization( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights );
	void initializeElkanStatistics( const CSparseFloatMatrixDesc& matrix, CArray<int>& assignments,
		CArray<float>& upperBounds, CVariableMatrix<float>& lowerBounds, CVariableMatrix<float>& clusterDists,
		CArray<float>& closestClusterDist, CArray<float>& moveDistance );
	void computeClustersDists( CVariableMatrix<float>& dists, CArray<float>& closestCluster ) const;
	void assignVectors( const CSparseFloatMatrixDesc& matrix, const CVariableMatrix<float>& clusterDists,
		const CArray<float>& closestClusterDist, CArray<int>& assignments, CArray<float>& upperBounds,
		CVariableMatrix<float>& lowerBounds ) const;
	void updateMoveDistance( const CArray<CClusterCenter>& oldCenters, CArray<float>& moveDistance ) const;
	double updateUpperAndLowerBounds( const CSparseFloatMatrixDesc& matrix,
		const CArray<float>& moveDistance, const CArray<int>& assignments,
		CArray<float>& upperBounds, CVariableMatrix<float>& lowerBounds ) const;
	bool isPruned( const CArray<float>& upperBounds, const CVariableMatrix<float>& lowerBounds,
		const CVariableMatrix<float>& clusterDists, int currentCluster, int clusterToProcess, int id) const;

	void storeClusterCenters( CArray<CClusterCenter>& result );
	bool updateClusters( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights,
		const CArray<int>& dataCluster, const CArray<CClusterCenter>& oldCenters );

	void selectInitialClusters( const CDnnBlob& data, CDnnBlob& centers );
	void defaultInitialization( const CDnnBlob& data, CDnnBlob& centers );
	void kMeansPlusPlusInitialization( const CDnnBlob& data, CDnnBlob& centers );

	bool lloydBlobClusterization( const CDnnBlob& data, CDnnBlob& centers, CDnnBlob& sizes, CDnnBlob& labels );
	double assignClosest( const CDnnBlob& data, const CDnnBlob& squaredData, const CDnnBlob& centers,
		CDnnBlob& labels );
	void recalcCenters( const CDnnBlob& data, const CDnnBlob& labels,
		CDnnBlob& centers, CDnnBlob& sizes );
	void calcClusterVariances( const CDnnBlob& data, const CDnnBlob& labels,
		const CDnnBlob& centers, const CDnnBlob& sizes, CDnnBlob& variances );
};

} // namespace NeoML
