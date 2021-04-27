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
		// Number of runs of algorithm
		// If more than one then the best variant (least ineratia) will be returned
		int RunCount;
		// Initial seed for random
		int Seed;

		CParam() : Algo( KMA_Lloyd ), DistanceFunc( DF_Euclid ), InitialClustersCount( 1 ), Initialization( KMI_Default ),
			MaxIterations( 1 ), Tolerance( 1e-5f ), ThreadCount( 1 ), RunCount( 1 ), Seed( 0xCEA )
		{
		}
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
	bool Clusterize( IClusteringData* data, CClusteringResult& result ) override;

private:
	const CParam params; // clustering parameters
	CTextStream* log; // the logging stream
	CObjectArray<CCommonCluster> clusters; // the current clusters
	CArray<CClusterCenter> initialClusterCenters; // the initial cluster centers

	// Single run of clusterization with given seed
	bool runClusterization( IClusteringData* input, int seed, CClusteringResult& result, double& inertia );

	// Initial cluster selection for sparse data
	void selectInitialClusters( const CFloatMatrixDesc& matrix, int seed );
	void defaultInitialization( const CFloatMatrixDesc& matrix, int seed );
	void kMeansPlusPlusInitialization( const CFloatMatrixDesc& matrix, int seed );

	// Sparse data clusterization
	bool clusterize( const CFloatMatrixDesc& matrix, const CArray<double>& weights, double& inertia );

	// Lloyd algorithm implementation for sparse data
	bool lloydClusterization( const CFloatMatrixDesc& matrix, const CArray<double>& weights, double& inertia );
	void classifyAllData( const CFloatMatrixDesc& matrix, CArray<int>& dataCluster, double& inertia );
	int findNearestCluster( const CFloatMatrixDesc& matrix, int dataIndex, double& inertia ) const;
	void storeClusterCenters( CArray<CClusterCenter>& result );
	bool updateClusters( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
		const CArray<int>& dataCluster, const CArray<CClusterCenter>& oldCenters );

	// Elkan algorithm implementation for sparse data
	bool elkanClusterization( const CFloatMatrixDesc& matrix, const CArray<double>& weights, double& inertia );
	void initializeElkanStatistics( const CFloatMatrixDesc& matrix, CArray<int>& assignments,
		CArray<float>& upperBounds, CVariableMatrix<float>& lowerBounds, CVariableMatrix<float>& clusterDists,
		CArray<float>& closestClusterDist, CArray<float>& moveDistance );
	void computeClustersDists( CVariableMatrix<float>& dists, CArray<float>& closestCluster ) const;
	void assignVectors( const CFloatMatrixDesc& matrix, const CVariableMatrix<float>& clusterDists,
		const CArray<float>& closestClusterDist, CArray<int>& assignments, CArray<float>& upperBounds,
		CVariableMatrix<float>& lowerBounds ) const;
	void updateMoveDistance( const CArray<CClusterCenter>& oldCenters, CArray<float>& moveDistance ) const;
	double updateUpperAndLowerBounds( const CFloatMatrixDesc& matrix,
		const CArray<float>& moveDistance, const CArray<int>& assignments,
		CArray<float>& upperBounds, CVariableMatrix<float>& lowerBounds ) const;
	bool isPruned( const CArray<float>& upperBounds, const CVariableMatrix<float>& lowerBounds,
		const CVariableMatrix<float>& clusterDists, int currentCluster, int clusterToProcess, int id) const;

	// Specific case for dense data with Euclidean metrics and Lloyd algorithm
	bool denseLloydL2Clusterize( IClusteringData* rawData, int seed, CClusteringResult& result, double& inertia );
	// Initial cluster selection
	void selectInitialClusters( const CDnnBlob& data, int seed, CDnnBlob& centers );
	void defaultInitialization( const CDnnBlob& data, int seed, CDnnBlob& centers );
	void kMeansPlusPlusInitialization( const CDnnBlob& data, int seed, CDnnBlob& centers );
	// Lloyd algorithm implementation
	bool lloydBlobClusterization( const CDnnBlob& data, const CDnnBlob& weight,
		CDnnBlob& centers, CDnnBlob& sizes, CDnnBlob& labels, double& inertia );
	double assignClosest( const CDnnBlob& data, const CDnnBlob& squaredData, const CDnnBlob& weight,
		const CDnnBlob& centers, CDnnBlob& labels );
	void recalcCenters( const CDnnBlob& data, const CDnnBlob& weight, const CDnnBlob& labels,
		CDnnBlob& centers, CDnnBlob& sizes );
	void calcClusterVariances( const CDnnBlob& data, const CDnnBlob& labels,
		const CDnnBlob& centers, const CDnnBlob& sizes, CDnnBlob& variances );
};

} // namespace NeoML
