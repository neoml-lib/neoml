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

struct CEmClusterizeResult;

// The general expectation-maximization (EM) algorithm
// Finds the optimal cluster count using BIC (Bayesian information criterion)
class NEOML_API CEMClustering : public IClustering {
public:
	// EM algorithm parameters
	struct NEOML_API CParam {
		// The number of rough EM algorithm runs with fixed cluster count
		// In general, a few runs are enough to determine cluster count
		int RoughInitCount;
		// The number of runs after the cluster count was determined 
		int FinalInitCount;
		// The maximum number of iterations with fixed cluster count
		int MaxFixedEmIteration;
		// The algorithm convergence threshold
		double ConvThreshold;
		// The initial number of clusters
		int InitialClustersCount;
		// The maximum number of clusters
		int MaxClustersCount;
		// The minimum cluster size
		int MinClusterSize;
		// The initial cluster centers. Should either contain InitialClustersCount cluster centers or be empty
		// (in which case the clusters will be initialized by the algorithm)
		CArray<CClusterCenter> InitialClusters;

		CParam();
		CParam( const CParam& result );
	};

	CEMClustering( const CParam& clusteringParams );
	virtual ~CEMClustering();

	// Sets a text stream for logging processing
	// By default logging is off (set to null to turn off)
	void SetLog( CTextStream* newLog ) { log = newLog; }

	// IClustering interface methods:
	// Returns true if the algorithm has converged and no clusters are smaller than MinClusterSize
	virtual bool Clusterize( IClusteringData* input, CClusteringResult& result );

private:
	// EM clustering result
	struct CEmClusteringResult {
		double Likelihood; // mixture likelihood
		double Bic; // the Bayesian information criterion (the smaller it is, the better)
		double Aic; // the Akaike information criterion (the smaller it is, the better)
		bool IsGood; // the result is good if the algorithm has converged and no clusters are smaller than MinClusterSize
		CClusteringResult Result; // the clustering result

		CEmClusteringResult();
		CEmClusteringResult( const CEmClusteringResult& result );

		CEmClusteringResult& operator=( const CEmClusteringResult& );
	};

	const CParam params; // the algorithm parameters
	CTextStream* log; // the logging stream
	CArray<CEmClusteringResult> history; // the history of operations
	
	CArray<CClusterCenter> clusters; // the current cluster centers
	// The hidden variables of the EM algorithm
	// hiddenVars[i][j] is the a posteriori probability that the i-th object was obtained from the j-th mixture component
	CFloatVectorArray hiddenVars;
	// The cluster densities
	// densitiesArgs[i][j] is the logarithm of cluster j density on the image i, multiplied by the cluster j weight.
	CFloatVectorArray densitiesArgs;

	void runEMFixedComponents( const CSparseFloatMatrixDesc& data, const CArray<double>& weights, int clustersCount,
		int iterationsCount, bool goodOnly, CEmClusteringResult& result );
	void calculateInitialClusters( const CSparseFloatMatrixDesc& data, int clustersCount,
		CArray<CClusterCenter>& initialClusters ) const;
	void recalculateInitialClusters( const CSparseFloatMatrixDesc& data, const CEmClusteringResult& prevResult,
		CArray<CClusterCenter>& initialClusters ) const;
	void initCumulativeFitnesses( const CArray<CClusterCenter>& initialClusters,
		CFastArray<double, 1>& cumulativeFitnesses ) const;

	int selectRandomCluster( const CFastArray<double, 1>& cumulativeFitnesses ) const;
	void findBestResult( const CSparseFloatMatrixDesc& data, const CArray<double>& weights, CEmClusteringResult& result );

	void clusterize( const CSparseFloatMatrixDesc& data, const CArray<double>& weights, const CArray<CClusterCenter>& clusters,
		CEmClusteringResult& result );
	void expectation();
	void maximization( const CSparseFloatMatrixDesc& data, const CArray<double>& weights );
	void calculateNewWeights();
	void calculateNewMeans( const CArray<CFloatVector>& vectors, const CArray<double>& vectorsWeight, double sumWeight );
	void calculateNewDisps( const CArray<CFloatVector>& vectors, const CArray<double>& vectorsWeight, double sumWeight );
	void calculateDensitiesArgs( const CSparseFloatMatrixDesc& data );
	double calculateDistance( int clusterIndex, const CSparseFloatVectorDesc& element ) const;

	void calculateResult( const CSparseFloatMatrixDesc& data, bool isConverged, CEmClusteringResult& result ) const;
	double calculateLogOfMixtureLikelihood() const;
};

} // namespace NeoML
