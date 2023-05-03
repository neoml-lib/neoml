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
#include <NeoML/TraditionalML/Problem.h>

namespace NeoML {

class CCommonCluster;

// ISODATA clustering algorithm
class NEOML_API CIsoDataClustering : public IClustering {
public:
	// ISODATA algorithm parameters
	struct CParam {
		// The number of initial clusters
		// The initial cluster centers are randomly selected from the input data
		int InitialClustersCount;
		// The maximum number of clusters
		int MaxClustersCount;
		// The minimum cluster size
		int MinClusterSize;
		// The maximum number of algorithm iterations
		int MaxIterations;
		// The minimum distance between the clusters
		// Whenever two clusters are closer they are merged
		double MinClustersDistance;
		// The maximum cluster diameter
		// Whenever a cluster is larger it may be split
		double MaxClusterDiameter;
		// Indicates how much the cluster diameter may exceed 
		// the mean diameter across all the clusters
		// If a cluster diameter is larger than the mean diameter multiplied by this value it may be split
		double MeanDiameterCoef;
	};

	explicit CIsoDataClustering( const CParam& params );
	~CIsoDataClustering() override;

	// Sets a text stream for logging processing
	// By default logging is off (set to null to turn off)
	void SetLog( CTextStream* newLog ) { log = newLog; }

	// IClustering inteface methods:
	// Clusterizes the input data and returns true if successful,
	// false if more iterations are needed
	bool Clusterize( const IClusteringData* input, CClusteringResult& result ) override;

private:
	// A pair of clusters that will be merged
	struct CIsoDataClustersPair {
		int Index1;
		int Index2;
		double Distance;

		CIsoDataClustersPair( int index1, int index2, double distance );
	};

	CTextStream* log; // the logging stream
	CParam params; // clustering parameters
	CObjectArray<CCommonCluster> clusters; // current clusters
	CPointerArray<CFloatVectorArray> history; // algorithm iteration history`

	void selectInitialClusters( const CFloatMatrixDesc& matrix );
	void classifyAllData( const CFloatMatrixDesc& matrix, const CArray<double>& weights );
	int findNearestCluster( const CFloatVectorDesc& vector, const CObjectArray<CCommonCluster>& clusters ) const;

	bool splitClusters( const CFloatMatrixDesc& matrix, const CArray<double>& weights );
	double calcMeanDiameter() const;
	double calcClusterDiameter( const CCommonCluster& cluster ) const;
	bool splitCluster( const CFloatMatrixDesc& matrix, const CArray<double>& weights, int index );
	bool splitByFeature(  const CFloatMatrixDesc& matrix, const CArray<double>& weights, int clusterNumber,
		CFloatVector& firstMeans, CFloatVector& secondMeans ) const;
	void splitData(  const CFloatMatrixDesc& matrix, const CArray<double>& weights, const CArray<int>& dataIndexes,
		int firstCluster, int secondCluster );

	void mergeClusters();
	void createPairList( CArray<CIsoDataClustersPair>& pairs ) const;
	void mergePairs( const CArray<CIsoDataClustersPair>& pairs );

	void addToHistory();
	bool detectLoop() const;
};

} // namespace NeoML
