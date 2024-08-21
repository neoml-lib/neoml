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
#include <NeoML/TraditionalML/Clustering.h>
#include <NeoML/TraditionalML/CommonCluster.h>

namespace NeoML {

// The smallest number of vectors in a cluster to consider that the variance is valid
static const int DefaultMinVectorCountForVariance = 4;

// First come clustering algorithm
class NEOML_API CFirstComeClustering : public IClustering {
public:
	struct CParam {
		// The distance function
		TDistanceFunc DistanceFunc;
		// The smallest number of vectors in a cluster to consider that the variance is valid
		int MinVectorCountForVariance;
		// The default variance (for when the number of vectors is smaller than MinVectorCountForVariance)
		double DefaultVariance;
		// The distance threshold for creating a new cluster
		double Threshold;
		// The minimum ratio of elements in a cluster (relative to the total number of vectors)
		double MinClusterSizeRatio;
		// The maximum number of clusters to prevent algorithm divergence in case of great differences in data
		int MaxClusterCount;

		CParam() :
			DistanceFunc( DF_Euclid ),
			MinVectorCountForVariance( DefaultMinVectorCountForVariance ),
			DefaultVariance( 1. ),
			Threshold( 0. ),
			MinClusterSizeRatio( 0.05 ),
			MaxClusterCount( 100 )
		{
		}
	};

	explicit CFirstComeClustering( const CParam& params );
	~CFirstComeClustering() override = default;

	// Sets a text stream for logging processing
	// By default logging is off (set to null to turn off)
	void SetLog( CTextStream* newLog ) { log = newLog; }

	bool Clusterize( const IClusteringData* input, CClusteringResult& result ) override;

private:
	const CParam init; // the clustering parameters
	CTextStream* log; // the logging stream

	void processVector( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
		int vecNum, bool canAddCluster, CObjectArray<CCommonCluster>& clusters );
	void deleteTinyClusters( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
		CObjectArray<CCommonCluster>& clusters );
};

} // namespace NeoML
