/* Copyright © 2017-2024 ABBYY

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

#include <common.h>
#pragma hdrstop

#include "PyClustering.h"

class CPyClusteringData : public IClusteringData {
public:
	CPyClusteringData( int height, int width, const int* columns, const float* values, const int* rowPtr, const float* _weights ) :
		weights( _weights )
	{
		desc.Height = height;
		desc.Width = width;
		desc.Columns = const_cast<int*>( columns );
		desc.Values = const_cast<float*>( values );
		desc.PointerB = const_cast<int*>( rowPtr );
		desc.PointerE = const_cast<int*>( rowPtr ) + 1;
	}

	// IClusteringData interface methods:
	virtual int GetVectorCount() const { return desc.Height; }
	virtual int GetFeaturesCount() const { return desc.Width; }
	virtual CFloatMatrixDesc GetMatrix() const { return desc; }
	virtual double GetVectorWeight( int index ) const { return weights[index]; };

private:
	CFloatMatrixDesc desc;
	const float* weights;
};

//------------------------------------------------------------------------------------------------------------

class CPyClustering {
public:
	explicit CPyClustering( IClustering* _clustering ) : clustering( _clustering ) {}
	virtual ~CPyClustering() { delete clustering; }

	py::tuple Clusterize( py::array indices, py::array data, py::array rowPtr, bool isSparse, int featureCount, py::array weight );

private:
	IClustering* clustering;
};

py::tuple CPyClustering::Clusterize( py::array indices, py::array data, py::array rowPtr, bool isSparse, int featureCount, py::array weight )
{
	CPtr<const CPyClusteringData> problem = new CPyClusteringData( static_cast<int>( weight.size() ), featureCount,
		reinterpret_cast<const int*>( isSparse ? indices.data() : nullptr ), reinterpret_cast<const float*>( data.data() ),
		reinterpret_cast<const int*>( rowPtr.data() ), reinterpret_cast<const float*>( weight.data() ) );

	CClusteringResult result;

	{
		py::gil_scoped_release release;
		clustering->Clusterize( problem.Ptr(), result );
	}

	py::array_t<int, py::array::c_style> clusters( py::ssize_t{ weight.size() } );
	NeoAssert( weight.size() == clusters.size() );
	auto tempClusters = clusters.mutable_unchecked<1>();
	for( int i = 0; i < result.Data.Size(); i++ ) {
		tempClusters(i) = result.Data[i];
	}

	py::array_t<double, py::array::c_style> clusterCentersMean( { result.ClusterCount, featureCount } );
	auto tempClusterCentersMean = clusterCentersMean.mutable_unchecked<2>();
	py::array_t<double, py::array::c_style> clusterCentersDisp( { result.ClusterCount, featureCount } );
	auto tempClusterCentersDisp = clusterCentersDisp.mutable_unchecked<2>();
	for( int i = 0; i < result.Clusters.Size(); i++ ) {
		for( int j = 0; j < featureCount; j++ ) {
			tempClusterCentersMean(i, j) = result.Clusters[i].Mean[j];
			tempClusterCentersDisp(i, j) = result.Clusters[i].Disp[j];
		}
	}

	auto t = py::tuple(3);
	t[0] = clusters;
	t[1] = clusterCentersMean;
	t[2] = clusterCentersDisp;

	return t;
}

//------------------------------------------------------------------------------------------------------------

class CPyHierarchical : public CPyClustering {
public:
	explicit CPyHierarchical( const CHierarchicalClustering::CParam& p ) : CPyClustering( new CHierarchicalClustering( p ) ) {}
};

//------------------------------------------------------------------------------------------------------------

class CPyFirstCome : public CPyClustering {
public:
	explicit CPyFirstCome( const CFirstComeClustering::CParam& p ) : CPyClustering( new CFirstComeClustering( p ) ) {}
};

//------------------------------------------------------------------------------------------------------------

class CPyIsoData : public CPyClustering {
public:
	explicit CPyIsoData( const CIsoDataClustering::CParam& p ) : CPyClustering( new CIsoDataClustering( p ) ) {}
};

//------------------------------------------------------------------------------------------------------------

class CPyKMeans : public CPyClustering {
public:
	explicit CPyKMeans( const CKMeansClustering::CParam& p ) : CPyClustering( new CKMeansClustering( p ) ) {}
};

void InitializeClustering(py::module& m)
{
	py::class_<CPyHierarchical>(m, "Hierarchical")
		.def( py::init(
			[]( const std::string& distance, float max_cluster_distance, int min_cluster_count, const std::string& linkage ) {
				CHierarchicalClustering::CParam p;
				p.DistanceType = DF_Undefined;
				if( distance == "euclid" ) {
					p.DistanceType = DF_Euclid;
				} else if( distance == "machalanobis" ) {
					p.DistanceType = DF_Machalanobis;
				} else if( distance == "cosine" ) {
					p.DistanceType = DF_Cosine;
				}
				p.MaxClustersDistance = max_cluster_distance;
				p.MinClustersCount = min_cluster_count;
				p.Linkage = CHierarchicalClustering::L_Count;
				if( linkage == "centroid" ) {
					p.Linkage = CHierarchicalClustering::L_Centroid;
				} else if( linkage == "single" ) {
					p.Linkage = CHierarchicalClustering::L_Single;
				} else if( linkage == "average" ) {
					p.Linkage = CHierarchicalClustering::L_Average;
				} else if( linkage == "complete" ) {
					p.Linkage = CHierarchicalClustering::L_Complete;
				} else if( linkage == "ward" ) {
					p.Linkage = CHierarchicalClustering::L_Ward;
				}

				return new CPyHierarchical( p );
			})
		)

		.def( "clusterize", &CPyHierarchical::Clusterize, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyFirstCome>(m, "FirstCome")
		.def( py::init(
			[]( const std::string& distance, int min_vector_count, float default_variance,
				float threshold, float min_cluster_size_ratio, int max_cluster_count )
			{
				CFirstComeClustering::CParam p;
				p.DistanceFunc = DF_Undefined;
				if( distance == "euclid" ) {
					p.DistanceFunc = DF_Euclid;
				} else if( distance == "machalanobis" ) {
					p.DistanceFunc = DF_Machalanobis;
				} else if( distance == "cosine" ) {
					p.DistanceFunc = DF_Cosine;
				}
				p.MinVectorCountForVariance = min_vector_count;
				p.DefaultVariance = default_variance;
				p.Threshold = threshold;
				p.MinClusterSizeRatio = min_cluster_size_ratio;
				p.MaxClusterCount = max_cluster_count;

				return new CPyFirstCome( p );
			})
		)

		.def( "clusterize", &CPyFirstCome::Clusterize, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyIsoData>(m, "IsoData")
		.def( py::init(
			[]( int init_cluster_count, int max_cluster_count, int min_cluster_size, int max_iteration_count,
				float min_cluster_distance, float max_cluster_diameter, float mean_diameter_coef )
			{
				CIsoDataClustering::CParam p;
				p.InitialClustersCount = init_cluster_count;
				p.MaxClustersCount = max_cluster_count;
				p.MinClusterSize = min_cluster_size;
				p.MaxIterations = max_iteration_count;
				p.MinClustersDistance = min_cluster_distance;
				p.MaxClusterDiameter = max_cluster_diameter;
				p.MeanDiameterCoef = mean_diameter_coef;

				return new CPyIsoData( p );
			})
		)

		.def( "clusterize", &CPyIsoData::Clusterize, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyKMeans>(m, "KMeans")
		.def( py::init(
			[]( const std::string& algo, const std::string& init, const std::string& distance,
				int max_iteration_count, int cluster_count, int thread_count, int run_count, int seed )
			{
				CKMeansClustering::CParam p;

				p.Algo = CKMeansClustering::KMA_Count;
				if( algo == "lloyd" ) {
					p.Algo = CKMeansClustering::KMA_Lloyd;
				} else if( algo == "elkan" ) {
					p.Algo = CKMeansClustering::KMA_Elkan;
				}
				p.Initialization = CKMeansClustering::KMI_Count;
				if( init == "default" ) {
					p.Initialization = CKMeansClustering::KMI_Default;
				} else if( init == "k++" ) {
					p.Initialization = CKMeansClustering::KMI_KMeansPlusPlus;
				}

				p.DistanceFunc = DF_Undefined;
				if( distance == "euclid" ) {
					p.DistanceFunc = DF_Euclid;
				} else if( distance == "machalanobis" ) {
					p.DistanceFunc = DF_Machalanobis;
				} else if( distance == "cosine" ) {
					p.DistanceFunc = DF_Cosine;
				}
				p.InitialClustersCount = cluster_count;
				p.MaxIterations = max_iteration_count;
				p.ThreadCount = thread_count;
				p.RunCount = run_count;
				p.Seed = seed;
				return new CPyKMeans( p );
			})
		)

		.def( "clusterize", &CPyKMeans::Clusterize, py::return_value_policy::reference )
	;
}
