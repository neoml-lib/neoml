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

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/HierarchicalClustering.h>
#include <NaiveHierarchicalClustering.h>
#include <NnChainHierarchicalClustering.h>

namespace NeoML {

// --------------------------------------------------------------------------------------------------------------------

CHierarchicalClustering::CHierarchicalClustering( const CArray<CClusterCenter>& clustersCenters, const CParam& _params ) :
	params( _params ),
	log( 0 )
{
	NeoAssert( params.MinClustersCount > 0 );
	// Initial cluster centers are supported only for centroid
	NeoAssert( params.Linkage == L_Centroid );
	clustersCenters.CopyTo( initialClusters );
}

CHierarchicalClustering::CHierarchicalClustering( const CParam& _params ) :
	params( _params ),
	log( 0 )
{
	NeoAssert( params.MinClustersCount > 0 );
}

bool CHierarchicalClustering::Clusterize( const IClusteringData* input, CClusteringResult& result )
{
	return clusterizeImpl( input, result, nullptr, nullptr );
}

bool CHierarchicalClustering::ClusterizeEx( const IClusteringData* input, CClusteringResult& result,
	CArray<CMergeInfo>& dendrogram, CArray<int>& dendrogramIndices )
{
	return clusterizeImpl( input, result, &dendrogram, &dendrogramIndices );
}

bool CHierarchicalClustering::clusterizeImpl( const IClusteringData* data, CClusteringResult& result,
	CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices ) const
{
	NeoAssert( params.Linkage != L_Ward || params.DistanceType == DF_Euclid ); // Ward works only in L2
	NeoAssert( ( dendrogram != nullptr && dendrogramIndices != nullptr )
		|| ( dendrogram == nullptr && dendrogramIndices == nullptr ) );
	NeoAssert( data != 0 );

	if( log != 0 ) {
		*log << "\nHierarchical clustering started:\n";
	}

	CFloatMatrixDesc matrix = data->GetMatrix();
	NeoAssert( matrix.Height == data->GetVectorCount() );
	NeoAssert( matrix.Width == data->GetFeaturesCount() );

	CArray<double> weights;
	for( int i = 0; i < data->GetVectorCount(); i++ ) {
		weights.Add( data->GetVectorWeight( i ) );
	}

	static_assert( L_Count == 5, "L_Count != 5" );
	bool success = false;
	switch( params.Linkage ) {
		case L_Centroid:
			success = naiveAlgo( matrix, weights, result, dendrogram, dendrogramIndices );
			break;
		case L_Single:
		case L_Average:
		case L_Complete:
		case L_Ward:
			if( initialClusters.IsEmpty() ) {
				success = nnChainAlgo( matrix, weights, result, dendrogram, dendrogramIndices );
			} else {
				success = naiveAlgo( matrix, weights, result, dendrogram, dendrogramIndices );
			}
			break;
		default:
			NeoAssert( false );
	}

	if( log != 0 ) {
		if( success ) {
			*log << "\nSuccessful!\n";
		} else {
			*log << "\nMaxClustersDistance is too small!\n";
		}
	}

	return success;
}

// Clusterizes data by using naive algorithm
bool CHierarchicalClustering::naiveAlgo( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
	CClusteringResult& result, CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices ) const
{
	CNaiveHierarchicalClustering naiveClustering( params, initialClusters, log );
	return naiveClustering.Clusterize( matrix, weights, result, dendrogram, dendrogramIndices );
}

// Clusterizes data by using nearest neighbor chain algorithm
bool CHierarchicalClustering::nnChainAlgo(const CFloatMatrixDesc& matrix, const CArray<double>& weights,
	CClusteringResult& result, CArray<CMergeInfo>* dendrogram, CArray<int>* dendrogramIndices ) const
{
	NeoAssert( params.Linkage != L_Centroid );
	NeoAssert( initialClusters.IsEmpty() );
	CNnChainHierarchicalClustering nnChainClustering( params, log );
	return nnChainClustering.Clusterize( matrix, weights, result, dendrogram, dendrogramIndices );
}

} // namespace NeoML
