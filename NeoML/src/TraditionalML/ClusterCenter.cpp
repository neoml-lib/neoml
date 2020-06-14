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

#include <NeoML/TraditionalML/ClusterCenter.h>

namespace NeoML {

// Distance functions
typedef double (*TClusterDistanceFunc)( const CClusterCenter& first, const CClusterCenter& second );
typedef double (*TVectorDistanceFunc)( const CClusterCenter& cluster, const CFloatVector& element );

// Euclidean distance
static double calcEuclidDistanceVector( const CClusterCenter& cluster, const CFloatVector& element )
{
	NeoAssert( cluster.Mean.Size() == element.Size() );

	double result = 0;
	for( int i = 0; i < element.Size(); i++ ) {
		const double diff = cluster.Mean[i] - element[i];
		result += diff * diff;
	}

	return result;
}

// Euclidean distance
static double calcEuclidDistanceCluster( const CClusterCenter& first, const CClusterCenter& second )
{
	return calcEuclidDistanceVector( first, second.Mean );
}

// Machalanobis distance
static double calcMachalanobisDistanceVector( const CClusterCenter& cluster, const CFloatVector& element )
{
	NeoAssert( cluster.Mean.Size() == element.Size() );

	double result = 0;
	for( int i = 0; i < element.Size(); i++ ) {
		const double diff = cluster.Mean[i] - element[i];
		result += diff * diff / cluster.Disp[i];
	}

	return result;
}

// Machalanobis distance
static double calcMachalanobisDistanceCluster( const CClusterCenter& first, const CClusterCenter& second )
{
	NeoAssert( first.Mean.Size() == second.Mean.Size() );
	NeoAssert( first.Disp.Size() == second.Disp.Size() );

	double result = 0;
	for( int i = 0; i < first.Mean.Size(); i++ ) {
		const double diff = first.Mean[i] - second.Mean[i];
		result += diff * diff / ( first.Disp[i] + second.Disp[i] );
	}

	return result;
}

// Cosine distance
static double calcCosineDistanceWorker( const CClusterCenter& cluster, const CFloatVector& element, double elementNorm )
{
	double dotProduct = DotProduct( cluster.Mean, element );
	return 1. - dotProduct * fabs( dotProduct ) / elementNorm / cluster.Norm;
}

// Cosine distance
static double calcCosineDistanceCluster( const CClusterCenter& first, const CClusterCenter& second )
{
	return calcCosineDistanceWorker( first, second.Mean, second.Norm );
}

// Cosine distance
static double calcCosineDistanceVector( const CClusterCenter& cluster, const CFloatVector& element )
{
	return calcCosineDistanceWorker( cluster, element, DotProduct( element, element ) );
}

// Distance function tables
const static TClusterDistanceFunc clusterDistanceFuncs[DF_Count] =
	{ &calcEuclidDistanceCluster, &calcMachalanobisDistanceCluster, &calcCosineDistanceCluster  };
const static TVectorDistanceFunc vectorDistanceFuncs[DF_Count] =
	{ &calcEuclidDistanceVector, &calcMachalanobisDistanceVector, &calcCosineDistanceVector };

static_assert( DF_Euclid == 0, "DF_Euclid != 0" );
static_assert( DF_Machalanobis == 1, "DF_Machalanobis != 1" );
static_assert( DF_Cosine == 2, "DF_Cosine != 2" );

// Calculates the distance between two cluster centers
double CalcDistance( const CClusterCenter& first, const CClusterCenter& second, TDistanceFunc distanceFunc )
{
	NeoPresume( distanceFunc >= 0 );
	NeoPresume( distanceFunc < DF_Count );

	return (*clusterDistanceFuncs[distanceFunc])( first, second );
}

// Calculates the distance from an element to the cluster center
double CalcDistance( const CClusterCenter& cluster, const CFloatVector& element, TDistanceFunc distanceFunc )
{
	NeoPresume( distanceFunc >= 0 );
	NeoPresume( distanceFunc < DF_Count );

	return (*vectorDistanceFuncs[distanceFunc])( cluster, element );
}

} // namespace NeoML
