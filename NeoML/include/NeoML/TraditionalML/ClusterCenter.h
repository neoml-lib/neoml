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
#include <NeoML/TraditionalML/FloatVector.h>

namespace NeoML {

// The distance function (note that all distance functions return the squared distance)
enum TDistanceFunc {
	DF_Undefined = -1,
	DF_Euclid = 0,			// Euclidean distance
	DF_Machalanobis = 1,	// Machalanobis distance (with a diagonal covariance matrix)
	DF_Cosine = 2,			// cosine distance (for the sake of speed, it is calculated as (1 - cos(x) * |cos(x)|) =
							// (1 - cos(2x))/2 when x < Pi and 1 + (1 - cos(2*(x - Pi/2)))/2 otherwise)

	DF_Count				// the number of distance functions
};

// Cluster center
struct CClusterCenter {
	CFloatVector Mean;
	CFloatVector Disp;
	double Norm;
	double Weight;

	CClusterCenter() : Norm( 0 ), Weight( 1 ) {}
	explicit CClusterCenter( const CFloatVector& mean );
};

inline CClusterCenter::CClusterCenter( const CFloatVector& mean ) :
	Mean( mean ),
	Disp( mean.Size(), 1.0 ),
	Norm( DotProduct( mean, mean ) ),
	Weight( 0 )
{
}

inline CArchive& operator << ( CArchive& archive, const CClusterCenter& center )
{
	archive << center.Mean;
	archive << center.Disp;
	archive << center.Norm;
	archive << center.Weight;
	return archive;
}

inline CArchive& operator >> ( CArchive& archive, CClusterCenter& center )
{
	archive >> center.Mean;
	archive >> center.Disp;
	archive >> center.Norm;
	archive >> center.Weight;
	return archive;
}

// Writing into a CTextStream
inline CTextStream& operator<<( CTextStream& stream, const CClusterCenter& cluster )
{
	stream << "Means: ";
	stream << cluster.Mean;
	stream << "\n";
	stream << "Disps: ";
	stream << cluster.Disp << "\n";
	stream << "Weight: ";
	stream << cluster.Weight << "\n";
	return stream;
}

// Calculates the distance between cluster centers
double NEOML_API CalcDistance( const CClusterCenter& first, const CClusterCenter& second, TDistanceFunc distanceFunc );

// Calculates the distance between the cluster center and the given element
double NEOML_API CalcDistance( const CClusterCenter& cluster, const CFloatVector& element, TDistanceFunc distanceFunc );

// Calculates the distance between the cluster center and the given element
inline double CalcDistance( const CClusterCenter& cluster, const CSparseFloatVector& element, TDistanceFunc distanceFunc )
{
	return CalcDistance( cluster, CFloatVector( cluster.Mean.Size(), element ), distanceFunc );
}

// Calculates the distance between the cluster center and the given element
inline double CalcDistance( const CClusterCenter& cluster, const CSparseFloatVectorDesc& element, TDistanceFunc distanceFunc )
{
	return CalcDistance( cluster, CFloatVector( cluster.Mean.Size(), element ), distanceFunc );
}

} // namespace NeoML
