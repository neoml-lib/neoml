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

#include <NeoML/TraditionalML/FloatVector.h>
#include <NeoML/TraditionalML/ClusterCenter.h>

namespace NeoML {

class IClusteringData;

// The common cluster representation for all clustering algorithms
class NEOML_API CCommonCluster : public virtual IObject {
public:
	// Parameters of the cluster
	struct CParams {
		// Minimum number of elements in the cluster to account for variance in distance calculation 
		int MinElementCountForVariance;
		// The default variance, for when the number of elements is too small
		double DefaultVariance;

		CParams() : MinElementCountForVariance( 4 ), DefaultVariance( 1.0 ) {}
	};

	// Creates a cluster with the specified center
	CCommonCluster( const CClusterCenter& center, const CParams& params = CParams() );
	// Creates a cluster by merging two clusters
	CCommonCluster( const CCommonCluster& first, const CCommonCluster& second );

	// Adds an element to the cluster
	void Add( int dataIndex, const CSparseFloatVectorDesc& desc, double weight );
	// Checks if the cluster is empty
	bool IsEmpty() const { return elements.IsEmpty(); }
	// Removes all elements of the cluster; the center remains the same
	void Reset();
	// Gets the number of elements
	int GetElementsCount() const { return elements.Size(); }
	// Gets all elements of the cluster
	void GetAllElements( CArray<int>& result ) const { elements.CopyTo( result ); }

	// Gets the cluster center
	const CClusterCenter& GetCenter() const { return center; }
	// Recalculates the cluster center
	void RecalcCenter();

	// Calculates the distance from another cluster (the distance between the centers is counted)
	double CalcDistance( const CCommonCluster& another, TDistanceFunc distanceFunc ) const;

	// Calculate the distance from the cluster center to a given element (not necessarily from this cluster)
	double CalcDistance( const CFloatVector& element, TDistanceFunc distanceFunc ) const;
	double CalcDistance( const CSparseFloatVector& element, TDistanceFunc distanceFunc ) const;
	double CalcDistance( const CSparseFloatVectorDesc& element, TDistanceFunc distanceFunc ) const;

protected:
	virtual ~CCommonCluster() {} // delete operation is prohibited

private:
	const CParams params; // cluster parameters

	CClusterCenter center; // cluster center
	mutable bool isCenterDirty; // shows if the center should be recalculated
	CArray<double> sum; // the sum of coordinates
	CArray<double> sumSquare; // the sum of squared coordinates
	double sumWeight; // the total element weight
	CArray<int> elements; // the list of cluster elements

	friend CTextStream& operator<<( CTextStream& stream, const CCommonCluster& cluster );
};

// Writing into a CTextStream
inline CTextStream& operator<<( CTextStream& stream, const CCommonCluster& cluster )
{
	stream << "Means: ";
	stream << cluster.center.Mean << "\n";
	stream << "Disps: ";
	stream << cluster.center.Disp << "\n";

	stream << "Elements: ( ";
	for( int i = 0; i < cluster.elements.Size(); i++ ) {
		stream << cluster.elements[i] << " ";
	}
	stream << ")\n";

	return stream;
}

inline double CCommonCluster::CalcDistance( const CCommonCluster& another, TDistanceFunc distanceFunc ) const
{
	return NeoML::CalcDistance( center, another.center, distanceFunc );
}

inline double CCommonCluster::CalcDistance( const CFloatVector& element, TDistanceFunc distanceFunc ) const
{
	return NeoML::CalcDistance( center, element, distanceFunc );
}

inline double CCommonCluster::CalcDistance( const CSparseFloatVector& element, TDistanceFunc distanceFunc ) const
{
	return NeoML::CalcDistance( center, element, distanceFunc );
}

inline double CCommonCluster::CalcDistance( const CSparseFloatVectorDesc& element, TDistanceFunc distanceFunc ) const
{
	return NeoML::CalcDistance( center, element, distanceFunc );
}

} // namespace NeoML
