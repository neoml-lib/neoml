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

#include <NeoML/TraditionalML/IsoDataClustering.h>
#include <float.h>
#include <NeoML/TraditionalML/CommonCluster.h>
#include <NeoML/TraditionalML/KMeansClustering.h>

namespace NeoML {

//---------------------------------------------------------------------------------------------------------
// CIsoData - public

CIsoDataClustering::CIsoDataClustering( const CParam& _params ) : 
	log( 0 ),
	params( _params )
{
	NeoAssert( params.MaxIterations > 0 );
	NeoAssert( params.InitialClustersCount > 0 );
	NeoAssert( params.MinClusterSize > 0 );

	history.SetBufferSize( params.MaxIterations );
}

CIsoDataClustering::~CIsoDataClustering() = default;

bool CIsoDataClustering::Clusterize( const IClusteringData* input, CClusteringResult& result )
{
	NeoAssert( params.MaxIterations > 0 );
	NeoAssert( params.InitialClustersCount > 0 );

	CFloatMatrixDesc matrix = input->GetMatrix();
	NeoAssert( matrix.Height == input->GetVectorCount() );
	NeoAssert( matrix.Width == input->GetFeaturesCount() );

	CArray<double> weights;
	for( int i = 0; i < input->GetVectorCount(); i++ ) {
		weights.Add( input->GetVectorWeight( i ) );
	}
	history.Empty();
	clusters.DeleteAll();

	if( log != 0 ) {
		*log << "\nISODATA clustering started:\n";
	}

	selectInitialClusters( matrix );

	bool success = false;
	for( int i = 0; i < params.MaxIterations; i++ ) {
		// Distribute all elements into clusters with specified centers and then update the centers
		classifyAllData( matrix, weights );

		addToHistory();

		if( log != 0 ) {
			*log << "\n[Step " << i << "]\nData classification result:\n";
			for( int j = 0; j < clusters.Size(); j++ ) {
				*log << "Cluster " << j << ": \n";
				*log << *clusters[j];
			}
		}

		if( detectLoop() ) {
			success = true;
			break;
		}

		if( i < params.MaxIterations - 1 ) { // on the last step, do nothing
			if( i % 2 == 0 ) {
				splitClusters( matrix, weights );
			} else {
				mergeClusters();
			}
		}
	}

	result.ClusterCount = clusters.Size();
	result.Data.SetSize( matrix.Height );
	result.Clusters.SetBufferSize( clusters.Size() );

	for( int i = 0; i < clusters.Size(); i++ ) {
		CArray<int> elements;
		clusters[i]->GetAllElements( elements );
		for(int j = 0; j < elements.Size(); j++ ) {
			result.Data[elements[j]]=i;
		}
		result.Clusters.Add( clusters[i]->GetCenter() );
	}

	if( log != 0 ) {
		if( success ) {
			*log << "\nSuccessful!\n";
		} else {
			*log << "\nNeed more iterations!\n";
		}
	}

	return success;
}

//---------------------------------------------------------------------------------------------------------
// CIsoData - private

CIsoDataClustering::CIsoDataClustersPair::CIsoDataClustersPair( int index1, int index2, double distance ) :
	Index1( index1 ),
	Index2( index2 ),
	Distance( distance )
{
}

// Selects initial clusters
void CIsoDataClustering::selectInitialClusters( const CFloatMatrixDesc& matrix )
{
	if( !clusters.IsEmpty() ) {
		// Initial cluster centers already defined
		return;
	}

	// If the initial clusters are not defined, use some of the input data
	const int vectorsCount = matrix.Height;
	const int step = max( vectorsCount / params.InitialClustersCount, 1 );
	NeoAssert( step > 0 );
	clusters.SetBufferSize( params.InitialClustersCount );
	for( int i = 0; i < params.InitialClustersCount; i++ ) {
		CFloatVectorDesc desc;
		matrix.GetRow( ( i * step ) % vectorsCount, desc );
		CFloatVector mean( matrix.Width, desc );
		clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( mean ) ) );
	}
}

// Distributes all data into clusters with given centers, then updates the centers
void CIsoDataClustering::classifyAllData( const CFloatMatrixDesc& matrix, const CArray<double>& weights )
{
	for( int i = 0; i < clusters.Size(); i++ ) {
		clusters[i]->Reset();
	}

	const int vectorsCount = matrix.Height;
	for( int i = 0; i < vectorsCount; i++ ) {
		CFloatVectorDesc desc;
		matrix.GetRow( i, desc );
		int clusterIndex = findNearestCluster( desc, clusters );
		clusters[clusterIndex]->Add( i, desc, weights[i] );
	}

	for( int i = clusters.Size() - 1; i >= 0; i-- ) {
		if( clusters[i]->GetElementsCount() >= params.MinClusterSize ) {
			continue;
		}

		// Take the cluster elements and delete the clusters
		CArray<int> elements; // smallest cluster elements' indices
		clusters[i]->GetAllElements( elements );
		clusters.DeleteAt( i );

		for( int j = 0; j < elements.Size(); j++ ) {
			CFloatVectorDesc desc;
			matrix.GetRow( i, desc );
			int clusterIndex = findNearestCluster( desc, clusters );
			clusters[clusterIndex]->Add( elements[j], desc, weights[i] );
		}
	}

	for( int i = 0; i < clusters.Size(); i++ ) {
		clusters[i]->RecalcCenter();
	}
}

// Finds the closest cluster
int CIsoDataClustering::findNearestCluster( const CFloatVectorDesc& vector, const CObjectArray<CCommonCluster>& allClusters ) const
{
	NeoAssert( !allClusters.IsEmpty() );

	int result = 0;
	double distance = allClusters[0]->CalcDistance( vector, DF_Machalanobis );
	for( int i = 1; i < allClusters.Size(); i++ ) {
		double curDistance = allClusters[i]->CalcDistance( vector, DF_Machalanobis );
		if( distance > curDistance ) {
			result = i;
			distance = curDistance;
		}
	}
	return result;
}

// Splits clusters
bool CIsoDataClustering::splitClusters( const CFloatMatrixDesc& matrix, const CArray<double>& weights )
{
	double meanDiameter = calcMeanDiameter();
	bool split = false;
	for( int i = clusters.Size() - 1; i >= 0; i-- ) { // new clusters added to the end
		double clusterDiameter = calcClusterDiameter( *clusters[i] );

		if( clusters.Size() < params.MaxClustersCount
			&& clusterDiameter > params.MaxClusterDiameter
			&& clusterDiameter >= params.MeanDiameterCoef * meanDiameter
			&& clusters[i]->GetElementsCount() > 2 * ( params.MinClusterSize + 1 ) )
		{
			if( splitCluster( matrix, weights, i ) ) {
				split = true;
			}
		}
	}
	return split;
}

// Calculates the average cluster diameter
double CIsoDataClustering::calcMeanDiameter() const
{	
	NeoAssert( clusters.Size() > 0 );

	double ret = 0;
	for( int i = 0; i < clusters.Size(); i++ ) {
		ret += calcClusterDiameter( *clusters[i] );
	}
	return ret / clusters.Size();
}

// Calculates the cluster diameter
double CIsoDataClustering::calcClusterDiameter( const CCommonCluster& cluster ) const
{
	double result = 0;
	const CClusterCenter& center = cluster.GetCenter();

	for( int i = 0; i < center.Disp.Size(); i++ ) {
		result += center.Disp[i];
	}

	return result;
}

// Tries to split the given cluster
bool CIsoDataClustering::splitCluster( const CFloatMatrixDesc& matrix, const CArray<double>& weights, int clusterNumber )
{
	NeoAssert( 0 <= clusterNumber && clusterNumber < clusters.Size() );

	CFloatVector firstMeans;
	CFloatVector secondMeans;

	if( !splitByFeature( matrix, weights, clusterNumber, firstMeans, secondMeans ) ) {
		return false;
	}

	if( log != 0 ) {
		*log << "\nSplit cluster " << clusterNumber << ":\n";
		*log << *clusters[clusterNumber];
	}

	CArray<int> elements;
	clusters[clusterNumber]->GetAllElements( elements );

	clusters.ReplaceAt( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( firstMeans ) ), clusterNumber );
	clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( secondMeans ) ) );

	splitData( matrix, weights, elements, clusterNumber, clusters.Size() - 1 );

	NeoAssert( clusters[clusterNumber]->GetElementsCount() > 0 );
	NeoAssert( clusters[clusters.Size() - 1]->GetElementsCount() > 0 );

	if( log != 0 ) {
		*log << "First new cluster:\n";
		*log << *clusters[clusterNumber];
		*log << "Second new cluster:\n";
		*log << *clusters[clusters.Size() - 1];
	}

	return true;
}

// Splits the cluster by the given feature values
bool CIsoDataClustering::splitByFeature(  const CFloatMatrixDesc& matrix, const CArray<double>& weights, int clusterNumber,
	CFloatVector& firstMeans, CFloatVector& secondMeans ) const
{
	CArray<int> elements;
	clusters[clusterNumber]->GetAllElements( elements );
	CArray<CFloatVector> elementsVectors;
	CArray<double> elementsWeight;
	for( int i = 0; i < elements.Size(); i++ ) {
		CFloatVectorDesc desc;
		matrix.GetRow( elements[i], desc );
		CFloatVector vector( matrix.Width, desc );
		elementsVectors.Add( vector );
		elementsWeight.Add( weights[elements[i]] );
	}
	const CFloatVector& means = clusters[clusterNumber]->GetCenter().Mean;
	const CFloatVector& dispSquare = clusters[clusterNumber]->GetCenter().Disp;

	double bestDistance = 0;
	int splitingFeature = NotFound;
	double splitingMeans1 = 0;
	double splitingMeans2 = 0;

	for( int i = 0; i < means.Size(); i++ ) {
		double m = means[i];
		double means1 = m - sqrt( dispSquare[i] / elements.Size() ) * 1.0;
		double means2 = m + sqrt( dispSquare[i] / elements.Size() ) * 1.0;
		double sum1 = 0;
		double sum2 = 0;
		double weight1 = 0;
		double weight2 = 0;

		for( int j = 0; j < elements.Size(); j++ ) {
			double value = elementsVectors[j][i];
			if( value < m ) {
				sum1 += value * elementsWeight[j];
				weight1 += elementsWeight[j];
			} else {
				sum2 += value * elementsWeight[j];
				weight2 += elementsWeight[j];
			}
		}

		if( weight1 >= params.MinClusterSize && weight2 >= params.MinClusterSize ) {
			NeoAssert( weight1 > 0 );
			NeoAssert( weight2 > 0 );

			double mean1 = sum1 / weight1;
			double mean2 = sum2 / weight2;
			double distance = ( mean1 - mean2 ) * ( mean1 - mean2 );

			if( splitingFeature == NotFound || distance > bestDistance ) {
				bestDistance = distance;
				splitingFeature = i;
				splitingMeans1 = means1;
				splitingMeans2 = means2;
			}
		}
	}

	if( splitingFeature == NotFound ) {
		return false;
	}

	firstMeans = means;
	firstMeans.SetAt( splitingFeature, static_cast<float>( splitingMeans1 ) );

	secondMeans = means;
	secondMeans.SetAt( splitingFeature, static_cast<float>( splitingMeans2 ) );

	return true;
}

// Distributes the data between two clusters
void CIsoDataClustering::splitData(  const CFloatMatrixDesc& matrix, const CArray<double>& weights,
	const CArray<int>& dataIndexes, int firstCluster, int secondCluster )
{
	clusters[firstCluster]->Reset();
	clusters[secondCluster]->Reset();

	for( int i = 0; i < dataIndexes.Size(); i++ ) {
		CFloatVectorDesc desc;
		matrix.GetRow( dataIndexes[i], desc );
		double firstDistance = clusters[firstCluster]->CalcDistance( desc, DF_Machalanobis );
		double secondDistance = clusters[secondCluster]->CalcDistance( desc, DF_Machalanobis );

		if( firstDistance < secondDistance ) {
			clusters[firstCluster]->Add( dataIndexes[i], desc, weights[dataIndexes[i]] );
		} else {
			clusters[secondCluster]->Add( dataIndexes[i], desc, weights[dataIndexes[i]] );
		}
	}
}

// Merges clusters that could be merged according to algorithm parameters
void CIsoDataClustering::mergeClusters()
{
	CArray<CIsoDataClustersPair> pairs;
	createPairList( pairs );
	mergePairs( pairs );
}

// Creates a list of cluster pairs that are closest with the given distance function
void CIsoDataClustering::createPairList( CArray<CIsoDataClustersPair>& pairs ) const
{
	NeoAssert( clusters.Size() > 0 );

	const double minDistance = params.MinClustersDistance;
	for( int i = 0; i < clusters.Size(); i++ ) {
		for( int j = i + 1; j < clusters.Size(); j++ ) {
			double distance = clusters[i]->CalcDistance( *clusters[j], DF_Machalanobis );

			if( distance <= minDistance ) {
				CIsoDataClustersPair pair( i, j, distance );
				pairs.Add( pair );
			}
		}
	}
	pairs.QuickSort< AscendingByMember<CIsoDataClustersPair, double, &CIsoDataClustersPair::Distance> >();
}

// Merges pairs of clusters found earlier
void CIsoDataClustering::mergePairs( const CArray<CIsoDataClustersPair>& pairs )
{
	if( pairs.IsEmpty() ) {
		return;
	}

	// Merge the pairs in order of increasing distance between the clusters
	for( int i = 0; i < pairs.Size(); i++ ) {
		const int index1 = pairs[i].Index1;
		const int index2 = pairs[i].Index2;
		if( clusters[index1]->IsEmpty() || clusters[index2]->IsEmpty() ) {
			continue;
		}

		if( log != 0 ) {
			*log << "\nMerge clusters:\n";
			*log << "First cluster::\n";
			*log << *clusters[index1];
			*log << "Second cluster::\n";
			*log << *clusters[index2];
		}

		clusters.Add( FINE_DEBUG_NEW CCommonCluster( *clusters[index1], *clusters[index2] ) );
		clusters[index1]->Reset();
		clusters[index2]->Reset();

		if( log != 0 ) {
			*log << "New cluster:\n";
			*log << *clusters.Last();
		}
	}

	// Delete the empty clusters (whose contents were merged into another cluster)
	int newSize = 0;
	for( int i = 0; i < clusters.Size(); i++ ) {
		if( !clusters[i]->IsEmpty() ) {
			clusters[newSize] = clusters[i];
			newSize++;
		}
	}
	clusters.SetSize( newSize );
}

// Adds the current cluster center list to history
void CIsoDataClustering::addToHistory()
{
	history.Add( FINE_DEBUG_NEW CFloatVectorArray() );

	for( int i = 0; i < clusters.Size(); i++ ) {
		history.Last()->Add( clusters[i]->GetCenter().Mean );
	}
}

// Detects if the algorithm has looped
bool CIsoDataClustering::detectLoop() const
{
	NeoAssert( history.Size() > 0 );
	const CFloatVectorArray* last = history.Last();

	const int lastClustersNumber = last->Size();
	for( int i = history.Size() - 3 ; i >= 0 ; i-- ) { // - 3 because we don't allow two failed splits or merges in a row
		const CFloatVectorArray* current = history[i];
		NeoAssert( current != 0 );
		if( current->Size() < lastClustersNumber ) {
			// The number of clusters is still growing, continue processing
			break;
		}
		if( current->Size() == lastClustersNumber ) {
			bool isEqual = true;
			for( int j = 0; j < lastClustersNumber; j++ ) {
				if( (*current)[j] != (*last)[j] ) {
					isEqual = false;
				}
			}
			if( isEqual ) {
				return true;
			}
		}
	}
	return false;
}

} // namespace NeoML
