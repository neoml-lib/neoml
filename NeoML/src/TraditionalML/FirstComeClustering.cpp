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

#include <NeoML/TraditionalML/FirstComeClustering.h>
#include <float.h>

using namespace NeoML;

CFirstComeClustering::CFirstComeClustering( const CParam& params ) :
	init( params ),
	log( 0 )
{
	NeoAssert( init.MaxClusterCount > 0 ); // at least one cluster should exist
	NeoAssert( 0 < init.MinClusterSizeRatio && init.MinClusterSizeRatio <= 1 );
}

bool CFirstComeClustering::Clusterize( const IClusteringData* input, CClusteringResult& result )
{
	result.ClusterCount = 0;
	result.Data.SetSize( input->GetVectorCount() );

	if( log != 0 ) {
		*log << "\nFirst come clustering started:\n";
	}

	CFloatMatrixDesc matrix = input->GetMatrix();
	NeoAssert( matrix.Height == input->GetVectorCount() );
	NeoAssert( matrix.Width == input->GetFeaturesCount() );

	CArray<double> weights;
	for( int i = 0; i < input->GetVectorCount(); i++ ) {
		weights.Add( input->GetVectorWeight( i ) );
	}

	// Cluster list
	CObjectArray<CCommonCluster> clusters;

	// Cluster distribution
	const int vectorsCount = input->GetVectorCount();
	for( int i = 0; i < vectorsCount; ++i ) {
		if( log != 0 ) {
			*log << "\nProcess vector #" << i << "\n";
			*log << "Clusters count: " << clusters.Size() << "\n";
			for( int j = 0; j < clusters.Size(); j++ ) {
				*log << "Cluster " << j << ":\n";
				*log << *clusters[j];
			}
		}

		processVector( matrix, weights, i, true, clusters );
	}

	// Delete the clusters that are too small
	deleteTinyClusters( matrix, weights, clusters );

	// Save the result
	result.ClusterCount = clusters.Size();
	result.Clusters.SetBufferSize( clusters.Size() );
	for( int cluster = 0; cluster < clusters.Size(); ++cluster ) {
		CArray<int> elements;
		clusters[cluster]->GetAllElements( elements );
		for( int i = 0; i < elements.Size(); ++i ) {
			result.Data[elements[i]] = cluster;
		}
		result.Clusters.Add( clusters[cluster]->GetCenter() );
	}

	if( log != 0 ) {
		*log << "\nSuccessful!\n";
	}

	return true;
}

// The initial clustering
void CFirstComeClustering::processVector( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
	int vecNum, bool canAddCluster, CObjectArray<CCommonCluster>& clusters )
{
	NeoPresume( init.MaxClusterCount > 0 ); // at least one cluster should exist

	// Find the nearest cluster
	int minCluster = clusters.Size();
	double minDistance = DBL_MAX;

	CFloatVectorDesc desc;
	matrix.GetRow( vecNum, desc );
	for( int clusterNum = 0; clusterNum < clusters.Size(); ++clusterNum ) {
		double distance = clusters[clusterNum]->CalcDistance( desc, init.DistanceFunc );

		if( distance < minDistance ) {
			minDistance = distance;
			minCluster = clusterNum;
		}
	}

	// Create a new cluster if necessary
	if( canAddCluster && clusters.Size() < init.MaxClusterCount && minDistance >= init.Threshold ) {
		if( log != 0 ) {
			*log << "Create new cluster " << clusters.Size() << " \n";
		}
		CCommonCluster::CParams clusterParams;
		clusterParams.MinElementCountForVariance = init.MinVectorCountForVariance;
		clusterParams.DefaultVariance = init.DefaultVariance;
		CFloatVector mean( matrix.Width, desc );
		clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( mean ), clusterParams ) );
		minCluster = clusters.Size() - 1;
	}

	// Add the vector to the cluster
	clusters[minCluster]->Add( vecNum, desc, weights[vecNum] );
	clusters[minCluster]->RecalcCenter();

	if( log != 0 ) {
		*log << "Vector add to cluster " << minCluster << "\n";
		*log << "Distance: " << minDistance << "\n";
	}
}

// Deletes the clusters that are too small
void CFirstComeClustering::deleteTinyClusters( const CFloatMatrixDesc& matrix, const CArray<double>& weights,
	CObjectArray<CCommonCluster>& clusters )
{
	int threshold = Round( init.MinClusterSizeRatio * matrix.Height );

	for( int cluster = clusters.Size() - 1; cluster >= 0; cluster-- ) {

		if( clusters[cluster]->GetElementsCount() >= threshold ) {
			continue;
		}

		// Get all the vectors
		CArray<int> elements;
		clusters[cluster]->GetAllElements( elements );

		// Delete the cluster
		clusters.DeleteAt( cluster );

		// Redistribute its vectors to other clusters
		for( int vecNum = 0; vecNum < elements.Size(); ++vecNum ) {
			processVector( matrix, weights, elements[vecNum], false, clusters );
		}
	}
}
