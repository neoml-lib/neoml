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

#include <NeoML/TraditionalML/KMeansClustering.h>
#include <NeoML/TraditionalML/CommonCluster.h>
#include <NeoML/TraditionalML/VariableMatrix.h>
#include <NeoML/Random.h>
#include <NeoMathEngine/OpenMP.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/DnnBlob.h>
#include <NeoML/Dnn/Dnn.h>
#include <float.h>
#include <memory>

namespace NeoML {

static CPtr<CDnnBlob> createDataBlob( IMathEngine& mathEngine, const CFloatVectorArray& data )
{
	NeoAssert( !data.IsEmpty() );
	const int vectorCount = data.Size();
	const int featureCount = data[0].Size();
	CPtr<CDnnBlob> result = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, vectorCount, featureCount );
	float* buffer = result->GetBuffer<float>( 0, vectorCount * featureCount );
	float* currPtr = buffer;
	for( int i = 0; i < data.Size(); ++i ) {
		::memcpy( currPtr, data[i].GetPtr(), featureCount * sizeof( float ) );
		currPtr += featureCount;
	}
	result->ReleaseBuffer( buffer, true );
	return result;
}

CKMeansClustering::CKMeansClustering( const CArray<CClusterCenter>& _clusters, const CParam& _params ) :
	params( _params ),
	log( 0 )
{
	NeoAssert( !_clusters.IsEmpty() );
	NeoAssert( _clusters.Size() == params.InitialClustersCount );

	_clusters.CopyTo( initialClusterCenters );
}

CKMeansClustering::CKMeansClustering( const CParam& _params ) :
	params( _params ),
	log( 0 )
{
}

bool CKMeansClustering::Clusterize( IClusteringData* input, CClusteringResult& result )
{
	NeoAssert( input != 0 );

	CSparseFloatMatrixDesc matrix = input->GetMatrix();
	NeoAssert( matrix.Height == input->GetVectorCount() );
	NeoAssert( matrix.Width == input->GetFeaturesCount() );

	CArray<double> weights;
	for( int i = 0; i < input->GetVectorCount(); i++ ) {
		weights.Add( input->GetVectorWeight( i ) );
	}

	if( log != 0 ) {
		*log << "\nK-means clustering started:\n";
	}

	selectInitialClusters( matrix );

	if( log != 0 ) {
		*log << "Initial clusters:\n";

		for( int i = 0; i < clusters.Size(); i++ ) {
			*log << *clusters[i] << "\n";
		}
	}

	bool success = clusterize( matrix, weights );

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

bool CKMeansClustering::Clusterize( const CFloatVectorArray& rawData, CClusteringResult& result )
{
	NeoAssert( params.DistanceFunc == DF_Euclid );
	NeoAssert( params.Algo == KMA_Lloyd );
	NeoAssert( rawData.Size() > params.InitialClustersCount );
	const int vectorCount = rawData.Size();
	const int featureCount = rawData[0].Size();
	const int clusterCount = params.InitialClustersCount;

	if( log != 0 ) {
		*log << L"\nK-means clustering started:\n";
	}

	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( params.ThreadCount, 0 ) );

	CPtr<CDnnBlob> data = createDataBlob( *mathEngine, rawData );
	CPtr<CDnnBlob> centers = CDnnBlob::CreateDataBlob( *mathEngine, CT_Float, 1, clusterCount, featureCount );

	selectInitialClusters( *data, *centers );

	if( log != 0 ) {
		*log << L"Initial clusters:\n";

		for( int i = 0; i < clusters.Size(); i++ ) {
			*log << *clusters[i] << L"\n";
		}
	}

	bool success = false;

	CPtr<CDnnBlob> sizes = CDnnBlob::CreateVector( *mathEngine, CT_Int, clusterCount );
	CPtr<CDnnBlob> labels = CDnnBlob::CreateVector( *mathEngine, CT_Int, vectorCount );

	compileTimeAssert( KMA_Count == 2 );
	switch( params.Algo ) {
		case KMA_Lloyd:
			success = lloydBlobClusterization( *data, *centers, *sizes, *labels );
			break;
		case KMA_Elkan:
			// Не поддерживается для блобов.
			// Elkan работает только для IClusteringData.
		default:
			NeoAssert( false );
	}

	//	оформление результата кластеризации
	result.ClusterCount = clusterCount;
	result.Data.SetSize( vectorCount );
	labels->CopyTo( result.Data.GetPtr() );

	CPtr<CDnnBlob> variances = CDnnBlob::CreateDataBlob( *mathEngine, CT_Float, 1, clusterCount, featureCount );
	calcClusterVariances( *data, *labels, *centers, *sizes, *variances );
	
	CConstFloatHandle rawCentersPtr = centers->GetData();
	CConstFloatHandle rawVariancesPtr = variances->GetData();

	result.Clusters.SetBufferSize( result.ClusterCount );

	int* clusterSizes = sizes->GetBuffer<int>( 0, clusterCount );
	for( int i = 0, id = 0; i < clusterCount; i++ ) {
		//	пустые кластеры пропускаем
		if( clusterSizes[i] > 0 ) {
			CFloatVector center( featureCount );
			CFloatVector variance( featureCount );

			mathEngine->DataExchangeTyped( center.CopyOnWrite(), rawCentersPtr, featureCount );
			mathEngine->DataExchangeTyped( variance.CopyOnWrite(), rawVariancesPtr, featureCount );

			CClusterCenter& currentCenter = result.Clusters.Append();
			currentCenter.Mean = center;
			currentCenter.Disp = variance;
			currentCenter.Norm = DotProduct( currentCenter.Mean, currentCenter.Mean );
			//	у нас могли быть пустые кластеры, поэтому придется смещать метки.
			for( int j = 0; j < result.Data.Size(); j++ ) {
				if( result.Data[j] == i ) {
					result.Data[j] = id;
				}
			}
			id++;
		}
		rawCentersPtr += featureCount;
		rawVariancesPtr += featureCount;
	}
	sizes->ReleaseBuffer( clusterSizes, false );

	result.ClusterCount = result.Clusters.Size();
	assert( result.Clusters.Size() > 0 );
	return success;
}

// Selects the initial clusters
void CKMeansClustering::selectInitialClusters( const CSparseFloatMatrixDesc& matrix )
{
	if( !clusters.IsEmpty() ) {
		// The initial clusters have been set already
		return;
	}
	
	// If the initial cluster centers have been specified, create the clusters from that
	if( !initialClusterCenters.IsEmpty() ) {
		clusters.SetBufferSize( params.InitialClustersCount );
		for( int i = 0; i < initialClusterCenters.Size(); i++ ) {
			clusters.Add( FINE_DEBUG_NEW CCommonCluster( initialClusterCenters[i] ) );
		}
		return;
	}

	if( params.Initialization == KMI_Default ) {
		defaultInitialization( matrix );
	} else if( params.Initialization == KMI_KMeansPlusPlus ) {
		kMeansPlusPlusInitialization( matrix );
	} else {
		NeoAssert( false );
	}
}

void CKMeansClustering::defaultInitialization( const CSparseFloatMatrixDesc& matrix )
{
	const int vectorCount = matrix.Height;
	// If the cluster centers have not been specified, use some elements of the input data
	const int step = max( vectorCount / params.InitialClustersCount, 1 );
	NeoAssert( step > 0 );
	clusters.SetBufferSize( params.InitialClustersCount );
	for( int i = 0; i < params.InitialClustersCount; i++ ) {
		CSparseFloatVectorDesc desc;
		matrix.GetRow( ( i * step ) % vectorCount, desc );
		CFloatVector mean( matrix.Width, desc );
		clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( mean ) ) );
	}
}

void CKMeansClustering::kMeansPlusPlusInitialization( const CSparseFloatMatrixDesc& matrix )
{
	const int vectorCount = matrix.Height;
	NeoAssert( params.InitialClustersCount <= vectorCount );

	// Use random element as the first center
	CRandom random( 0xCEA );
	const int firstCenterIndex = random.UniformInt( 0, vectorCount - 1 );
	CFloatVector firstCenter( matrix.Width, matrix.GetRow( firstCenterIndex ) );
	clusters.Add( FINE_DEBUG_NEW CCommonCluster( CClusterCenter( firstCenter ) ) );

	CArray<double> dists;
	dists.Add( HUGE_VAL, vectorCount );

	while( clusters.Size() < params.InitialClustersCount ) {
		double distSum = 0;
		for( int i = 0; i < vectorCount; ++i ) {
			dists[i] = min( dists[i], clusters.Last()->CalcDistance( matrix.GetRow( i ), params.DistanceFunc ) );
			if( params.DistanceFunc == DF_Cosine ) {
				dists[i] *= dists[i];
			}
			distSum += dists[i];
		}

		double selectedSum = random.Uniform( 0, distSum );
		double prefixSum = 0;
		int nextCenterIndex = 0;

		while( prefixSum + dists[nextCenterIndex] < selectedSum ) {
			prefixSum += dists[nextCenterIndex++];
		}

		CFloatVector nextCenter( matrix.Width, matrix.GetRow( nextCenterIndex ) );
		clusters.Add( new CCommonCluster( CClusterCenter( nextCenter ) ) );
	}
}

bool CKMeansClustering::clusterize( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights )
{
	if( params.Algo == KMA_Lloyd ) {
		return lloydClusterization( matrix, weights );
	} else {
		return elkanClusterization( matrix, weights );
	}
}

bool CKMeansClustering::lloydClusterization( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights )
{
	CArray<int> dataCluster; // the cluster for this element
	dataCluster.SetBufferSize( matrix.Height );
	bool success = false;
	for( int i = 0; i < params.MaxIterations; i++ ) {
		classifyAllData( matrix, dataCluster );

		if( log != 0 ) {
			*log << "\n[Step " << i << "]\nData classification result:\n";
			for( int j = 0; j < clusters.Size(); j++ ) {
				*log << "Cluster " << j << ": \n";
				*log << *clusters[j];
			}
		}

		CArray<CClusterCenter> oldCenters;
		storeClusterCenters( oldCenters );
		if( !updateClusters( matrix, weights, dataCluster, oldCenters ) ) {
			// Cluster centers stay the same, no need to continue
			success = true;
			break;
		}
	}

	return success;
}

// Distributes all elements over the existing clusters
void CKMeansClustering::classifyAllData( const CSparseFloatMatrixDesc& matrix, CArray<int>& dataCluster )
{
	// Each element is assigned to the nearest cluster
	dataCluster.SetSize( matrix.Height );
	NEOML_OMP_NUM_THREADS( params.ThreadCount ) {
		int firstVector = 0;
		int vectorCount = 0;
		if( OmpGetTaskIndexAndCount( matrix.Height, firstVector, vectorCount ) ) {
			const int lastVector = firstVector + vectorCount;
			for( int i = firstVector; i < lastVector; i++ ) {
				dataCluster[i] = findNearestCluster( matrix, i );
			}
		}
	}
}

// Finds the nearest cluster for the element
int CKMeansClustering::findNearestCluster( const CSparseFloatMatrixDesc& matrix, int dataIndex ) const
{
	double bestDistance = DBL_MAX;
	int res = NotFound;
	
	for( int i = 0; i < clusters.Size(); i++ ) {
		CSparseFloatVectorDesc desc;
		matrix.GetRow( dataIndex, desc );
		const double distance = clusters[i]->CalcDistance( desc, params.DistanceFunc );
		if( distance < bestDistance ) {
			bestDistance = distance;
			res = i;
		}
	}

	NeoAssert( res != NotFound );
	return res;
}

void CKMeansClustering::storeClusterCenters( CArray<CClusterCenter>& result )
{
	result.SetBufferSize( clusters.Size() );
	for( int i = 0; i < clusters.Size(); ++i ) {
		result.Add( clusters[i]->GetCenter() );
	}
}

// Updates the clusters and returns true if the clusters were changed, false if they stayed the same
bool CKMeansClustering::updateClusters( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights,
	const CArray<int>& dataCluster, const CArray<CClusterCenter>& oldCenters )
{
	// Store the old cluster centers
	for( int i = 0; i < clusters.Size(); i++ ) {
		clusters[i]->Reset();
	}

	// Update the cluster contents
	for( int i = 0; i < dataCluster.Size(); i++ ) {
		CSparseFloatVectorDesc desc;
		matrix.GetRow( i, desc );
		clusters[dataCluster[i]]->Add( i, desc, weights[i] );
	}

	// Update the cluster centers
	for( int i = 0; i < clusters.Size(); i++ ) {
		if( clusters[i]->GetElementsCount() > 0 ) {
			clusters[i]->RecalcCenter();
		}
	}

	// Compare the new cluster centers with the old
	for( int i = 0; i < clusters.Size(); i++ ) {
		if( oldCenters[i].Mean != clusters[i]->GetCenter().Mean ) {
			return true;
		}
	}

	return false;
}

bool CKMeansClustering::elkanClusterization( const CSparseFloatMatrixDesc& matrix, const CArray<double>& weights )
{
	// Metric must support triangle inequality
	NeoAssert( params.DistanceFunc == DF_Euclid );

	// Distances bounds
	CArray<float> upperBounds; // upper bounds for every object (objectCount)
	CVariableMatrix<float> lowerBounds; // lower bounds for every object and every cluster (clusterCount x objectCount)
	// Distances between old and updated centers of each cluster (clusterCount)
	CArray<float> moveDistance;
	// Distances between clusters (clusterCount x clusteCount)
	CVariableMatrix<float> clusterDists;
	// Distance to the closest center of another cluster (clusterCount)
	CArray<float> closestClusterDist;
	// Element assignments (objectCount)
	CArray<int> assignments;

	initializeElkanStatistics( matrix, assignments, upperBounds, lowerBounds, clusterDists,
		closestClusterDist, moveDistance );

	double lastResidual = DBL_MAX;
	double inertia = DBL_MAX;
	for( int i = 0; i < params.MaxIterations; i++ ) {
		// Calculaate pairwise and closest cluster distances
		computeClustersDists( clusterDists, closestClusterDist );
		// Reassign vectors
		assignVectors( matrix, clusterDists, closestClusterDist, assignments,
			upperBounds, lowerBounds );
		// Recalculate centers
		CArray<CClusterCenter> oldCenters;
		storeClusterCenters( oldCenters );
		updateClusters( matrix, weights, assignments, oldCenters );
		// Update move distances
		updateMoveDistance( oldCenters, moveDistance );
		// Update bounds based on move distance
		inertia = updateUpperAndLowerBounds( matrix, moveDistance, assignments, upperBounds, lowerBounds );
		// Check stop criteria
		if( abs( inertia - lastResidual ) <= params.Tolerance ) {
			return true;
		}
		lastResidual = inertia;
		if( log != 0 ) {
			*log << L"Step " << i << L"Itertia: " << inertia << L"\n";
		}
	}

	return false;
}

// Initializes all required statistics for Elkan algorithm
void CKMeansClustering::initializeElkanStatistics( const CSparseFloatMatrixDesc& matrix,
	CArray<int>& assignments, CArray<float>& upperBounds, CVariableMatrix<float>& lowerBounds,
	CVariableMatrix<float>& clusterDists, CArray<float>& closestClusterDist, CArray<float>& moveDistance )
{
	// initialize mapping between elements and cluster index
	assignments.DeleteAll();
	assignments.Add( 0, matrix.Height );

	// initialize bounds
	upperBounds.DeleteAll();
	upperBounds.Add( FLT_MAX, matrix.Height );
	lowerBounds.SetSize( params.InitialClustersCount, matrix.Height );
	lowerBounds.Set( 0.f );

	// initialize pairwise cluster distances
	clusterDists.SetSize( params.InitialClustersCount, params.InitialClustersCount );
	clusterDists.Set( FLT_MAX );

	// initialize closest cluster distances
	closestClusterDist.DeleteAll();
	closestClusterDist.Add( FLT_MAX, params.InitialClustersCount );

	// initialize move distances
	moveDistance.DeleteAll();
	moveDistance.Add( 0, params.InitialClustersCount );
}

void CKMeansClustering::computeClustersDists( CVariableMatrix<float>& dists, CArray<float>& closestCluster ) const
{
	for( int i = 0; i < clusters.Size(); i++ ) {
		closestCluster[i] = FLT_MAX;
	}
	for( int i = 0; i < clusters.Size() - 1; i++ ) {
		dists( i, i ) = FLT_MAX;
		for( int j = i + 1; j < clusters.Size(); j++ ) {
			const float dist = static_cast<float>(
				sqrt( clusters[i]->CalcDistance( *clusters[j], params.DistanceFunc ) ) );
			dists( i, j ) = dist;
			dists( j, i ) = dist;
			closestCluster[i] = min( 0.5f * dist, closestCluster[i] );
			closestCluster[j] = min( 0.5f * dist, closestCluster[j] );
		}
	}
}

void CKMeansClustering::assignVectors( const CSparseFloatMatrixDesc& matrix, const CVariableMatrix<float>& clusterDists,
	const CArray<float>& closestClusterDist, CArray<int>& assignments, CArray<float>& upperBounds,
	CVariableMatrix<float>& lowerBounds ) const
{
	NeoAssert( assignments.Size() == matrix.Height );
	NeoAssert( lowerBounds.SizeX() == params.InitialClustersCount );
	NeoAssert( upperBounds.Size() == assignments.Size() );
	NEOML_OMP_NUM_THREADS( params.ThreadCount ) {
		int firstVector = 0;
		int vectorCount = 0;
		if( OmpGetTaskIndexAndCount( matrix.Height, firstVector, vectorCount ) ) {
			const int lastVector = firstVector + vectorCount;
			for( int i = firstVector; i < lastVector; i++ ) {
				bool mustRecalculate = true;
				if( upperBounds[i] <= closestClusterDist[assignments[i]] ) {
					continue;
				} else {
					for( int c = 0; c < clusters.Size(); c++ ) {
						if( isPruned( upperBounds, lowerBounds, clusterDists, assignments[i], c, i ) ) {
							continue;
						}
						float dist = upperBounds[i];
						if( mustRecalculate ) {
							mustRecalculate = false;
							dist = static_cast<float>(
								sqrt( clusters[assignments[i]]->CalcDistance( matrix.GetRow( i ),
									params.DistanceFunc ) ) );
							lowerBounds( assignments[i], i ) = dist;
							upperBounds[i] = dist;
						}
						if( dist > lowerBounds( c, i ) || dist > 0.5 * clusterDists( assignments[i], c ) ) {
							const float pointDist = static_cast<float>(
								sqrt( clusters[c]->CalcDistance( matrix.GetRow( i ),
									params.DistanceFunc ) ) );
							lowerBounds( c, i ) = pointDist;
							if( pointDist < dist ) {
								upperBounds[i] = pointDist;
								assignments[i] = c;
							}
						}
					}
				}
			}
		}
	}
}

void CKMeansClustering::updateMoveDistance( const CArray<CClusterCenter>& oldCenters, CArray<float>& moveDistance ) const
{
	for( int i = 0; i < clusters.Size(); ++i ) {
		const double moveNorm = sqrt( clusters[i]->CalcDistance( oldCenters[i].Mean, params.DistanceFunc ) );
		moveDistance[i] = static_cast<float>( moveNorm );
	}
}

double CKMeansClustering::updateUpperAndLowerBounds( const CSparseFloatMatrixDesc& matrix,
	const CArray<float>& moveDistance, const CArray<int>& assignments,
	CArray<float>& upperBounds, CVariableMatrix<float>& lowerBounds ) const
{
	CFastArray<double, 16> localInertia;
	localInertia.Add( 0., params.ThreadCount );

	NEOML_OMP_NUM_THREADS( params.ThreadCount ) {
		const int threadIndex = OmpGetThreadNum();
		int firstVector = 0;
		int vectorCount = 0;
		if( OmpGetTaskIndexAndCount( matrix.Height, firstVector, vectorCount ) ) {
			const int lastVector = firstVector + vectorCount;
			for( int j = firstVector; j < lastVector; j++ ) {
				for( int c = 0; c < clusters.Size(); c++ ) {
					lowerBounds( c, j ) = max( lowerBounds( c, j ) - moveDistance[c], 0.f );
				}
				upperBounds[j] += moveDistance[assignments[j]];
				localInertia[threadIndex] += clusters[assignments[j]]->CalcDistance( matrix.GetRow( j ), params.DistanceFunc );
			}
		}
	}

	double inertia = 0;
	for( int i = 0; i < localInertia.Size(); ++i ) {
		inertia += localInertia[i];
	}

	return inertia;
}

// Checks if operation can be omitted
// If it's true element 'id' can't be reassigned to the cluster 'clusterToProcess'
// and all of the calculations related to this case can be skipped
bool CKMeansClustering::isPruned( const CArray<float>& upperBounds, const CVariableMatrix<float>& lowerBounds,
	const CVariableMatrix<float>& clusterDists, int currentCluster, int clusterToProcess, int id ) const
{
	return ( currentCluster == clusterToProcess ) ||
		( upperBounds[id] <= lowerBounds( clusterToProcess, id ) ) ||
		( upperBounds[id] <= 0.5 * clusterDists( currentCluster, clusterToProcess ) );
}

// Выбирает набор начальных кластеров.
void CKMeansClustering::selectInitialClusters( const CDnnBlob& data, CDnnBlob& centers )
{
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();
	if( !initialClusterCenters.IsEmpty() ) {
		float* buffer = centers.GetBuffer<float>( 0, params.InitialClustersCount * featureCount );
		float* currPtr = buffer;
		for( int i = 0; i < params.InitialClustersCount; ++i ) {
			::memcpy( currPtr, initialClusterCenters[i].Mean.GetPtr(), featureCount * sizeof( float ) );
			currPtr += featureCount;
		}
		centers.ReleaseBuffer( buffer, true );
		return;
	}

	compileTimeAssert( KMI_Count == 2 );
	switch( params.Initialization ) {
		case KMI_Default:
			defaultInitialization( data, centers );
			break;
		case KMI_KMeansPlusPlus:
			kMeansPlusPlusInitialization( data, centers );
			break;
		default:
			NeoAssert( false );
	}
}

void CKMeansClustering::defaultInitialization( const CDnnBlob& data, CDnnBlob& centers )
{
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();
	NeoAssert( centers.GetObjectCount() == params.InitialClustersCount );
	NeoAssert( centers.GetObjectSize() == featureCount );

	const int step = max( vectorCount / params.InitialClustersCount, 1 );
	NeoAssert( step > 0 );
	IMathEngine& mathEngine = data.GetMathEngine();

	for( int i = 0; i < params.InitialClustersCount; i++ ) {
		const int pos = ( i * step ) % vectorCount;
		mathEngine.VectorCopy( centers.GetObjectData( i ), data.GetObjectData( pos ), featureCount );
	}
}

void CKMeansClustering::kMeansPlusPlusInitialization( const CDnnBlob& data, CDnnBlob& centers )
{
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();
	//	берем случайный вектор
	CRandom random( 0xCEA );
	const int firstChoice = random.UniformInt( 0, vectorCount - 1 );
	//	копируем его
	IMathEngine& mathEngine = centers.GetMathEngine();
	mathEngine.VectorCopy( centers.GetData(), data.GetObjectData( firstChoice ), data.GetObjectSize() );

	CFloatHandleStackVar stackBuff( mathEngine, vectorCount + 1 );
	CFloatHandle sumBlob = stackBuff;
	CFloatHandle currDists = stackBuff + 1;

	CHashTable<int> usedVectors;
	//	для каждой точки будем искать расстояния до близжайшего центроида.
	//	тут делаем инициализацию
	CPtr<CDnnBlob> prevDists = CDnnBlob::CreateVector( mathEngine, CT_Float, vectorCount );
	mathEngine.MatrixRowsToVectorSquaredL2Distance( data.GetData(), vectorCount, featureCount,
		centers.GetData(), prevDists->GetData() );
	//	теперь последовательно выбираем кластеры
	for( int k = 1; k < params.InitialClustersCount; k++ ) {
		CConstFloatHandle currentVector = centers.GetObjectData( k - 1 );
		//	для каждой точки ищем расстояние до близжайшего центроида
		mathEngine.MatrixRowsToVectorSquaredL2Distance( data.GetData(), vectorCount, featureCount,
			currentVector, currDists );
		//	ищем сам номер центроида
		mathEngine.VectorEltwiseMin( currDists, prevDists->GetData(), prevDists->GetData(), prevDists->GetDataSize() );
		//	вероятность выбора точки должна быть пропорциональная квадрату расстояния
		//	до нее
		mathEngine.VectorSum( prevDists->GetData(), prevDists->GetDataSize(), sumBlob );

		//	ищем следующий:
		//	берем случайную число от 0 до суммы
		const double sumValue = static_cast<double>( sumBlob.GetValue() );
		const double scaledSum = random.Uniform( 0, 1 ) * sumValue;

		CArray<float> squares;
		squares.SetSize( prevDists->GetDataSize() );
		prevDists->CopyTo( squares.GetPtr() );
	
		//	теперь снова начинаем накапливать сумму.
		//	когда превысим сгенерированное число, остановимся и зафиксируем результат
		double prefixSum = 0;
		int nextCoice = -1;
		for( int i = 0; i < vectorCount; i++ ) {
			prefixSum += squares[i];
			if( prefixSum > scaledSum ) {
				nextCoice = i;
				break;
			}
		}
		//	алгоритм by-default учитывает эти условия (расстояния будут равны 0 и точки мы пропустим)
		//	но проверять все же надо
		assert( nextCoice != -1 );
		assert( !usedVectors.Has( nextCoice ) );
		usedVectors.Add( nextCoice );
		//	копируем выбранный вектор
		mathEngine.VectorCopy( centers.GetObjectData( k ), data.GetObjectData( nextCoice ), featureCount );
	}
}

bool CKMeansClustering::lloydBlobClusterization( const CDnnBlob& data,
	CDnnBlob& centers, CDnnBlob& sizes, CDnnBlob& labels )
{
	double prevDist = FLT_MAX;
	double totalDist = 0;
	const float eps = 1e-3f;

	IMathEngine& mathEngine = data.GetMathEngine();
	// pre-calculate l2-norm of input data
	CPtr<CDnnBlob> squaredData = CDnnBlob::CreateVector( mathEngine, CT_Float, data.GetObjectCount() );
	mathEngine.RowMultiplyMatrixByMatrix( data.GetData(), data.GetData(), data.GetObjectCount(),
		data.GetObjectSize(), squaredData->GetData() );
	for( int iter = 0; iter < params.MaxIterations; iter++ ) {
		totalDist = assignClosest( data, *squaredData, centers, labels );
		recalcCenters( data, labels, centers, sizes );
		if( abs( prevDist - totalDist ) < eps ) {
			return true;
		}
		prevDist = totalDist;
	}

	return false;
}

//	пересчет расстояний от точек до кластеров
static void calcPairwiseDistances( const CDnnBlob& data, const CDnnBlob& squaredData, const CDnnBlob& centers,
	CFloatHandle& distances )
{
	IMathEngine& mathEngine = data.GetMathEngine();
	const int clusterCount = centers.GetObjectCount();
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();
	CFloatHandleStackVar stackBuff( mathEngine, clusterCount + 1 );
	CFloatHandle squaredCenters = stackBuff.GetHandle();
	CFloatHandle minusTwo = stackBuff.GetHandle() + clusterCount;
	
	minusTwo.SetValue( -2.f );
	// pre-calculate l2-norm of current cluster centers
	mathEngine.RowMultiplyMatrixByMatrix( centers.GetData(), centers.GetData(), clusterCount, featureCount, squaredCenters );

	// (a - b)^2 = a^2 + b^2 - 2*a*b
	mathEngine.MultiplyMatrixByTransposedMatrix( 1, centers.GetData(), clusterCount, featureCount, data.GetData(),
		vectorCount, distances, clusterCount * vectorCount );
	mathEngine.VectorMultiply( distances, distances, clusterCount * vectorCount, minusTwo );
	mathEngine.AddVectorToMatrixRows( 1, distances, distances, clusterCount, vectorCount, squaredData.GetData() );
	mathEngine.AddVectorToMatrixColumns( distances, distances, clusterCount, vectorCount, squaredCenters );
}

//	привязка точек к кластерам
double CKMeansClustering::assignClosest( const CDnnBlob& data, const CDnnBlob& squaredData, const CDnnBlob& centers, CDnnBlob& labels )
{
	IMathEngine& mathEngine = data.GetMathEngine();
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();
	const int clusterCount = centers.GetObjectCount();
	CFloatHandleStackVar stackBuff( mathEngine, vectorCount * clusterCount + vectorCount + 1 );
	CFloatHandle distances = stackBuff.GetHandle();
	CFloatHandle closestDist = stackBuff.GetHandle() + vectorCount * clusterCount;
	CFloatHandle totalDist = stackBuff.GetHandle() + vectorCount * clusterCount + vectorCount + clusterCount;
	calcPairwiseDistances( data, squaredData, centers, distances );
	mathEngine.FindMinValueInColumns( distances, clusterCount, vectorCount, closestDist, labels.GetData<int>() );
	mathEngine.VectorSum( closestDist, vectorCount, totalDist );
	const double result = static_cast<double>( totalDist.GetValue() );
	return result;
}

//	раскидываем объекты по текущим центрам, усредняем на кол-во объектов
void CKMeansClustering::recalcCenters( const CDnnBlob& data, const CDnnBlob& labels,
	CDnnBlob& centers, CDnnBlob& sizes )
{
	IMathEngine& mathEngine = data.GetMathEngine();
	const int clusterCount = params.InitialClustersCount;
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();

	mathEngine.LookupAndAddToTable( labels.GetData<int>(), vectorCount, 1, data.GetData(),
		data.GetObjectSize(), centers.GetData(), clusterCount );

	mathEngine.BuildIntegerHist( labels.GetData<int>(), vectorCount, sizes.GetData<int>(), clusterCount );

	int* rawSizes = sizes.GetBuffer<int>( 0, clusterCount );
	//	пустые кластеры игнорируем
	CFloatHandleStackVar invertedSize( mathEngine );
	for( int i = 0; i < clusterCount; i++ ) {
		if( rawSizes[i] > 0 ) {
			invertedSize.SetValue( 1.f / rawSizes[i] );
			mathEngine.VectorMultiply( centers.GetObjectData( i ), centers.GetObjectData( i ),
				featureCount, invertedSize );
		}
	}
	sizes.ReleaseBuffer( rawSizes, false );
}

// подсчет дисперсий кластеров
void CKMeansClustering::calcClusterVariances( const CDnnBlob& data, const CDnnBlob& labels,
	const CDnnBlob& centers, const CDnnBlob& sizes, CDnnBlob& variances )
{
	IMathEngine& mathEngine = data.GetMathEngine();
	const int vectorCount = data.GetObjectCount();
	const int featureCount = data.GetObjectSize();
	const int clusterCount = sizes.GetDataSize();
	
	// 1 / *размер кластера*
	CPtr<CDnnBlob> sizeInv = CDnnBlob::CreateVector( mathEngine, CT_Float, clusterCount );
	{
		int* sizeBuff = const_cast<CDnnBlob&>( sizes ).GetBuffer<int>( 0, clusterCount );
		float* sizeInvBuff = sizeInv->GetBuffer<float>( 0, clusterCount );
		for( int i = 0; i < clusterCount; ++i ) {
			sizeInvBuff[i] = sizeBuff[i] > 0 ? 1.f / sizeBuff[i] : 1.f;
		}
		sizeInv->ReleaseBuffer( sizeInvBuff, true );
		const_cast<CDnnBlob&>( sizes ).ReleaseBuffer( sizeBuff, false );
	}

	{
		// Подсчитываем суммы квадратов значений каждого признака для каждого кластера.
		CFloatHandleStackVar stackBuff( mathEngine, vectorCount * featureCount + clusterCount * featureCount + 1 );
		CFloatHandle squaredData = stackBuff.GetHandle();
		mathEngine.VectorEltwiseMultiply( data.GetData(), data.GetData(), squaredData,
			data.GetDataSize() );
		CFloatHandle sumOfSquares = stackBuff.GetHandle() + vectorCount * featureCount;
		mathEngine.VectorFill( sumOfSquares, 0, clusterCount * featureCount );
		CFloatHandle one = stackBuff.GetHandle() + vectorCount * featureCount + clusterCount * featureCount;
		one.SetValue( 1.f );
		CLookupDimension dim;
		dim.VectorCount = params.InitialClustersCount;
		dim.VectorSize = featureCount;
		variances.Clear();
		mathEngine.VectorMultichannelLookupAndAddToTable( vectorCount, 1, labels.GetData<int>(),
			&sumOfSquares, &dim, 1, one, squaredData, featureCount );
		// Подсчитываем суммы квадратов значений каждого признака для каждого кластера, поделенные на размер кластера.
		mathEngine.MultiplyDiagMatrixByMatrix( sizeInv->GetData(), clusterCount, sumOfSquares, featureCount,
			variances.GetData(), variances.GetDataSize() );
	}

	{
		// Подсчитываем квадрат 
		CFloatHandleStackVar squaredMean( mathEngine, clusterCount * featureCount );
		mathEngine.VectorEltwiseMultiply( centers.GetData(), centers.GetData(),
			squaredMean, clusterCount * featureCount );
		// Вычитаем из усредненных квадратов квадраты средних и получаем дисперсию.
		mathEngine.VectorSub( variances.GetData(), squaredMean, variances.GetData(), clusterCount * featureCount );
	}
}

} // namespace NeoML
