/* Copyright © 2021 ABBYY Production LLC

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

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

typedef void (*TClusteringFunction)( IClusteringData* data, CClusteringResult& result );

class CClusteringTest : public CNeoMLTestFixture, public ::testing::WithParamInterface<TClusteringFunction> {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

class CClusteringTestData : public IClusteringData {
public:
	CClusteringTestData( const CArray<CSparseFloatVector>& vectors, int featureCount, bool isDense );

	int GetVectorCount() const override { return desc.Height; }
	int GetFeaturesCount() const override { return desc.Width; }
	CSparseFloatMatrixDesc GetMatrix() const override { return desc; }
	double GetVectorWeight( int /* index */ ) const override { return 1; }

private:
	CSparseFloatMatrixDesc desc;
	CArray<int> columns;
	CArray<float> values;
	CArray<int> beginPointers;
	CArray<int> endPointers;
};

CClusteringTestData::CClusteringTestData( const CArray<CSparseFloatVector>& vectors, int featureCount, bool isDense )
{
	desc.Height = vectors.Size();
	desc.Width = featureCount;

	if( isDense ) {
		values.Add( 0, desc.Height * desc.Width );
		float* rowPtr = values.GetPtr();
		for( int i = 0; i < vectors.Size(); ++i ) {
			const CSparseFloatVectorDesc& desc = vectors[i].GetDesc();
			for( int j = 0; j < desc.Size; ++j ) {
				rowPtr[desc.Indexes[j]] = desc.Values[j];
			}
			rowPtr += featureCount;
		}
		desc.Values = values.GetPtr();
	} else {
		beginPointers.SetBufferSize( desc.Height );
		endPointers.SetBufferSize( desc.Height );
		for( int i = 0; i < vectors.Size(); ++i ) {
			const CSparseFloatVectorDesc& desc = vectors[i].GetDesc();
			int prevElementCount = values.Size();
			beginPointers.Add( prevElementCount );
			endPointers.Add( prevElementCount + desc.Size );
			values.SetSize( prevElementCount + desc.Size );
			::memcpy( values.GetPtr() + prevElementCount, desc.Values, desc.Size * sizeof( float ) );
			columns.SetSize( prevElementCount + desc.Size );
			::memcpy( columns.GetPtr() + prevElementCount, desc.Indexes, desc.Size * sizeof( int ) );
		}
		desc.Values = values.GetPtr();
		desc.Columns = columns.GetPtr();
		desc.PointerB = beginPointers.GetPtr();
		desc.PointerE = endPointers.GetPtr();
	}
}

// --------------------------------------------------------------------------------------------------------------------
// Data functions

static void getSampleData( CPtr<IClusteringData>& sparseData, CPtr<IClusteringData>& denseData )
{
	// Points on plane which can be easily divided into 2 clusters
	const CArray<float> flattenedData = { 0, 0,
		1, 1,
		2, 2,
		4, 3,
		4, 4,
		5, 3,
		5, 4,
		6, 5 };
	const int vectorCount = flattenedData.Size() / 2;
	const int featureCount = 2;

	CArray<CSparseFloatVector> vectors;
	vectors.SetBufferSize( vectorCount );
	for( int i = 0; i < vectorCount; ++i ) {
		CSparseFloatVector& vector = vectors.Append();
		vector.SetAt( 0, flattenedData[2 * i] );
		vector.SetAt( 1, flattenedData[2 * i + 1] );
	}

	sparseData = new CClusteringTestData( vectors, featureCount, false );
	denseData = new CClusteringTestData( vectors, featureCount, true );
}

static void generateData( int vectorCount, int featureCount, int seed,
	CPtr<IClusteringData>& sparseData, CPtr<IClusteringData>& denseData )
{
	CRandom random( seed );

	CArray<CSparseFloatVector> vectors;
	vectors.SetBufferSize( vectorCount );
	for( int i = 0; i < vectorCount; ++i ) {
		CSparseFloatVector& vector = vectors.Append();
		bool hasElements = false;
		for( int j = 0; j < featureCount; ++j ) {
			if( random.Next() % 3 == 2 ) {
				vector.SetAt( j, static_cast<float>( random.Uniform( -1, 1 ) ) );
			}
		}
		if( !hasElements ) {
			// Avoiding 0 vectors
			vector.SetAt( random.UniformInt( 0, featureCount - 1 ), static_cast<float>( random.Uniform( -1, 1 ) ) );
		}
	}

	sparseData = new CClusteringTestData( vectors, featureCount, false );
	denseData = new CClusteringTestData( vectors, featureCount, true );
}

// --------------------------------------------------------------------------------------------------------------------
// Clustering functions

static void firstComeClustering( IClusteringData* data, CClusteringResult& result )
{
	CFirstComeClustering::CParam params;
	params.Threshold = 5.0;

	CFirstComeClustering firstCome( params );
	firstCome.Clusterize( data, result );
}

static void hierarchicalClustering( IClusteringData* data, CClusteringResult& result )
{
	CHierarchicalClustering::CParam params;
	params.DistanceType = DF_Euclid;
	params.MinClustersCount = 2;
	params.MaxClustersDistance = 5;

	CHierarchicalClustering hierarchical( params );
	hierarchical.Clusterize( data, result );
}

static void isoDataClustering( IClusteringData* data, CClusteringResult& result )
{
	CIsoDataClustering::CParam params;
	params.MinClusterSize = 1;
	params.MinClustersDistance = 2.0;
	params.MaxClustersCount = 100;
	params.MaxClusterDiameter = 2.0;
	params.MeanDiameterCoef = 0.5;
	params.MaxIterations = 100;
	params.InitialClustersCount = 1;

	CIsoDataClustering isoData( params );
	isoData.Clusterize( data, result );
}

static void kmeansLloydClustering( IClusteringData* data, CClusteringResult& result )
{
	CKMeansClustering::CParam params;
	params.DistanceFunc = DF_Euclid;
	params.InitialClustersCount = 2;
	params.MaxIterations = 50;
	params.Algo = CKMeansClustering::KMA_Lloyd;
	params.Initialization = CKMeansClustering::KMI_Default;
	params.ThreadCount = 1;

	CKMeansClustering kMeans( params );
	kMeans.Clusterize( data, result );
}

static void kmeansElkanClustering( IClusteringData* data, CClusteringResult& result )
{
	CKMeansClustering::CParam params;
	params.DistanceFunc = DF_Euclid;
	params.InitialClustersCount = 2;
	params.MaxIterations = 50;
	params.Algo = CKMeansClustering::KMA_Elkan;
	params.Initialization = CKMeansClustering::KMI_KMeansPlusPlus;
	params.ThreadCount = 1;

	CKMeansClustering kMeans( params );
	kMeans.Clusterize( data, result );
}

// --------------------------------------------------------------------------------------------------------------------
// Result check functions

static const float eps = 1e-4f;

static bool isEqual( const CFloatVector& first, const CFloatVector& second )
{
	if( first.Size() != second.Size() ) {
		return false;
	}

	const float* firstPtr = first.GetPtr();
	const float* secondPtr = second.GetPtr();

	for( int i = 0; i < first.Size(); ++i ) {
		if( fabs( firstPtr[i] - secondPtr[i] ) > eps ) {
			return false;
		}
	}

	return true;
}

static bool isEqual( const CClusteringResult& first, const CClusteringResult& second )
{
	if( first.ClusterCount != second.ClusterCount ) {
		return false;
	}

	for( int clusterIndex = 0; clusterIndex < first.ClusterCount; ++clusterIndex ) {
		const CClusterCenter& firstCluster = first.Clusters[clusterIndex];
		const CClusterCenter& secondCluster = second.Clusters[clusterIndex];
		if( abs( firstCluster.Norm - secondCluster.Norm ) > eps
			|| abs( firstCluster.Weight - secondCluster.Weight ) > eps
			|| !isEqual( firstCluster.Mean, secondCluster.Mean )
			|| !isEqual( firstCluster.Disp, secondCluster.Disp ) )
		{
			return false;
		}
	}

	return true;
}

static bool isCorrectSampleResult( const CClusteringResult& result )
{
	if( result.Data.Size() != 8 ) {
		return false;
	}
	// The sample data must split into 2 clusters in following way:
	// 1st cluster: 0, 1, 2
	// 2nd cluster: 3, 4, 5, 6, 7
	return ( result.Data[0] == result.Data[1] && result.Data[1] == result.Data[2] )
		&& ( result.Data[3] == result.Data[4] && result.Data[4] == result.Data[5]
			&& result.Data[5] == result.Data[6] && result.Data[6] == result.Data[7] );
}

// --------------------------------------------------------------------------------------------------------------------
// test implementation

TEST_P( CClusteringTest, Sample )
{
	TClusteringFunction clusterize = GetParam();

	CPtr<IClusteringData> sparseData = nullptr;
	CPtr<IClusteringData> denseData = nullptr;

	getSampleData( sparseData, denseData );

	CClusteringResult sparseResult;
	clusterize( sparseData, sparseResult );

	CClusteringResult denseResult;
	clusterize( denseData, denseResult );

	EXPECT_TRUE( isEqual( sparseResult, denseResult ) );
	EXPECT_TRUE( isCorrectSampleResult( sparseResult ) );
	EXPECT_TRUE( isCorrectSampleResult( denseResult ) );
}

TEST_P( CClusteringTest, Generated )
{
	TClusteringFunction clusterize = GetParam();

	CPtr<IClusteringData> sparseData = nullptr;
	CPtr<IClusteringData> denseData = nullptr;

	generateData( 512, 32, 0x1984, sparseData, denseData );

	CClusteringResult sparseResult;
	clusterize( sparseData, sparseResult );

	CClusteringResult denseResult;
	clusterize( denseData, denseResult );

	EXPECT_TRUE( isEqual( sparseResult, denseResult ) );
}

INSTANTIATE_TEST_CASE_P( CClusteringTestInstantiation, CClusteringTest,
	::testing::Values( firstComeClustering, hierarchicalClustering,
		isoDataClustering, kmeansElkanClustering, kmeansLloydClustering ) );
