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
	CArray<int> pointers;
};

CClusteringTestData::CClusteringTestData( const CArray<CSparseFloatVector>& vectors, int featureCount, bool isDense )
{
	desc.Height = vectors.Size();
	desc.Width = featureCount;
	pointers.SetBufferSize( desc.Height + 1 );

	if( isDense ) {
		values.Add( 0, desc.Height * desc.Width );
		float* rowPtr = values.GetPtr();
		for( int i = 0; i < vectors.Size(); ++i ) {
			const CSparseFloatVectorDesc& vectorDesc = vectors[i].GetDesc();
			for( int j = 0; j < vectorDesc.Size; ++j ) {
				rowPtr[vectorDesc.Indexes[j]] = vectorDesc.Values[j];
			}
			rowPtr += featureCount;
			pointers.Add( i * desc.Width );
		}
		pointers.Add( desc.Height * desc.Width );
	} else {
		pointers.Add( 0 );
		for( int i = 0; i < vectors.Size(); ++i ) {
			const CSparseFloatVectorDesc& vectorDesc = vectors[i].GetDesc();
			int prevElementCount = values.Size();
			pointers.Add( prevElementCount + vectorDesc.Size );
			values.SetSize( prevElementCount + vectorDesc.Size );
			::memcpy( values.GetPtr() + prevElementCount, vectorDesc.Values, vectorDesc.Size * sizeof( float ) );
			columns.SetSize( prevElementCount + vectorDesc.Size );
			::memcpy( columns.GetPtr() + prevElementCount, vectorDesc.Indexes, vectorDesc.Size * sizeof( int ) );
		}
		desc.Columns = columns.GetPtr();
	}

	desc.Values = values.GetPtr();
	desc.PointerB = pointers.GetPtr();
	desc.PointerE = desc.PointerB + 1;
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

static CFloatVector buildFloatVector( const CArray<float>& data )
{
	CFloatVector res( data.Size() );
	for( int i = 0; i < data.Size(); ++i ) {
		res.SetAt( i, data[i] );
	}
	return res;
}

// --------------------------------------------------------------------------------------------------------------------
// test implementation

// Check on trivial sample
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

// Compare sparse vs dense on generated data
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

// --------------------------------------------------------------------------------------------------------------------
// Check backward compatibility

static void precalcTestImpl( TClusteringFunction clusterize, const CClusteringResult& expectedResult )
{
	CPtr<IClusteringData> sparseData = nullptr;
	CPtr<IClusteringData> denseData = nullptr;

	generateData( 256, 3, 0x451, sparseData, denseData );

	CClusteringResult sparseResult;
	clusterize( sparseData, sparseResult );

	CClusteringResult denseResult;
	clusterize( denseData, denseResult );

	EXPECT_TRUE( isEqual( sparseResult, denseResult ) );
	EXPECT_TRUE( isEqual( sparseResult, expectedResult ) );
	EXPECT_TRUE( isEqual( denseResult, expectedResult ) );
}

TEST_F( CClusteringTest, PrecalcFirstCome )
{
	CClusteringResult expectedResult;
	expectedResult.ClusterCount = 1;
	expectedResult.Clusters.SetSize( 1 );
	expectedResult.Clusters[0].Mean = buildFloatVector( { -0.011701, 0.014629, 0.044868 } );
	expectedResult.Clusters[0].Disp = buildFloatVector( { 0.186866, 0.176815, 0.194769 } );
	expectedResult.Clusters[0].Norm = 0.002364;

	precalcTestImpl( firstComeClustering, expectedResult );
}

TEST_F( CClusteringTest, PrecalcHierarchical )
{
	CClusteringResult expectedResult;
	expectedResult.ClusterCount = 2;
	expectedResult.Clusters.SetSize( 2 );
	expectedResult.Clusters[0].Mean = buildFloatVector( { 0.000190, 0.025189, 0.057201 } );
	expectedResult.Clusters[0].Disp = buildFloatVector( { 0.180517, 0.172180, 0.187494 } );
	expectedResult.Clusters[0].Norm = 0.003906;
	expectedResult.Clusters[1].Mean = buildFloatVector( { -0.760800, -0.650639, -0.732119 } );
	expectedResult.Clusters[1].Disp = buildFloatVector( { 0.016785, 0.019241, 0.039801 } );
	expectedResult.Clusters[1].Norm = 1.538146;

	precalcTestImpl( hierarchicalClustering, expectedResult );
}

TEST_F( CClusteringTest, PrecalcIsoData )
{
	CClusteringResult expectedResult;
	expectedResult.ClusterCount = 1;
	expectedResult.Clusters.SetSize( 1 );
	expectedResult.Clusters[0].Mean = buildFloatVector( { -0.011701, 0.014629, 0.044868 } );
	expectedResult.Clusters[0].Disp = buildFloatVector( { 0.186866, 0.176815, 0.194769 } );
	expectedResult.Clusters[0].Norm = 0.002364;

	precalcTestImpl( isoDataClustering, expectedResult );
}

static void kmeansElkanDefaultInitClustering( IClusteringData* data, CClusteringResult& result )
{
	CKMeansClustering::CParam params;
	params.DistanceFunc = DF_Euclid;
	params.InitialClustersCount = 2;
	params.MaxIterations = 50;
	params.Algo = CKMeansClustering::KMA_Elkan;
	params.Initialization = CKMeansClustering::KMI_Default;
	params.ThreadCount = 1;

	CKMeansClustering kMeans( params );
	kMeans.Clusterize( data, result );
}

TEST_F( CClusteringTest, PrecalcKmeans )
{
	CClusteringResult expectedResult;
	expectedResult.ClusterCount = 2;
	expectedResult.Clusters.SetSize( 2 );
	expectedResult.Clusters[0].Mean = buildFloatVector( { -0.009386, -0.184222, -0.177035 } );
	expectedResult.Clusters[0].Disp = buildFloatVector( { 0.221126, 0.123710, 0.128456 } );
	expectedResult.Clusters[0].Norm = 0.065367;
	expectedResult.Clusters[1].Mean = buildFloatVector( { -0.014822, 0.282805, 0.344131 } );
	expectedResult.Clusters[1].Disp = buildFloatVector( { 0.140646, 0.123189, 0.128234 } );
	expectedResult.Clusters[1].Norm = 0.198624;

	precalcTestImpl( kmeansLloydClustering, expectedResult );
	// Check that different algos with the same initialization return similar results
	precalcTestImpl( kmeansElkanDefaultInitClustering, expectedResult );
}

INSTANTIATE_TEST_CASE_P( CClusteringTestInstantiation, CClusteringTest,
	::testing::Values( firstComeClustering, hierarchicalClustering,
		isoDataClustering, kmeansElkanClustering, kmeansLloydClustering ) );
