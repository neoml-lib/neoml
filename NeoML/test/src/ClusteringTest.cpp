/* Copyright © 2021-2024 ABBYY

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

namespace NeoMLTest {

//---------------------------------------------------------------------------------------------------------------------

typedef void ( *TClusteringFunction )( const IClusteringData* data, CClusteringResult& result );

struct CClusteringTestParams {
	TClusteringFunction Clusterize{};
	CClusteringTestParams( TClusteringFunction f ) : Clusterize( f ) {}
};

struct CClusteringTest : public CNeoMLTestFixture, public ::testing::WithParamInterface<CClusteringTestParams> {
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

//---------------------------------------------------------------------------------------------------------------------

typedef void ( *THierarchicalDendrogram )( const IClusteringData* data, CClusteringResult& result,
	CArray<CHierarchicalClustering::CMergeInfo>& dendrogram, CArray<int>& dendrogramIndices );

struct CClusteringDendrogramTestParams {
	THierarchicalDendrogram Dendrogram{};
	CClusteringDendrogramTestParams( THierarchicalDendrogram f ) : Dendrogram( f ) {}
};

struct CClusteringDendrogramTest : public CNeoMLTestFixture,
		public ::testing::WithParamInterface<CClusteringDendrogramTestParams> {
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

//---------------------------------------------------------------------------------------------------------------------

class CClusteringTestData : public IClusteringData {
public:
	CClusteringTestData( const CArray<CSparseFloatVector>& vectors, int featureCount, bool isDense );

	int GetVectorCount() const override { return desc.Height; }
	int GetFeaturesCount() const override { return desc.Width; }
	CFloatMatrixDesc GetMatrix() const override { return desc; }
	double GetVectorWeight( int /* index */ ) const override { return 1; }

private:
	CFloatMatrixDesc desc;
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
			const CFloatVectorDesc& vectorDesc = vectors[i].GetDesc();
			for( int j = 0; j < vectorDesc.Size; ++j ) {
				rowPtr[vectorDesc.Indexes[j]] = vectorDesc.Values[j];
			}
			rowPtr += featureCount;
			pointers.Add( i * desc.Width );
		}
		pointers.Add( desc.Height * desc.Width );
		desc.Columns = nullptr;
	} else {
		pointers.Add( 0 );
		for( int i = 0; i < vectors.Size(); ++i ) {
			const CFloatVectorDesc& vectorDesc = vectors[i].GetDesc();
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

//---------------------------------------------------------------------------------------------------------------------

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
		double mean = random.Next() % 2 == 0 ? -1 : 1;
		double sigma = 1. / 4;
		for( int j = 0; j < featureCount; ++j ) {
			if( random.Next() % 3 != 0 ) {
				vector.SetAt( j, static_cast<float>( random.Normal( mean, sigma ) ) );
				hasElements = true;
			}
		}
		if( !hasElements ) {
			// Avoiding 0 vectors
			const int index = random.UniformInt( 0, featureCount - 1 );
			vector.SetAt( index, static_cast<float>( random.Normal( mean, sigma ) ) );
		}
	}

	sparseData = new CClusteringTestData( vectors, featureCount, false );
	denseData = new CClusteringTestData( vectors, featureCount, true );
}

//---------------------------------------------------------------------------------------------------------------------

// Clustering functions

static void firstComeClustering( const IClusteringData* data, CClusteringResult& result )
{
	CFirstComeClustering::CParam params;
	params.Threshold = 5.0;

	CFirstComeClustering firstCome( params );
	firstCome.Clusterize( data, result );
}

template<CHierarchicalClustering::TLinkage LINKAGE>
static void hierarchicalClustering( const IClusteringData* data, CClusteringResult& result )
{
	CHierarchicalClustering::CParam params;
	params.Linkage = LINKAGE;
	params.DistanceType = DF_Euclid;
	params.MinClustersCount = 2;
	params.MaxClustersDistance = 10;

	CHierarchicalClustering hierarchical( params );
	if( static_cast<int>( LINKAGE ) % 2 == 0 ) {
		CArray<CHierarchicalClustering::CMergeInfo> dendrogram;
		CArray<int> dendrogramIndices;
		hierarchical.ClusterizeEx( data, result, dendrogram, dendrogramIndices );
	} else {
		hierarchical.Clusterize( data, result );
	}
}

static void isoDataClustering( const IClusteringData* data, CClusteringResult& result )
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

static void kmeansLloydClustering( const IClusteringData* data, CClusteringResult& result )
{
	CKMeansClustering::CParam params;
	params.DistanceFunc = DF_Euclid;
	params.InitialClustersCount = 2;
	params.MaxIterations = 50;
	params.Algo = CKMeansClustering::KMA_Lloyd;
	params.Initialization = CKMeansClustering::KMI_Default;
	params.ThreadCount = -1;

	CKMeansClustering kMeans( params );
	kMeans.Clusterize( data, result );
}

static void kmeansElkanClustering( const IClusteringData* data, CClusteringResult& result )
{
	CKMeansClustering::CParam params;
	params.DistanceFunc = DF_Euclid;
	params.InitialClustersCount = 2;
	params.MaxIterations = 50;
	params.Algo = CKMeansClustering::KMA_Elkan;
	params.Initialization = CKMeansClustering::KMI_KMeansPlusPlus;
	params.ThreadCount = -1;

	CKMeansClustering kMeans( params );
	kMeans.Clusterize( data, result );
}

//---------------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------------

// Check backward compatibility

static void precalcTestImpl( TClusteringFunction clusterize, const CClusteringResult& expectedResult )
{
	CPtr<IClusteringData> sparseData = nullptr;
	CPtr<IClusteringData> denseData = nullptr;

	generateData( 128, 3, 0x451, sparseData, denseData );

	CClusteringResult sparseResult;
	clusterize( sparseData, sparseResult );

	CClusteringResult denseResult;
	clusterize( denseData, denseResult );

	EXPECT_TRUE( isEqual( sparseResult, denseResult ) );
	EXPECT_TRUE( isEqual( sparseResult, expectedResult ) );
	EXPECT_TRUE( isEqual( denseResult, expectedResult ) );
}

static void kmeansElkanDefaultInitClustering( const IClusteringData* data, CClusteringResult& result )
{
	CKMeansClustering::CParam params;
	params.DistanceFunc = DF_Euclid;
	params.InitialClustersCount = 2;
	params.MaxIterations = 50;
	params.Algo = CKMeansClustering::KMA_Elkan;
	params.Initialization = CKMeansClustering::KMI_Default;
	params.ThreadCount = 4;

	CKMeansClustering kMeans( params );
	kMeans.Clusterize( data, result );
}

// Returns data with a specific dendrogram (only Distances may vary)
static void getDendrogramData( CPtr<IClusteringData>& denseData, CArray<CHierarchicalClustering::CMergeInfo>& dendrogram,
	CArray<int>& dendrogramIndices )
{
	// Points on plane which can be easily divided into 2 clusters
	// but greedy algorithms won't process it in correct order (and will sort their result afterwards)
	const CArray<float> flattenedData = { 0.f, 0.f,
		0.f, 1.f,
		2.f, 0.f,
		10.f, 10.f,
		10.1f, 10.f,
		10.f, 10.2f,
		25.f, 25.f };
	const int vectorCount = flattenedData.Size() / 2;
	const int featureCount = 2;

	CArray<CSparseFloatVector> vectors;
	vectors.SetBufferSize( vectorCount );
	for( int i = 0; i < vectorCount; ++i ) {
		CSparseFloatVector& vector = vectors.Append();
		vector.SetAt( 0, flattenedData[2 * i] );
		vector.SetAt( 1, flattenedData[2 * i + 1] );
	}

	denseData = new CClusteringTestData( vectors, featureCount, true );
	CFloatVector mean( 2 );

	dendrogram.SetSize( 4 );
	dendrogram[0].First = 3;
	dendrogram[0].Second = 4;
	mean.SetAt( 0, 10.05f );
	mean.SetAt( 1, 10.f );
	dendrogram[0].Center = CClusterCenter( mean );
	dendrogram[0].Center.Weight = 2;

	dendrogram[1].First = 5;
	dendrogram[1].Second = 7;
	mean.SetAt( 0, 30.1f / 3.f );
	mean.SetAt( 1, 30.2f / 3.f );
	dendrogram[1].Center = CClusterCenter( mean );
	dendrogram[1].Center.Weight = 3;

	dendrogram[2].First = 0;
	dendrogram[2].Second = 1;
	mean.SetAt( 0, 0.f );
	mean.SetAt( 1, 0.5f );
	dendrogram[2].Center = CClusterCenter( mean );
	dendrogram[2].Center.Weight = 2;

	dendrogram[3].First = 2;
	dendrogram[3].Second = 9;
	mean.SetAt( 0, 2.f / 3.f );
	mean.SetAt( 1, 1.f / 3.f );
	dendrogram[3].Center = CClusterCenter( mean );
	dendrogram[3].Center.Weight = 3;

	dendrogramIndices.Empty();
	dendrogramIndices.Add( { 6, 8, 10 } );
}

template<CHierarchicalClustering::TLinkage LINKAGE>
static void hierarchicalDendrogram( const IClusteringData* data, CClusteringResult& result,
	CArray<CHierarchicalClustering::CMergeInfo>& dendrogram, CArray<int>& dendrogramIndices )
{
	CHierarchicalClustering::CParam params;
	params.Linkage = LINKAGE;
	params.DistanceType = DF_Euclid;
	params.MinClustersCount = 1; // The algo shouldn't merge last 2 clusters into 1 and should return true
	params.MaxClustersDistance = 7;
	CHierarchicalClustering clustering( params );

	EXPECT_TRUE( clustering.ClusterizeEx( data, result, dendrogram, dendrogramIndices ) );
}

typedef void ( *THierarchicalClusteringFunction )( const IClusteringData* data, CClusteringResult& result,
	CArray<CHierarchicalClustering::CMergeInfo>& dendrogram, CArray<int>& dendrogramIndices );

static inline bool compareVectors( const CFloatVector& first, const CFloatVector& second, float eps = 1e-5f )
{
	if( first.Size() != second.Size() ) {
		return false;
	}

	for( int i = 0; i < first.Size(); ++i ) {
		if( ::fabsf( first[i] - second[i] ) >= eps ) {
			return false;
		}
	}

	return true;
}

static inline bool compareCenters( const CClusterCenter& first, const CClusterCenter& second, float eps = 1e-5f )
{
	return compareVectors( first.Mean, second.Mean ) && compareVectors( first.Disp, second.Disp )
		&& ::abs( first.Norm - second.Norm ) < eps && ::abs( first.Weight - second.Weight ) < eps;
}

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------

// Tests implementation

TEST_F( CClusteringTest, PrecalcFirstCome )
{
	CClusteringResult expectedResult;
	expectedResult.ClusterCount = 2;
	expectedResult.Clusters.SetSize( 2 );
	expectedResult.Clusters[0].Mean = buildFloatVector( { 0.561685, 0.627736, 0.710235 } );
	expectedResult.Clusters[0].Disp = buildFloatVector( { 0.290764, 0.269636, 0.278547 } );
	expectedResult.Clusters[0].Norm = 1.213978;
	expectedResult.Clusters[1].Mean = buildFloatVector( { -0.675586, -0.655485, -0.643366 } );
	expectedResult.Clusters[1].Disp = buildFloatVector( { 0.215411, 0.280084, 0.265546 } );
	expectedResult.Clusters[1].Norm = 1.299998;

	precalcTestImpl( firstComeClustering, expectedResult );
}

TEST_F( CClusteringTest, PrecalcHierarchicalCentroid )
{
	CClusteringResult expectedResult;
	expectedResult.ClusterCount = 2;
	expectedResult.Clusters.SetSize( 2 );
	expectedResult.Clusters[0].Mean = buildFloatVector( { 0.580782, 0.636106, 0.719705 } );
	expectedResult.Clusters[0].Disp = buildFloatVector( { 0.266925, 0.267907, 0.275445 } );
	expectedResult.Clusters[0].Norm = 1.259914;
	expectedResult.Clusters[1].Mean = buildFloatVector( { -0.679265, -0.643118, -0.631227 } );
	expectedResult.Clusters[1].Disp = buildFloatVector( { 0.212050, 0.282753, 0.268198 } );
	expectedResult.Clusters[1].Norm = 1.273450;

	precalcTestImpl( hierarchicalClustering<CHierarchicalClustering::L_Centroid>, expectedResult );
}

TEST_F( CClusteringTest, PrecalcIsoData )
{
	CClusteringResult expectedResult;
	expectedResult.ClusterCount = 2;
	expectedResult.Clusters.SetSize( 2 );
	expectedResult.Clusters[0].Mean = buildFloatVector( { -0.256940, -0.233290, -0.418645 } );
	expectedResult.Clusters[0].Disp = buildFloatVector( { 0.604588, 0.636909, 0.282105 } );
	expectedResult.Clusters[0].Norm = 0.295706;
	expectedResult.Clusters[1].Mean = buildFloatVector( { 0.551979, 0.636387, 1.063543 } );
	expectedResult.Clusters[1].Disp = buildFloatVector( { 0.269381, 0.263475, 0.052157 } );
	expectedResult.Clusters[1].Norm = 1.840793;

	precalcTestImpl( isoDataClustering, expectedResult );
}

TEST_F( CClusteringTest, PrecalcKmeans )
{
	CClusteringResult expectedResult;
	expectedResult.ClusterCount = 2;
	expectedResult.Clusters.SetSize( 2 );
	expectedResult.Clusters[0].Mean = buildFloatVector( { -0.679265, -0.643118, -0.631227 } );
	expectedResult.Clusters[0].Disp = buildFloatVector( { 0.212050, 0.282753, 0.268198 } );
	expectedResult.Clusters[0].Norm = 1.273450;
	expectedResult.Clusters[1].Mean = buildFloatVector( { 0.580782, 0.636106, 0.719705 } );
	expectedResult.Clusters[1].Disp = buildFloatVector( { 0.266925, 0.267907, 0.275445 } );
	expectedResult.Clusters[1].Norm = 1.259914;

	precalcTestImpl( kmeansLloydClustering, expectedResult );
	// Check that different algos with the same initialization return similar results
	precalcTestImpl( kmeansElkanDefaultInitClustering, expectedResult );
}

//---------------------------------------------------------------------------------------------------------------------

// Check on trivial sample
TEST_P( CClusteringTest, Sample )
{
	TClusteringFunction clusterize = GetParam().Clusterize;

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
	TClusteringFunction clusterize = GetParam().Clusterize;

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
	::testing::Values(
		firstComeClustering,
		hierarchicalClustering<CHierarchicalClustering::L_Centroid>,
		hierarchicalClustering<CHierarchicalClustering::L_Single>,
		hierarchicalClustering<CHierarchicalClustering::L_Average>,
		hierarchicalClustering<CHierarchicalClustering::L_Complete>,
		hierarchicalClustering<CHierarchicalClustering::L_Ward>,
		isoDataClustering,
		kmeansElkanClustering,
		kmeansLloydClustering
	)
);

//---------------------------------------------------------------------------------------------------------------------

TEST_P( CClusteringDendrogramTest, HierarchicalDendrogram )
{
	THierarchicalDendrogram dendrogram = GetParam().Dendrogram;
	{
		CPtr<IClusteringData> data;
		CArray<CHierarchicalClustering::CMergeInfo> expectedDendrogram;
		CArray<int> expectedIndices;
		getDendrogramData( data, expectedDendrogram, expectedIndices );

		CClusteringResult result;
		CArray<CHierarchicalClustering::CMergeInfo> actualDendrogram;
		CArray<int> actualIndices;
		dendrogram( data, result, actualDendrogram, actualIndices );

		ASSERT_EQ( expectedDendrogram.Size(), actualDendrogram.Size() );
		for( int i = 0; i < expectedDendrogram.Size(); ++i ) {
			const CHierarchicalClustering::CMergeInfo& expected = expectedDendrogram[i];
			const CHierarchicalClustering::CMergeInfo& actual = actualDendrogram[i];
			ASSERT_GT( actual.Distance, 0.f );
			ASSERT_TRUE( ( expected.First == actual.First && expected.Second == actual.Second )
				|| ( expected.First == actual.Second && expected.Second == actual.First ) );
			if( i < expectedDendrogram.Size() - 1 ) {
				ASSERT_GT( actualDendrogram[i + 1].Distance, actual.Distance );
			}
			ASSERT_TRUE( compareCenters( expected.Center, actual.Center ) );
		}

		ASSERT_EQ( expectedIndices.Size(), actualIndices.Size() );
		for( int i = 0; i < expectedIndices.Size(); ++i ) {
			int resultPos = actualIndices.Find( expectedIndices[i] );
			ASSERT_NE( NotFound, resultPos );
			int index = expectedIndices[i];
			if( index < data->GetVectorCount() ) {
				// One of the original 1-element clusters
				ASSERT_TRUE( compareVectors( result.Clusters[resultPos].Mean,
					CFloatVector( data->GetFeaturesCount(), data->GetMatrix().GetRow( index ) ) ) );
				ASSERT_NEAR( result.Clusters[resultPos].Weight, data->GetVectorWeight( index ), 1e-5 );
			} else {
				// One of the dendrogram roots
				index -= data->GetVectorCount();
				ASSERT_TRUE( compareCenters( result.Clusters[resultPos], actualDendrogram[index].Center ) );
			}
		}
	}
}

INSTANTIATE_TEST_CASE_P( CClusteringTestDendrogramInstantiation, CClusteringDendrogramTest,
	::testing::Values(
		hierarchicalDendrogram<CHierarchicalClustering::L_Centroid>,
		hierarchicalDendrogram<CHierarchicalClustering::L_Single>,
		hierarchicalDendrogram<CHierarchicalClustering::L_Average>,
		hierarchicalDendrogram<CHierarchicalClustering::L_Complete>,
		hierarchicalDendrogram<CHierarchicalClustering::L_Ward>
	)
);
