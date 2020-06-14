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

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

static void matrixRowsToVectorSquaredL2DistanceTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval bufferHeightInterval = params.GetInterval( "BufferHeight" );
	const CInterval bufferWidthInterval = params.GetInterval( "BufferWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int matrixHeight = random.UniformInt( bufferHeightInterval.Begin, bufferHeightInterval.End );
	const int matrixWidth = random.UniformInt( bufferWidthInterval.Begin, bufferWidthInterval.End );

	std::vector<float> vector;
	vector.reserve( matrixWidth );
	for( int i = 0; i < matrixWidth; ++i ) {
		vector.push_back( static_cast<float>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) ) );
	}

	std::vector<float> matrix;
	matrix.reserve( matrixHeight * matrixWidth );

	std::vector<float> expected;
	expected.reserve( matrixHeight );

	for( int i = 0; i < matrixHeight; ++i ) {
		expected.push_back( 0.f );
		for( int j = 0; j < matrixWidth; ++j ) {
			matrix.push_back( static_cast<float>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) ) );
			expected[expected.size() - 1] += ( matrix[matrix.size() - 1] - vector[j] ) * ( matrix[matrix.size() - 1] - vector[j] );
		}
	}

	CFloatBlob vectorBlob( MathEngine(), 1, 1, 1, matrixWidth );
	vectorBlob.CopyFrom( vector.data() );

	CFloatBlob matrixBlob( MathEngine(), matrixHeight, matrixWidth, 1, 1, 1, 1, 1 );
	matrixBlob.CopyFrom( matrix.data() );

	CFloatBlob resultBlob( MathEngine(), 1, 1, 1, matrixHeight );

	MathEngine().MatrixRowsToVectorSquaredL2Distance( matrixBlob.GetData(), matrixHeight, matrixWidth,
		vectorBlob.GetData(), resultBlob.GetData() );

	std::vector<float> result;
	result.resize( matrixHeight );
	resultBlob.CopyTo( result.data() );

	for( size_t i = 0; i < result.size(); ++i ) {
		ASSERT_TRUE( FloatEq( expected[i], result[i], 1e-3f ) );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMatrixRowsToVectorSquaredL2DistanceTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMatrixRowsToVectorSquaredL2DistanceTestInstantiation, CMatrixRowsToVectorSquaredL2DistanceTest,
	::testing::Values(
		CTestParams(
			"BufferHeight = (1..50);"
			"BufferWidth = (1..50);"
			"Values = (-10..10);"
			"TestCount = 100;"
		)
	)
);


TEST_P( CMatrixRowsToVectorSquaredL2DistanceTest, Random )
{
	RUN_TEST_IMPL( matrixRowsToVectorSquaredL2DistanceTestImpl )
}
