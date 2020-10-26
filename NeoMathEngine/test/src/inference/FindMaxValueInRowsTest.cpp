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

static void findMaxValueInRowsImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval matrixHeightInterval = params.GetInterval( "MatrixHeight" );
	const CInterval matrixWidthInterval = params.GetInterval( "MatrixWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	
	const int matrixHeight = random.UniformInt( matrixHeightInterval.Begin, matrixHeightInterval.End );
	const int matrixWidth = random.UniformInt( matrixWidthInterval.Begin, matrixWidthInterval.End );

	CREATE_FILL_FLOAT_ARRAY( matrix, valuesInterval.Begin, valuesInterval.End, matrixHeight * matrixWidth, random )

	std::vector<float> expected;
	std::vector<int> expectedIndices;
	expected.resize( matrixHeight );
	expectedIndices.resize( matrixHeight );
	for( int y = 0; y < matrixHeight; y++ ) {
		float max = matrix[y * matrixWidth];
		int maxIndex = 0;
		for( int x = 1; x < matrixWidth; x++ ) {
			if( matrix[y * matrixWidth + x] > max ) {
				max = matrix[y * matrixWidth + x];
				maxIndex = x;
			}
		}
		expected[y] = max;
		expectedIndices[y] = maxIndex;
	}

	std::vector<float> result;
	result.resize( matrixHeight );
	MathEngine().FindMaxValueInRows( CARRAY_FLOAT_WRAPPER( matrix ), matrixHeight, matrixWidth, CARRAY_FLOAT_WRAPPER( result ), matrixHeight );

	for( int i = 0; i < matrixHeight; i++ ) {
		ASSERT_NEAR( expected[i], result[i], 1e-3 );
	}

	std::vector<float> result1;
	result1.resize( matrixHeight );
	std::vector<int> resultIndices;
	resultIndices.resize( matrixHeight );
	MathEngine().FindMaxValueInRows( CARRAY_FLOAT_WRAPPER( matrix ), matrixHeight, matrixWidth, CARRAY_FLOAT_WRAPPER( result1 ), CARRAY_INT_WRAPPER( resultIndices ), matrixHeight );

	for( int i = 0; i < matrixHeight; i++ ) {
		ASSERT_NEAR( expected[i], result1[i], 1e-3 );
		ASSERT_TRUE( expectedIndices[i] == resultIndices[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineFindMaxValueInRowsTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineFindMaxValueInRowsTestInstantiation, CMathEngineFindMaxValueInRowsTest,
	::testing::Values(
		CTestParams(
			"MatrixHeight = (10..100);"
			"MatrixWidth = (10..100);"
			"Values = (-5000..5000);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineFindMaxValueInRowsTest, Random)
{
	RUN_TEST_IMPL(findMaxValueInRowsImpl)
}
