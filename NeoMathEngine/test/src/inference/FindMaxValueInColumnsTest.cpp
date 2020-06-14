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

static void findMaxValueInColumnsImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval matrixHeightInterval = params.GetInterval( "MatrixHeight" );
	const CInterval matrixWidthInterval = params.GetInterval( "MatrixWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );

	const int matrixHeight = random.UniformInt( matrixHeightInterval.Begin, matrixHeightInterval.End );
	const int matrixWidth = random.UniformInt( matrixWidthInterval.Begin, matrixWidthInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );

	CREATE_FILL_FLOAT_ARRAY( matrix, valuesInterval.Begin, valuesInterval.End, batchSize * matrixHeight * matrixWidth, random )

	std::vector<float> expected;
	std::vector<int> expectedIndices;
	expected.resize( batchSize * matrixWidth );
	expectedIndices.resize( batchSize * matrixWidth );
	for( int b = 0; b < batchSize; b++ ) {
		for( int col = 0; col < matrixWidth; col++ ) {
			int startIndex = b * matrixHeight * matrixWidth + col;
			float maxValue = matrix[startIndex];
			int maxIndex = 0;
			for( int row = 1; row < matrixHeight; row++ ) {
				const float curValue = matrix[startIndex + row * matrixWidth];
				if( curValue > maxValue ) {
					maxValue = curValue;
					maxIndex = row;
				}
			}
			expected[b * matrixWidth + col] = maxValue;
			expectedIndices[b * matrixWidth + col] = maxIndex;
		}
	}

	std::vector<float> result;
	result.resize( batchSize * matrixWidth );
	std::vector<int> resultIndices;
	resultIndices.resize( batchSize * matrixWidth );
	MathEngine().FindMaxValueInColumns( batchSize, CARRAY_FLOAT_WRAPPER( matrix ), matrixHeight, matrixWidth, CARRAY_FLOAT_WRAPPER( result ), CARRAY_INT_WRAPPER( resultIndices ), batchSize * matrixWidth );

	for( int i = 0; i < batchSize * matrixWidth; i++ ) {
		ASSERT_NEAR( expected[i], result[i], 1e-3 );
		ASSERT_EQ( expectedIndices[i], resultIndices[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineFindMaxValueInColumnsTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineFindMaxValueInColumnsTestInstantiation, CMathEngineFindMaxValueInColumnsTest,
	::testing::Values(
		CTestParams(
			"BatchSize = (1..15);"
			"MatrixHeight = (1..10);"
			"MatrixWidth = (1..10);"
			"Values = (-50..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineFindMaxValueInColumnsTest, Random)
{
	RUN_TEST_IMPL(findMaxValueInColumnsImpl)
}
