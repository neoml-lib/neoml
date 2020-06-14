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

static void batchAddVectorToMatrixRowsTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval matrixHeightInterval = params.GetInterval( "MatrixHeight" );
	const CInterval matrixWidthInterval = params.GetInterval( "MatrixWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int matrixHeight = random.UniformInt( matrixHeightInterval.Begin, matrixHeightInterval.End );
	const int matrixWidth = random.UniformInt( matrixWidthInterval.Begin, matrixWidthInterval.End );

	CREATE_FILL_FLOAT_ARRAY( vector, valuesInterval.Begin, valuesInterval.End, batchSize * matrixWidth, random )
	CREATE_FILL_FLOAT_ARRAY( matrix, valuesInterval.Begin, valuesInterval.End, batchSize * matrixHeight * matrixWidth, random )

	std::vector<float> expected;
	expected.resize( batchSize * matrixHeight * matrixWidth );

	for( int b = 0; b < batchSize; ++b ) {
		int index = 0;
		for(int j = 0; j < matrixHeight; ++j) {
			for(int i = 0; i < matrixWidth; ++i, ++index) {
				expected[b * matrixHeight * matrixWidth + index] = matrix[b * matrixHeight * matrixWidth + index] + vector[b * matrixWidth + i];
			}
		}
	}

	std::vector<float> result;
	result.resize( batchSize * matrixHeight * matrixWidth );
	MathEngine().AddVectorToMatrixRows( batchSize, CARRAY_FLOAT_WRAPPER( matrix ), CARRAY_FLOAT_WRAPPER( result ), matrixHeight, matrixWidth, CARRAY_FLOAT_WRAPPER( vector ) );

	for(size_t i = 0; i < result.size(); ++i) {
		ASSERT_NEAR( expected[i], result[i], 1e-3);
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineAddVectorToMatrixRowsTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineAddVectorToMatrixRowsTestInstantiation, CMathEngineAddVectorToMatrixRowsTest,
	::testing::Values(
		CTestParams(
			"BatchSize = (1..10);"
			"MatrixHeight = (1..100);"
			"MatrixWidth = (1..100);"
			"Values = (-50..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineAddVectorToMatrixRowsTest, Random)
{
	RUN_TEST_IMPL(batchAddVectorToMatrixRowsTestImpl)
}
