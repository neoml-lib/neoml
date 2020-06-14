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

static void multiplyDiagMatrixByMatrixAndAddTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval matrixSizeInterval = params.GetInterval( "MatrixSize" );
	const CInterval matrixValuesInterval = params.GetInterval( "MatrixValues" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int diagMatrixSize = random.UniformInt( matrixSizeInterval.Begin, matrixSizeInterval.End );
	const int secondWidth = random.UniformInt( matrixSizeInterval.Begin, matrixSizeInterval.End );

	CREATE_FILL_FLOAT_ARRAY( diagMatrix, matrixValuesInterval.Begin, matrixValuesInterval.End, diagMatrixSize * batchSize, random )
	CREATE_FILL_FLOAT_ARRAY( second, matrixValuesInterval.Begin, matrixValuesInterval.End, batchSize * diagMatrixSize * secondWidth, random )
	CREATE_FILL_FLOAT_ARRAY( result, matrixValuesInterval.Begin, matrixValuesInterval.End, diagMatrixSize * secondWidth, random )

	std::vector<float> expected;
	expected = result;
	for( int b = 0; b < batchSize; b++ ) {
		for( int i = 0; i < diagMatrixSize; i++ ) {
			float mult = diagMatrix[b * diagMatrixSize + i];
			for( int j = 0; j < secondWidth; j++ ) {
				int index = b * diagMatrixSize * secondWidth + i * secondWidth + j;
				expected[i * secondWidth + j] += second[index] * mult;
			}
		}
	}
	
	MathEngine().MultiplyDiagMatrixByMatrixAndAdd( batchSize, CARRAY_FLOAT_WRAPPER( diagMatrix ), diagMatrixSize, CARRAY_FLOAT_WRAPPER( second ), secondWidth, CARRAY_FLOAT_WRAPPER( result ) );
	
	for( int i = 0; i < diagMatrixSize * secondWidth; i++ ) {
		ASSERT_NEAR( expected[i], result[i], 1e-03f );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineMultiplyDiagMatrixByMatrixAndAddTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineMultiplyDiagMatrixByMatrixAndAddTestInstantiation, CMathEngineMultiplyDiagMatrixByMatrixAndAddTest,
	::testing::Values(
		CTestParams(
			"BatchSize = (1..5);"
			"MatrixSize = (5..100);"
			"MatrixValues = (-50..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineMultiplyDiagMatrixByMatrixAndAddTest, Random)
{
	RUN_TEST_IMPL( multiplyDiagMatrixByMatrixAndAddTestImpl )
}
