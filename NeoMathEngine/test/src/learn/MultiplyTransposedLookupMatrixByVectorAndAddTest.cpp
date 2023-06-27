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
#include <MeTestCommon.h>

using namespace NeoML;
using namespace NeoMLTest;

static void multiplyTransposedLookupMatrixByVectorAndAddTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );

	int lookupMatrixSize = random.UniformInt( heightInterval.Begin, 3 * heightInterval.End );
	CREATE_FILL_FLOAT_ARRAY( matrixTable, valuesInterval.Begin, valuesInterval.End, lookupMatrixSize * width, random )

	std::vector<float> matrixData;
	matrixData.resize( batchSize * height * width );
	std::vector<int> rows;
	rows.resize( batchSize * height );
	for( int i = 0; i < batchSize * height; ++i ) {
		rows[i] = random.UniformInt( 0, lookupMatrixSize - 1 );
		for( int j = 0; j < width; ++j ) {
			matrixData[i * width + j] = matrixTable[rows[i] * width + j];
		}
	}

	CREATE_FILL_FLOAT_ARRAY( vector, valuesInterval.Begin, valuesInterval.End, batchSize * height, random )

	{
		CREATE_FILL_FLOAT_ARRAY( get, valuesInterval.Begin, valuesInterval.End, batchSize * width, random )
		std::vector<float> expected;
		expected = get;
		MathEngine().MultiplyTransposedLookupMatrixByVectorAndAdd( batchSize,
			CLookupMatrix( CARRAY_FLOAT_WRAPPER( matrixTable ), lookupMatrixSize, width, CARRAY_INT_WRAPPER( rows ), height ),
			CARRAY_FLOAT_WRAPPER( vector ), CARRAY_FLOAT_WRAPPER( get ), batchSize * width );

		batchMultiplyMatrixByMatrixAndAddNaive( batchSize, vector, matrixData, 1, height, width, expected );

		for( size_t i = 0; i < expected.size(); ++i ) {
			ASSERT_NEAR( get[i], expected[i], 1e-3 );
		}
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMultiplyTransposedLookupMatrixByVectorAndAddTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMultiplyTransposedLookupMatrixByVectorAndAddTestInstantiation, CMultiplyTransposedLookupMatrixByVectorAndAddTest,
	::testing::Values(
		CTestParams(
			"Height = (1..50);"
			"Width = (1..50);"
			"BatchSize = (1..5);"
			"VectorSize = (1..20);"
			"Values = (-1..1);"
			"Channels = (1..5);"
			"TestCount = 100;"
		),
		CTestParams(
			"Height = (100..500);"
			"Width = (100..500);"
			"BatchSize = (1..5);"
			"VectorSize = (30..50);"
			"Values = (-1..1);"
			"Channels = (1..5);"
			"TestCount = 5;"
		)
	)
);

TEST_P( CMultiplyTransposedLookupMatrixByVectorAndAddTest, Random )
{
	RUN_TEST_IMPL( multiplyTransposedLookupMatrixByVectorAndAddTestImpl )
}
