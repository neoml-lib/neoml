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

static void multiplyLookupMatrixByLookupVectorTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );

	const int lookupMatrixSize = random.UniformInt( heightInterval.Begin, 3 * heightInterval.End );
	CREATE_FILL_FLOAT_ARRAY( matrixTable, valuesInterval.Begin, valuesInterval.End, lookupMatrixSize * width, random )

	int lookupVectorSize = random.UniformInt( heightInterval.Begin, 2 * heightInterval.End );
	CREATE_FILL_FLOAT_ARRAY( vectorTable, valuesInterval.Begin, valuesInterval.End, lookupVectorSize * width, random )

	std::vector<float> matrixData;
	matrixData.resize( batchSize * height * width );
	CREATE_FILL_INT_ARRAY( matrix, 0, lookupMatrixSize - 1, batchSize * height, random )
	for( size_t i = 0; i < matrix.size(); ++i ) {
		for( int j = 0; j < width; ++j ) {
			matrixData[i * width + j] = matrixTable[matrix[i] * width + j];
		}
	}

	std::vector<float> vectorData;
	vectorData.resize( batchSize * width );
	CREATE_FILL_INT_ARRAY( vector, 0, lookupVectorSize - 1, batchSize, random )
	for( size_t i = 0; i < vector.size(); ++i ) {
		for( int j = 0; j < width; ++j ) {
			vectorData[i * width + j] = vectorTable[vector[i] * width + j];
		}
	}

	std::vector<float> res0;
	res0.resize( batchSize * height );
	MathEngine().MultiplyLookupMatrixByLookupVector( batchSize,
		CLookupMatrix( CARRAY_FLOAT_WRAPPER( matrixTable ), lookupMatrixSize, width, CARRAY_INT_WRAPPER( matrix ), height ),
		CLookupVector( CARRAY_FLOAT_WRAPPER( vectorTable ), lookupMatrixSize, width, CARRAY_INT_WRAPPER( vector ) ),
		CARRAY_FLOAT_WRAPPER( res0 ), batchSize * height );

	std::vector<float> res1;
	res1.resize( batchSize * height );
	MathEngine().MultiplyMatrixByMatrix( batchSize, CARRAY_FLOAT_WRAPPER( matrixData ), height, width,
		CARRAY_FLOAT_WRAPPER( vectorData ), 1, CARRAY_FLOAT_WRAPPER( res1 ), batchSize * height );

	for( size_t i = 0; i < res0.size(); ++i ) {
		ASSERT_NEAR( res0[i], res1[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMultiplyLookupMatrixByLookupVectorTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMultiplyLookupMatrixByLookupVectorTestInstantiation, CMultiplyLookupMatrixByLookupVectorTest,
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

TEST_P( CMultiplyLookupMatrixByLookupVectorTest, Random )
{
	RUN_TEST_IMPL( multiplyLookupMatrixByLookupVectorTestImpl )
}
