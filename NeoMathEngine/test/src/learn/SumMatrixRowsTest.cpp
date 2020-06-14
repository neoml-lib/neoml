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

static void sumMatrixRowsAddNaive( std::vector<float>& vector, const std::vector<float>& matrix, int batchSize, int height, int width )
{
	for( int b = 0; b < batchSize; ++b ) {
		for( int h = 0; h < height; ++h ) {
			for( int w = 0; w < width; ++w ) {
				vector[b * width + w] += matrix[b * height * width +  h * width + w];
			}
		}
	}
}

static void sumMatrixRowsTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );

	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );

	CREATE_FILL_FLOAT_ARRAY( matrix, valuesInterval.Begin, valuesInterval.End, batchSize * height * width, random )
	CREATE_FILL_FLOAT_ARRAY( getVector, valuesInterval.Begin, valuesInterval.End, batchSize * width, random )
	std::vector<float> expectedVector;
	expectedVector = getVector;

	for( size_t i = 0; i < expectedVector.size(); ++i ) {
		expectedVector[i] = 0.f;
	}
	MathEngine().SumMatrixRows( batchSize, CARRAY_FLOAT_WRAPPER( getVector ), CARRAY_FLOAT_WRAPPER( matrix ), height, width );
	sumMatrixRowsAddNaive( expectedVector, matrix, batchSize, height, width );

	for( int i = 0; i < batchSize * width; ++i ) {
		ASSERT_NEAR( expectedVector[i], getVector[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CSumMatrixRowsTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CSumMatrixRowsTestInstantiation, CSumMatrixRowsTest,
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

TEST_P( CSumMatrixRowsTest, Random )
{
	RUN_TEST_IMPL( sumMatrixRowsTestImpl )
}
