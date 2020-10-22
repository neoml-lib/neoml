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

static void addVectorToMatrixColumnsFloatTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval matrixHeightInterval = params.GetInterval( "MatrixHeight" );
	const CInterval matrixWidthInterval = params.GetInterval( "MatrixWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int matrixHeight = random.UniformInt( matrixHeightInterval.Begin, matrixHeightInterval.End );
	const int matrixWidth = random.UniformInt( matrixWidthInterval.Begin, matrixWidthInterval.End );

	CREATE_FILL_FLOAT_ARRAY( vector, valuesInterval.Begin, valuesInterval.End, matrixHeight, random )
	CREATE_FILL_FLOAT_ARRAY( matrix, valuesInterval.Begin, valuesInterval.End, matrixHeight * matrixWidth, random )

	std::vector<float> expected;
	expected.resize( matrixHeight * matrixWidth );

	int index = 0;
	for(int j = 0; j < matrixHeight; ++j) {
		float val = vector[j];
		for(int i = 0; i < matrixWidth; ++i, ++index) {
			expected[index] = matrix[index] + val;
		}
	}

	std::vector<float> result;
	result.resize( matrixHeight * matrixWidth );
	MathEngine().AddVectorToMatrixColumns( CARRAY_FLOAT_WRAPPER( matrix ), CARRAY_FLOAT_WRAPPER( result ), matrixHeight, matrixWidth, CARRAY_FLOAT_WRAPPER( vector ) );

	for( size_t i = 0; i < result.size(); ++i ) {
		ASSERT_NEAR( expected[i], result[i], 1e-3 );
	}
}

static void addVectorToMatrixColumnsIntTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval matrixHeightInterval = params.GetInterval( "MatrixHeight" );
	const CInterval matrixWidthInterval = params.GetInterval( "MatrixWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int matrixHeight = random.UniformInt( matrixHeightInterval.Begin, matrixHeightInterval.End );
	const int matrixWidth = random.UniformInt( matrixWidthInterval.Begin, matrixWidthInterval.End );

	CREATE_FILL_INT_ARRAY( vector, valuesInterval.Begin, valuesInterval.End, matrixHeight, random )
	CREATE_FILL_INT_ARRAY( matrix, valuesInterval.Begin, valuesInterval.End, matrixHeight * matrixWidth, random )

	std::vector<int> expected;
	expected.resize( matrixHeight * matrixWidth );

	int index = 0;
	for(int j = 0; j < matrixHeight; ++j) {
		int val = vector[j];
		for(int i = 0; i < matrixWidth; ++i, ++index) {
			expected[index] = matrix[index] + val;
		}
	}

	std::vector<int> result;
	result.resize( matrixHeight * matrixWidth );
	MathEngine().AddVectorToMatrixColumns( CARRAY_INT_WRAPPER( matrix ), CARRAY_INT_WRAPPER( result ), matrixHeight, matrixWidth, CARRAY_INT_WRAPPER( vector ) );

	for(size_t i = 0; i < result.size(); ++i) {
		EXPECT_EQ( expected[i], result[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineAddVectorToMatrixColumnsTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineAddVectorToMatrixColumnsTestInstantiation, CMathEngineAddVectorToMatrixColumnsTest,
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

TEST_P(CMathEngineAddVectorToMatrixColumnsTest, Random)
{
	RUN_TEST_IMPL(addVectorToMatrixColumnsIntTestImpl)
	RUN_TEST_IMPL(addVectorToMatrixColumnsFloatTestImpl)
}
