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

static void multiplyTransposedMatrixByMatrixAndAddNaive( const std::vector<float>& first, const std::vector<float>& second,
	int firstHeight, int firstWidth, int secondWidth, std::vector<float>& result )
{
	for( int i = 0; i < firstWidth; ++i ) {
		for( int j = 0; j < secondWidth; ++j ) {
			for( int k = 0; k < firstHeight; ++k ) {
				result[i * secondWidth + j] += first[k * firstWidth + i] * second[k * secondWidth + j];
			}
		}
	}
}

static void multiplyTransposedMatrixByMatrixAndAddTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int firstHeight = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int firstWidth = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int secondWidth = random.UniformInt( widthInterval.Begin, widthInterval.End );

	CREATE_FILL_FLOAT_ARRAY( first, valuesInterval.Begin, valuesInterval.End, firstHeight * firstWidth, random )
		CREATE_FILL_FLOAT_ARRAY( second, valuesInterval.Begin, valuesInterval.End, firstHeight * secondWidth, random )

	CREATE_FILL_FLOAT_ARRAY( get, valuesInterval.Begin, valuesInterval.End, firstWidth * secondWidth, random )
	std::vector<float> expected;
	expected = get;

	multiplyTransposedMatrixByMatrixAndAddNaive( first, second, firstHeight, firstWidth, secondWidth, expected );

	MathEngine().MultiplyTransposedMatrixByMatrixAndAdd( CARRAY_FLOAT_WRAPPER( first ), firstHeight, firstWidth, firstWidth,
		CARRAY_FLOAT_WRAPPER( second ), secondWidth, secondWidth, CARRAY_FLOAT_WRAPPER( get ), secondWidth, firstWidth * secondWidth );

	for( int i = 0; i < firstWidth * secondWidth; ++i ) {
		ASSERT_NEAR( expected[i], get[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMultiplyTransposedMatrixByMatrixAndAddTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMultiplyTransposedMatrixByMatrixAndAddTestInstantiation, CMultiplyTransposedMatrixByMatrixAndAddTest,
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

TEST_P( CMultiplyTransposedMatrixByMatrixAndAddTest, Random )
{
	RUN_TEST_IMPL( multiplyTransposedMatrixByMatrixAndAddTestImpl )
}
