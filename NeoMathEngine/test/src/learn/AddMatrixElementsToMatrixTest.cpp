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

static void addMatrixElementsToMatrixTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );

	CREATE_FILL_FLOAT_ARRAY( matrixToAdd, valuesInterval.Begin, valuesInterval.End, height * width, random )
	CREATE_FILL_FLOAT_ARRAY( actualMatrix, valuesInterval.Begin, valuesInterval.End, height * width, random )
	CREATE_FILL_INT_ARRAY( indices, 0, width - 1, height, random )
	std::vector<float> expectedMatrix = actualMatrix;

	MathEngine().AddMatrixElementsToMatrix( CARRAY_FLOAT_WRAPPER( matrixToAdd ), height, width, CARRAY_FLOAT_WRAPPER( actualMatrix ), CARRAY_INT_WRAPPER( indices ) );

	for( int i = 0; i < height; ++i ) {
		expectedMatrix[i * width + indices[i]] += matrixToAdd[i * width + indices[i]];
	}

	for( int i = 0; i < height * width; ++i ) {
		ASSERT_NEAR( expectedMatrix[i], actualMatrix[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CAddMatrixElementsToMatrixTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CAddMatrixElementsToMatrixTestInstantiation, CAddMatrixElementsToMatrixTest,
	::testing::Values(
		CTestParams(
			"Height = (1..100);"
			"Width = (1..100);"
			"Values = (-10..10);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CAddMatrixElementsToMatrixTest, Random )
{
	RUN_TEST_IMPL( addMatrixElementsToMatrixTestImpl )
}
