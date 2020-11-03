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

static void findMinValueInColumnsTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );

	CREATE_FILL_FLOAT_ARRAY( matrix, valuesInterval.Begin, valuesInterval.End, height * width, random )

	std::vector<int> expectedIndices, getIndices;
	std::vector<float> expected, get;
	getIndices.resize( width );
	expectedIndices.resize( width );
	expected.insert( expected.begin(), width, static_cast<float>( valuesInterval.End ) );
	get.resize( width );

	for( int j = 0; j < height; ++j ) {
		for( int i = 0; i < width; ++i ) {
			if( matrix[j * width + i] < expected[i] ) {
				expected[i] = matrix[j * width + i];
				expectedIndices[i] = j;
			}
		}
	}

	MathEngine().FindMinValueInColumns( CARRAY_FLOAT_WRAPPER( matrix ), height, width, CARRAY_FLOAT_WRAPPER( get ), CARRAY_INT_WRAPPER( getIndices ) );

	for( int i = 0; i < width; ++i ) {
		ASSERT_NEAR( expected[i], get[i], 1e-3 );
		ASSERT_EQ( expectedIndices[i], getIndices[i] );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CFindMinValueInColumnsTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CFindMinValueInColumnsTestInstantiation, CFindMinValueInColumnsTest,
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

TEST_P( CFindMinValueInColumnsTest, Random )
{
	RUN_TEST_IMPL( findMinValueInColumnsTestImpl )
}
