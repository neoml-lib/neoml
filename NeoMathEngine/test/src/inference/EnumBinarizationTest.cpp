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

static void enumBinarizationFloatTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval enumSizeInterval = params.GetInterval( "EnumSize" );
	
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int enumSize = random.UniformInt( enumSizeInterval.Begin, enumSizeInterval.End );

	CREATE_FILL_FLOAT_ARRAY( source, 0, enumSize - 1, batchSize, random )
	
	std::vector<float> expected;
	expected.insert( expected.begin(), batchSize * enumSize, 0.0f );
	for( int i = 0; i < batchSize; ++i ) {
		int enumValue = (int)source[i];
		if(enumValue >= 0) {
			expected[i * enumSize + enumValue] = 1;
		}
	}

	std::vector<float> result;
	result.resize( batchSize * enumSize );
	MathEngine().EnumBinarization( batchSize, CARRAY_FLOAT_WRAPPER( source ), enumSize, CARRAY_FLOAT_WRAPPER( result ) );

	for( size_t i = 0; i < result.size(); i++ ) {
		ASSERT_NEAR( expected[i], result[i], 1e-3 );
	}
}

static void enumBinarizationIntTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval enumSizeInterval = params.GetInterval( "EnumSize" );
	
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int enumSize = random.UniformInt( enumSizeInterval.Begin, enumSizeInterval.End );

	CREATE_FILL_INT_ARRAY( source, 0, enumSize - 1, batchSize, random )
	
	std::vector<float> expected;
	expected.insert( expected.begin(), batchSize * enumSize, 0 );
	for( int i = 0; i < batchSize; ++i ) {
		int enumValue = source[i];
		if(enumValue >= 0) {
			expected[i * enumSize + enumValue] = 1;
		}
	}

	std::vector<float> result;
	result.resize( batchSize * enumSize );
	MathEngine().EnumBinarization( batchSize, CARRAY_INT_WRAPPER( source ), enumSize, CARRAY_FLOAT_WRAPPER( result ) );

	for( size_t i = 0; i < result.size(); i++ ) {
		ASSERT_NEAR( expected[i], result[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineEnumBinarizationTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineEnumBinarizationTestInstantiation, CMathEngineEnumBinarizationTest,
	::testing::Values(
		CTestParams(
			"BatchSize = (1..100);"
			"EnumSize = (1..100);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineEnumBinarizationTest, Random)
{
	RUN_TEST_IMPL(enumBinarizationIntTestImpl)
	RUN_TEST_IMPL(enumBinarizationFloatTestImpl)
}
