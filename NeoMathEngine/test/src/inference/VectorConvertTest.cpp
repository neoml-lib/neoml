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

static void vectorConvertFloatToIntTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );

	CREATE_FILL_FLOAT_ARRAY( fromArr, -100500.f, 123456.f, vectorSize, random );
	std::vector<int> toArr;
	toArr.resize( vectorSize );

	MathEngine().VectorConvert( CARRAY_FLOAT_WRAPPER( fromArr ), CARRAY_INT_WRAPPER( toArr ), vectorSize );
	for( int i = 0; i < vectorSize; ++i ) {
		ASSERT_EQ( static_cast<int>( fromArr[i] ), toArr[i] ) << fromArr[i];
	}
}

static void vectorConvertIntToFloatTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	
	CREATE_FILL_INT_ARRAY( fromArr, -100500, 123456, vectorSize, random );
	std::vector<float> toArr;
	toArr.resize( vectorSize );

	MathEngine().VectorConvert( CARRAY_INT_WRAPPER( fromArr ), CARRAY_FLOAT_WRAPPER( toArr ), vectorSize );
	for( int i = 0; i < vectorSize; ++i ) {
		ASSERT_NEAR( static_cast<float>( fromArr[i] ), toArr[i], 1e-5f ) << fromArr[i];
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineVectorConvertTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineVectorConvertTestInstantiation, CMathEngineVectorConvertTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (1..10000);"
			"TestCount = 100;"
		),
		CTestParams(
			"VectorSize = (1..1000);"
			"TestCount = 100;"
		),
		CTestParams(
			"VectorSize = (1179648..1179648);"
			"TestCount = 10;"
		)
	)
);

TEST_P( CMathEngineVectorConvertTest, FloatToIntRandom )
{
	RUN_TEST_IMPL( vectorConvertFloatToIntTestImpl );
}

TEST_P( CMathEngineVectorConvertTest, IntToFloatRandom )
{
	RUN_TEST_IMPL( vectorConvertIntToFloatTestImpl );
}
