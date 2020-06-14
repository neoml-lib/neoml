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

static void vectorMultiplyImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval vectorValuesInterval = params.GetInterval( "VectorValues" );
	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );

	float mult = static_cast<float>( random.Uniform( vectorValuesInterval.Begin, vectorValuesInterval.End ) );
	CREATE_FILL_FLOAT_ARRAY( a, vectorValuesInterval.Begin, vectorValuesInterval.End, vectorSize, random )

	std::vector<float> result;
	result.resize( vectorSize );
	MathEngine().VectorMultiply( CARRAY_FLOAT_WRAPPER( a ), CARRAY_FLOAT_WRAPPER( result ), vectorSize, FLOAT_WRAPPER( &mult ) );

	for( int i = 0; i < vectorSize; i++ ) {
		float expected = a[i] * mult;
		ASSERT_NEAR( expected, result[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineVectorMultiplyTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineVectorMultiplyTestInstantiation, CMathEngineVectorMultiplyTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (1..10000);"
			"VectorValues = (-50..50);"
			"TestCount = 100;"
		),
		CTestParams(
			"VectorSize = (1..1000);"
			"VectorValues = (-10..10);"
			"TestCount = 100;"
		),
		CTestParams(
			"VectorSize = (1179648..1179648);"
			"VectorValues = (-1..1);"
			"TestCount = 10;"
		)
	)
);

TEST_P( CMathEngineVectorMultiplyTest, Random )
{
	RUN_TEST_IMPL( vectorMultiplyImpl );
}
