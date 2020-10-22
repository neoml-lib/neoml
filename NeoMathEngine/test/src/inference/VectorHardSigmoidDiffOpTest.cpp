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

static void vectorHardSigmoidDiffOpImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval vectorValuesInterval = params.GetInterval( "VectorValues" );
	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );

	CREATE_FILL_FLOAT_ARRAY( a, vectorValuesInterval.Begin, vectorValuesInterval.End, vectorSize, random )
	CREATE_FILL_FLOAT_ARRAY( b, vectorValuesInterval.Begin, vectorValuesInterval.End, vectorSize, random )

	float slope = static_cast<float>( random.Uniform( 0.1, 0.9 ) );
	float bias = static_cast<float>( random.Uniform( -1, 1 ) );
	std::vector<float> result;
	result.resize( vectorSize );

	MathEngine().VectorHardSigmoidDiffOp( CARRAY_FLOAT_WRAPPER( a ), CARRAY_FLOAT_WRAPPER( b ),CARRAY_FLOAT_WRAPPER( result ), vectorSize,
		FLOAT_WRAPPER( &slope ), FLOAT_WRAPPER( &bias ) );

	for( int i = 0; i < vectorSize; i++ ) {
		float expected;
		if( a[i] <= 0 || a[i] >= 1 ) {
			expected = 0;
		} else {
			expected = b[i] * slope;
		}
		ASSERT_NEAR( expected, result[i], 1e-5 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineVectorHardSigmoidDiffOpTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineVectorHardSigmoidDiffOpTestInstantiation, CMathEngineVectorHardSigmoidDiffOpTest,
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

TEST_P( CMathEngineVectorHardSigmoidDiffOpTest, Random )
{
	RUN_TEST_IMPL( vectorHardSigmoidDiffOpImpl );
}
