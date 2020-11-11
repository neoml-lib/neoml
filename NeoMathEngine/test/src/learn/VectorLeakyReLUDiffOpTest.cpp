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

static void vectorLeakyReLUDiffOpImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	CREATE_FILL_FLOAT_ARRAY( a, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
	CREATE_FILL_FLOAT_ARRAY( b, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
	std::vector<float> result;
	result.resize( vectorSize );
	std::vector<float> alpha = { static_cast<float>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) ) };

	MathEngine().VectorLeakyReLUDiffOp( CARRAY_FLOAT_WRAPPER( a ), CARRAY_FLOAT_WRAPPER( b ), CARRAY_FLOAT_WRAPPER( result ), vectorSize, CARRAY_FLOAT_WRAPPER( alpha ) );

	for( int i = 0; i < vectorSize; i++ ) {
		float expected = a[i] > 0 ? b[i] : alpha[0] * b[i];
		ASSERT_NEAR( expected, result[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CVectorLeakyReLUDiffOpTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CVectorLeakyReLUDiffOpTestInstantiation, CVectorLeakyReLUDiffOpTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (10..100);"
			"Values = (-50..50);"
			"TestCount = 100;"
			"VectorCount = (5..10);"
		)
	)
);

TEST_P( CVectorLeakyReLUDiffOpTest, Random )
{
	RUN_TEST_IMPL( vectorLeakyReLUDiffOpImpl );
}
