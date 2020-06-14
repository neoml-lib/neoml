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

static void filterSmallValuesImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const float threshold = static_cast<float>( random.Uniform( 0.01, valuesInterval.End ) );
	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	CREATE_FILL_FLOAT_ARRAY( vector, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
	std::vector<float> vectorCopy;
	vectorCopy = vector;

	MathEngine().FilterSmallValues( CARRAY_FLOAT_WRAPPER( vector ), vectorSize, threshold );

	for( int i = 0; i < vectorSize; i++ ) {
		if( fabs( vectorCopy[i] ) < threshold ) {
			ASSERT_TRUE( FloatEq( vector[i], 0 ) );
		}
	}
}

//------------------------------------------------------------------------------------------------------------

class CFilterSmallValuesTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CFilterSmallValuesTestInstantiation, CFilterSmallValuesTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (10..100);"
			"Values = (-50..50);"
			"TestCount = 100;"
			"VectorCount = (5..10);"
		)
	)
);

TEST_P( CFilterSmallValuesTest, Random )
{
	RUN_TEST_IMPL( filterSmallValuesImpl );
}
