/* Copyright © 2017-2022 ABBYY Production LLC

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

template<class T>
static void vectorEltwiseEqualImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	CREATE_FILL_ARRAY( T, first, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
	CREATE_FILL_ARRAY( T, second, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
	// Guarantee some matches in float case
	for( size_t i = 0; i < first.size(); ++i ) {
		if( random.Next() % 5 == 3 ) {
			second[i] = first[i];
		}
	}
	CREATE_FILL_INT_ARRAY( result, valuesInterval.Begin, valuesInterval.End, vectorSize, random )

	MathEngine().VectorEltwiseEqual( CARRAY_WRAPPER( T, first ), CARRAY_WRAPPER( T, second ),
		CARRAY_INT_WRAPPER( result ), vectorSize );

	for( int i = 0; i < vectorSize; i++ ) {
		const int expected = first[i] == second[i] ? 1 : 0;
		ASSERT_EQ( expected, result[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CVectorEltwiseEqualTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CVectorEltwiseEqualTestInstantiation, CVectorEltwiseEqualTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (10..100);"
			"Values = (-50..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CVectorEltwiseEqualTest, Random )
{
	RUN_TEST_IMPL( vectorEltwiseEqualImpl<float> );
	RUN_TEST_IMPL( vectorEltwiseEqualImpl<int> );
}
