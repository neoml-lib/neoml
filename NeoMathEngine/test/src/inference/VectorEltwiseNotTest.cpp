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

static void vectorEltwiseNotImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	CREATE_FILL_INT_ARRAY( vector, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
	CREATE_FILL_INT_ARRAY( result, valuesInterval.Begin, valuesInterval.End, vectorSize, random )

	MathEngine().VectorEltwiseNot( CARRAY_INT_WRAPPER( vector ), CARRAY_INT_WRAPPER( result ), vectorSize );

	for( int i = 0; i < vectorSize; i++ ) {
		const int expected = ( vector[i] == 0 ) ? 1 : 0;
		ASSERT_EQ( expected, result[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CVectorEltwiseNotTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CVectorEltwiseNotTestInstantiation, CVectorEltwiseNotTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (10..100);"
			"Values = (-50..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CVectorEltwiseNotTest, Random )
{
	RUN_TEST_IMPL( vectorEltwiseNotImpl );
}
