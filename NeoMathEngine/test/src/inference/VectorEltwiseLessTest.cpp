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

template<class TSrc, class TDst>
static void vectorEltwiseLessImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	CREATE_FILL_ARRAY( TSrc, first, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
	CREATE_FILL_ARRAY( TSrc, second, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
	CREATE_FILL_ARRAY( TDst, result, valuesInterval.Begin, valuesInterval.End, vectorSize, random )

	MathEngine().VectorEltwiseLess( CARRAY_WRAPPER( TSrc, first ), CARRAY_WRAPPER( TSrc, second ),
		CARRAY_WRAPPER( TDst, result ), vectorSize );

	for( int i = 0; i < vectorSize; i++ ) {
		const TDst expected = static_cast<TDst>( first[i] < second[i] ? 1 : 0 );
		ASSERT_EQ( expected, result[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CVectorEltwiseLessTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CVectorEltwiseLessTestInstantiation, CVectorEltwiseLessTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (10..100);"
			"Values = (-50..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CVectorEltwiseLessTest, Random )
{
#define VecEltwiseLessTestComma ,
	RUN_TEST_IMPL( vectorEltwiseLessImpl<float VecEltwiseLessTestComma float> );
	RUN_TEST_IMPL( vectorEltwiseLessImpl<float VecEltwiseLessTestComma int> );
	RUN_TEST_IMPL( vectorEltwiseLessImpl<int VecEltwiseLessTestComma int> );
}
