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

template<class T>
static void vectorEltwiseDivideImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval vectorValuesInterval = params.GetInterval( "VectorValues" );
	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );

	CREATE_FILL_ARRAY( T, a, vectorValuesInterval.Begin, vectorValuesInterval.End, vectorSize, random )
	CREATE_FILL_ARRAY( T, b, vectorValuesInterval.Begin, vectorValuesInterval.End, vectorSize, random )

	for( T& denominator : b ) {
		while( denominator == 0 ) {
			denominator = static_cast<T>( random.Uniform( vectorValuesInterval.Begin, vectorValuesInterval.End ) );
		}
	}

	std::vector<T> result;
	result.resize( vectorSize );
	MathEngine().VectorEltwiseDivide( CARRAY_WRAPPER( T, a ), CARRAY_WRAPPER( T, b ), CARRAY_WRAPPER( T, result ), vectorSize );

	for( int i = 0; i < vectorSize; i++ ) {
		T expected = a[i] / b[i];
		ASSERT_NEAR( expected, result[i], 5e-3 ) << "\t " << a[i] << " / " << b[i] << "\tat index " << i;
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineVectorEltwiseDivideTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineVectorEltwiseDivideTestInstantiation, CMathEngineVectorEltwiseDivideTest,
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
			"VectorValues = (-10..10);"
			"TestCount = 10;"
		)
	)
);

TEST_P( CMathEngineVectorEltwiseDivideTest, Random )
{
	RUN_TEST_IMPL( vectorEltwiseDivideImpl<float> );
	RUN_TEST_IMPL( vectorEltwiseDivideImpl<int> );
}
