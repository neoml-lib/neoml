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

static void vectorFindMaxValueInSetImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval vectorCountInterval = params.GetInterval( "VectorCount" );
	const CInterval vectorValuesInterval = params.GetInterval( "VectorValues" );

	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	const int vectorCount = random.UniformInt( vectorCountInterval.Begin, vectorCountInterval.End );

	std::vector<float> expected;
	std::vector<int> expectedIndices( vectorSize );
	expected.insert( expected.begin(), vectorSize, (float)vectorValuesInterval.Begin - 1 );

	std::vector<CConstFloatHandle> vectors;
	std::vector<float> vector( vectorSize );
	for( int i = 0; i < vectorCount; i++ ) {
		for( int j = 0; j < vectorSize; j++ ) {
			vector[j] = static_cast<float>( random.Uniform( vectorValuesInterval.Begin, vectorValuesInterval.End ) );
			if( expected[j] < vector[j] ) {
				expected[j] = vector[j];
				expectedIndices[j] = i;
			}
		}
		CFloatHandle constHandle = CFloatHandle( MathEngine().HeapAllocTyped<float>( vectorSize ) );
		MathEngine().DataExchangeTyped<float>( constHandle, vector.data(), vectorSize );
		vectors.emplace_back( constHandle );
	}

	std::vector<float> actual( vectorSize );
	MathEngine().VectorFindMaxValueInSet( vectors.data(), vectorCount, CARRAY_FLOAT_WRAPPER( actual ), vectorSize );

	for( int i = 0; i < vectorSize; i++ ) {
		FloatEq( expected[i], actual[i] );
	}

	std::vector<int> actualIndices( vectorSize );
	MathEngine().VectorFindMaxValueInSet( vectors.data(), vectorCount, CARRAY_FLOAT_WRAPPER( actual ), CARRAY_INT_WRAPPER( actualIndices ), vectorSize );

	for( int i = 0; i < vectorSize; i++ ) {
		FloatEq( expected[i], actual[i] );
		ASSERT_EQ( expectedIndices[i], actualIndices[i] );
	}

	for( int i = 0; i < vectorCount; i++ ) {
		MathEngine().HeapFree( CFloatHandle( vectors[i] ) );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineVectorFindMaxValueInSetTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineVectorFindMaxValueInSetTestInstantiation, CMathEngineVectorFindMaxValueInSetTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (1..100);"
			"VectorCount = (1..100);"
			"VectorValues = (-20..20);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CMathEngineVectorFindMaxValueInSetTest, Random )
{
	RUN_TEST_IMPL( vectorFindMaxValueInSetImpl );
}
