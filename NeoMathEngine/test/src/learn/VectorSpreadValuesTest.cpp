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

static void vectorSpreadValuesImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval vectorCountInterval = params.GetInterval( "VectorCount" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	const int vectorCount = random.UniformInt( vectorCountInterval.Begin, vectorCountInterval.End );

	std::vector<CFloatHandle> vectors;

	CREATE_FILL_FLOAT_ARRAY( source, vectorSizeInterval.Begin, vectorSizeInterval.End, vectorSize, random )
	CREATE_FILL_INT_ARRAY( indices, 0, vectorCount - 1, vectorSize, random )
	for( int j = 0; j < vectorCount; j++ ) {
		CREATE_FILL_FLOAT_ARRAY( vector, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
		CFloatHandle handle = CFloatHandle( MathEngine().HeapAllocTyped<float>( vectorSize ) );
		MathEngine().DataExchangeTyped<float>( handle, vector.data(), vectorSize );
		vectors.push_back( handle );
	}

	MathEngine().VectorSpreadValues( CARRAY_FLOAT_WRAPPER( source ), vectors.data(), vectorCount, CARRAY_INT_WRAPPER( indices ), vectorSize );

	for( int i = 0; i < vectorSize; i++ ) {
		const int vectorIndex = indices[i];
		ASSERT_FLOAT_EQ( vectors[vectorIndex].GetValueAt( i ), source[i] );
	}

	for( int i = 0; i < vectorCount; i++ ) {
		MathEngine().HeapFree( vectors[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CVectorSpreadValuesTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CVectorSpreadValuesTestInstantiation, CVectorSpreadValuesTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (10..100);"
			"Values = (-50..50);"
			"TestCount = 100;"
			"VectorCount = (5..10);"
		)
	)
);

TEST_P( CVectorSpreadValuesTest, Random )
{
	RUN_TEST_IMPL( vectorSpreadValuesImpl );
}
