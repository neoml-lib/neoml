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

static void lookupAndSumImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval indexCountInterval = params.GetInterval( "IndexCount" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	
	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	const int indexCount = random.UniformInt( indexCountInterval.Begin, indexCountInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );

	CREATE_FILL_FLOAT_ARRAY( table, valuesInterval.Begin, valuesInterval.End, vectorSize * indexCount, random )
	CREATE_FILL_INT_ARRAY( indices, -1, indexCount - 1, indexCount * batchSize, random )

	std::vector<float> expected;
	expected.reserve( batchSize * vectorSize );
	for( int b = 0; b < batchSize; b++ ) {
		std::vector<float> curBatchSum;
		curBatchSum.resize( vectorSize );
		for( int j = 0; j < vectorSize; j++ ) {
			curBatchSum[j] = 0.;
		}

		for( int i = 0; i < indexCount; i++ ) {
			int index = indices[b * indexCount + i];
			if( index >= 0 ) {
				for( int j = 0; j < vectorSize; j++ ) {
					curBatchSum[j] += table[vectorSize * index + j];
				}
			}
		}
		expected.insert( expected.end(), curBatchSum.begin(), curBatchSum.end() );
	}

	std::vector<float> result;
	result.resize( batchSize * vectorSize );
	MathEngine().LookupAndSum( CARRAY_INT_WRAPPER( indices ), batchSize, indexCount, CARRAY_FLOAT_WRAPPER( table ), vectorSize, CARRAY_FLOAT_WRAPPER( result ) );

	for( int i = 0; i < batchSize * vectorSize; i++ ) {
		ASSERT_NEAR( expected[i], result[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineLookupAndSumTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineLookupAndSumTestInstantiation, CMathEngineLookupAndSumTest,
	::testing::Values(
		CTestParams(
			"IndexCount = (10..100);"
			"VectorSize = (10..100);"
			"BatchSize = (1..5);"
			"Values = (-50..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineLookupAndSumTest, Random)
{
	RUN_TEST_IMPL(lookupAndSumImpl)
}
