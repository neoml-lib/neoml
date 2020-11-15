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

static void bitSetBinarizationIntTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval bitSetSizeInterval = params.GetInterval( "BitSetSize" );
	
	const int BitsPerElement = sizeof( int ) * CHAR_BIT;

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int bitSetSize = random.UniformInt( bitSetSizeInterval.Begin, bitSetSizeInterval.End );
	const int outputVectorSize = random.UniformInt( 1, bitSetSize ) * BitsPerElement;

	std::vector<int> source;
	source.resize( batchSize * bitSetSize );
	std::vector<float> expected;
	expected.resize( batchSize * outputVectorSize );

	size_t resIndex = 0;
	for( size_t i = 0; i < source.size(); i++ ) {
		unsigned int value = random.Next();
		source[i] = value;
		int sourceObjIndex = i % bitSetSize;
		if (sourceObjIndex >= outputVectorSize / BitsPerElement) {
			continue;
		}
		for( size_t j = 0; j < BitsPerElement; j++ ) {
			if( resIndex + j < expected.size() ) {
				if( ( value & ( 1 << j ) ) != 0 ) {
					expected[resIndex + j] = 1.0;
				} else {
					expected[resIndex + j] = 0.0;
				}
			}
		}
		resIndex += BitsPerElement;
	}

	std::vector<float> result;
	result.resize( batchSize * outputVectorSize );
	MathEngine().BitSetBinarization( batchSize, bitSetSize, CARRAY_INT_WRAPPER( source ), outputVectorSize, CARRAY_FLOAT_WRAPPER( result ) );

	for( size_t i = 0; i < result.size(); i++ ) {
		ASSERT_NEAR( expected[i], result[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBitSetBinarizationTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineBitSetBinarizationTestInstantiation, CMathEngineBitSetBinarizationTest,
	::testing::Values(
		CTestParams(
			"BatchSize = (1..50);"
			"BitSetSize = (1..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineBitSetBinarizationTest, Random)
{
	RUN_TEST_IMPL(bitSetBinarizationIntTestImpl)
}
