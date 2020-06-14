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

static void buildIntegerHistTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const int maxValue = params.GetValue<int>( "Values" );
	const CInterval bufferSizeInterval = params.GetInterval( "BufferSize" );

	const int bufferSize = random.UniformInt( bufferSizeInterval.Begin, bufferSizeInterval.End );

	std::vector<int> numbers;
	numbers.reserve( bufferSize );

	std::vector<int> expected;
	expected.insert( expected.begin(), maxValue, 0 );

	for( int i = 0; i < bufferSize; ++i ) {
		int number = random.UniformInt( -maxValue / 2, maxValue - 1 );
		if( number >= 0 ) {
			expected[number]++;
		}
		numbers.push_back( number );
	}

	CIntBlob numberBlob( MathEngine(), 1, 1, 1, bufferSize );
	numberBlob.CopyFrom( numbers.data() );

	CIntBlob resultBlob( MathEngine(), 1, 1, 1, maxValue );
	MathEngine().BuildIntegerHist( numberBlob.GetData(), bufferSize, resultBlob.GetData(), maxValue );

	std::vector<int> result;
	result.resize( maxValue );
	resultBlob.CopyTo( result.data() );

	for( int i = 0; i < maxValue; ++i ) {
		ASSERT_EQ( expected[i], result[i] ) << "at index " << i;
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CBuildIntegerHistTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CBuildIntegerHistTestInstantiation, CBuildIntegerHistTest,
	::testing::Values(
		CTestParams(
			"BufferSize = (1000..3000);"
			"Values = 100;"
			"TestCount = 100;"
		)
	)
);

TEST_P( CBuildIntegerHistTest, Random )
{
	RUN_TEST_IMPL( buildIntegerHistTestImpl )
}
