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
#include <MeTestCommon.h>

using namespace NeoML;
using namespace NeoMLTest;

static void vectorFillBernoulliTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval valueInterval = params.GetInterval( "Value" );
	const CInterval probabilityInterval = params.GetInterval( "Prob" );

	const float value = static_cast<float>(random.Uniform( valueInterval.Begin, valueInterval.End ));
	const float prob = static_cast<float>(random.Uniform( probabilityInterval.Begin, probabilityInterval.End ));
	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	
	std::vector<float> expected;
	expected.resize( vectorSize );
	CExpectedRandom expectedRandom( seed );
	const unsigned int threshold = ( unsigned int ) ( ( double ) prob * UINT_MAX );
	int index = 0;
	for( int i = 0; i < ( vectorSize + 3 ) / 4; ++i ) {
		CIntArray<4> generated = expectedRandom.Next();
		for( int j = 0; j < 4 && index < vectorSize; ++j ) {
			expected[index++] = ( generated[j] <= threshold ) ? value : 0.f;
		}
	}

	std::vector<float> result;
	result.resize( vectorSize );
	MathEngine().VectorFillBernoulli( CARRAY_FLOAT_WRAPPER( result ), prob, vectorSize, value, seed );
	
	for( int i = 0; i < vectorSize; i++ ) {
		ASSERT_EQ( expected[i], result[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineVectorFillBernoulliTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineVectorFillBernoulliTestInstantiation, CMathEngineVectorFillBernoulliTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (10..1000);"
			"Value = (-100..100);"
			"Prob = (0..1);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineVectorFillBernoulliTest, Random)
{
	RUN_TEST_IMPL( vectorFillBernoulliTestImpl )
}
