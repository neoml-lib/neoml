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

static const float logZero = -FLT_MAX / 4;
static const float logOneNeg = -FLT_MIN * 2;

static void ctcFillPaddingNaive( int maxSeqLen, int batchSize, int classCount, int blankLabel,
	float* data, const int* seqLens )
{
	for( int b = 0; b < batchSize; ++b ) {
		const int seqLen = seqLens[b];
		for( int seq = seqLen; seq < maxSeqLen; ++seq ) {
			for( int classIndex = 0; classIndex < classCount; ++classIndex ) {
				const float value = blankLabel == -1 ? 0.f
					: ( blankLabel == classIndex ? logOneNeg : logZero );
				data[( seq * batchSize + b ) * classCount + classIndex] = value;
			}
		}
	}
}

static void ctcFillPaddingTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval maxSeqLenInterval = params.GetInterval( "SeqLen" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval classCountInterval = params.GetInterval( "ClassCount" );

	const int maxSeqLen = random.UniformInt( maxSeqLenInterval.Begin, maxSeqLenInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int classCount = random.UniformInt( classCountInterval.Begin, classCountInterval.End );
	const int blankLabel = random.UniformInt( -1, classCount - 1 );

	CREATE_FILL_FLOAT_ARRAY( actual, 0.f, 1.f, maxSeqLen * batchSize * classCount, random );
	CREATE_FILL_INT_ARRAY( seqLens, 1, maxSeqLen, batchSize, random );

	std::vector<float> expected( actual );
	ctcFillPaddingNaive( maxSeqLen, batchSize, classCount, blankLabel, expected.data(), seqLens.data() );

	MathEngine().CtcFillPadding( maxSeqLen, batchSize, classCount, blankLabel,
		CARRAY_FLOAT_WRAPPER( actual ), CARRAY_INT_WRAPPER( seqLens ) );
	
	for( int i = 0; i < maxSeqLen * batchSize * classCount; ++i ) {
		ASSERT_NEAR( expected[i], actual[i], 1e-5f );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineCtcFillPaddingTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineCtcFillPaddingTestInstantiation, CMathEngineCtcFillPaddingTest,
	::testing::Values(
		CTestParams(
			"SeqLen = (10..100);"
			"BatchSize = (10..100);"
			"ClassCount = (2..10);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CMathEngineCtcFillPaddingTest, Random )
{
	RUN_TEST_IMPL( ctcFillPaddingTestImpl );;
}
