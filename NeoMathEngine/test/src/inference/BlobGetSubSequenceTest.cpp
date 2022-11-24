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

static void blobSubSequenceTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval listSizeInterval = params.GetInterval( "ListSize" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval subseqStartInterval = params.GetInterval( "SubseqStart" );
	const CInterval subseqLengthInterval = params.GetInterval( "SubseqLength" );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int listSize = random.UniformInt( listSizeInterval.Begin, listSizeInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int subseqStart = random.UniformInt( subseqStartInterval.Begin, std::min( batchLength - 1, subseqStartInterval.End ) );
	const int subseqLength = random.UniformInt( subseqLengthInterval.Begin, std::min( batchLength - subseqStart, subseqLengthInterval.End ) );
	const bool isRev = random.Next() % 2 == 1;

	const int seqElemSize = batchWidth * listSize * channels;

	CFloatBlob input( MathEngine(), batchLength, batchWidth, listSize, 1, 1, 1, channels );
	CFloatBlob output( MathEngine(), subseqLength, batchWidth, listSize, 1, 1, 1, channels );

	std::vector<float> inputBuff;
	inputBuff.resize( input.GetDataSize() );
	std::vector<float> expected;
	expected.resize( output.GetDataSize() );
	std::vector<float> actual;
	actual.resize( output.GetDataSize() );
	for( int s = 0; s < batchLength; ++s ) {
		const int outputSeq = isRev ? subseqLength - 1 - ( s - subseqStart ) : s - subseqStart;
		for( int rest = 0; rest < seqElemSize; ++rest ) {
			const int inputIndex = s * seqElemSize + rest;
			inputBuff[inputIndex] = static_cast<float>( random.Uniform( -10, 10 ) );
			if( outputSeq >= 0 && outputSeq < subseqLength ) {
				const int outputIndex = outputSeq * seqElemSize + rest;
				expected[outputIndex] = inputBuff[inputIndex];
			}
		}
	}

	input.CopyFrom( inputBuff.data() );
	MathEngine().BlobGetSubSequence( input.GetDesc(), input.GetData(), CIntHandle(),
		output.GetDesc(), output.GetData(), isRev ? subseqStart + subseqLength - 1 : subseqStart, isRev );
	output.CopyTo( actual.data() );

	for( size_t i = 0; i < expected.size(); ++i ) {
		ASSERT_NEAR( expected[i], actual[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineGetSubsequenceTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineGetSubsequenceTestInstantiation, CMathEngineGetSubsequenceTest,
	::testing::Values(
		CTestParams(
			"BatchLength = 5;"
			"BatchWidth = 2;"
			"ListSize = 1;"
			"Channels = 15000;"
			"SubseqStart = 2;"
			"SubseqLength = 1;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = (3..7);"
			"BatchWidth = (1..8);"
			"ListSize = (1..5);"
			"Channels = (1..3);"
			"SubseqStart = (0..6);"
			"SubseqLength = (1..7);"
			"TestCount = 10000"
		),
		CTestParams(
			"BatchLength = (10..20);"
			"BatchWidth = (7..13);"
			"ListSize = (25..35);"
			"Channels = (25..35);"
			"SubseqStart = (0..19);"
			"SubseqLength = (1..20);"
			"TestCount = 100"
		)
	)
);

TEST_P(CMathEngineGetSubsequenceTest, Random)
{
	RUN_TEST_IMPL(blobSubSequenceTestImpl)
}
