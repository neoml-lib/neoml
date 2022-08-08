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

static void blobGlobalMaxPoolingTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval listSizeInterval = params.GetInterval( "ListSize" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval depthInterval = params.GetInterval( "Depth" );
	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval maxCountInterval = params.GetInterval( "MaxCount" );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int listSize = random.UniformInt( listSizeInterval.Begin, std::min( listSizeInterval.End,
		( batchLengthInterval.End * batchWidthInterval.End ) / ( batchLength * batchWidth ) ) );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int depth = random.UniformInt( depthInterval.Begin, depthInterval.End );
	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int maxCount = random.UniformInt( maxCountInterval.Begin, maxCountInterval.End );

	CFloatBlob output( MathEngine(), batchLength, batchWidth, listSize, 1, maxCount, 1, channels );
	CIntBlob indices( MathEngine(), batchLength, batchWidth, listSize, 1, maxCount, 1, channels );
	CFloatBlob input( MathEngine(), batchLength, batchWidth, listSize, height, width, depth, channels);

	std::vector<float> inputBuff;
	inputBuff.resize( input.GetDataSize() );
	std::vector<float> expected;
	expected.insert( expected.begin(), output.GetDataSize(), -FLT_MAX );
	std::vector<float> actual;
	actual.resize( output.GetDataSize() );

	std::vector<int> expectedIndices;
	expectedIndices.resize( output.GetDataSize() );
	std::vector<int> actualIndices;
	actualIndices.resize( output.GetDataSize() );

	for( size_t i = 0; i < inputBuff.size(); ++i ) {
		inputBuff[i] = static_cast<float>( random.Uniform( -10, 10 ) );
	}

	std::vector<int> perm;
	for( int i = 0; i < height * width * depth; ++i ) {
		perm.push_back( i );
	}

	for( int b = 0; b < batchLength * batchWidth * listSize; ++b ) {
		for( int ch = 0; ch < channels; ++ch ) {
			for( size_t i = 0; i < perm.size(); ++i ) {
				size_t j = static_cast<size_t>( random.UniformInt( 0, static_cast<int>( perm.size() - 1 ) ) );
				if( i != j ) {
					std::swap( perm[i], perm[j] );
				}
			}

			for( size_t outW = 0; outW < static_cast<size_t>( maxCount ); ++outW ) {
				const int outputBlobIndex = getFlatIndex( output, 0, 0, b, ch, 0, 0, static_cast<int>( outW ) );
				if( outW < perm.size() ) {
					const int inputBlobIndex = getFlatIndex( input, 0, 0, b, ch, perm[outW], 0, 0 );
					inputBuff[inputBlobIndex] += 50.f * ( maxCount - outW );
					expected[outputBlobIndex] = inputBuff[inputBlobIndex];
					expectedIndices[outputBlobIndex] = perm[outW];
				} else {
					expected[outputBlobIndex] = -FLT_MAX;
					expectedIndices[outputBlobIndex] = -1;
				}
			}
		}
	}

	input.CopyFrom( inputBuff.data() );
	CGlobalMaxPoolingDesc* poolingDesc;

	poolingDesc = MathEngine().InitGlobalMaxPooling( input.GetDesc(), indices.GetDesc(),
		output.GetDesc() );
	MathEngine().BlobGlobalMaxPooling( *poolingDesc, input.GetData(), indices.GetData(),
		output.GetData() );
	output.CopyTo( actual.data() );
	indices.CopyTo( actualIndices.data() );

	delete poolingDesc;

	for( size_t i = 0; i < expected.size(); ++i ) {
		ASSERT_NEAR( expected[i], actual[i], 1e-3 ) << params;
		ASSERT_EQ( expectedIndices[i], actualIndices[i] ) << params;
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineGlobalMaxPoolingTest : public CTestFixtureWithParams {
public:
	void SetUp() override { MathEngine().CleanUp(); }
};

INSTANTIATE_TEST_CASE_P( CMathEngineGlobalMaxPoolingTestInstantiation, CMathEngineGlobalMaxPoolingTest,
	::testing::Values(
		CTestParams(
			"BatchLength = 1;"
			"BatchWidth = 2;"
			"ListSize = 1;"
			"Channels = 5;"
			"Depth = 1;"
			"Height = 1;"
			"Width = 6;"
			"MaxCount = 3;"
			"TestCount = 1;"
		),
		CTestParams(
			"BatchLength = 1;"
			"BatchWidth = 1;"
			"ListSize = 1;"
			"Channels = 1;"
			"Depth = 1;"
			"Height = 1;"
			"Width = 75421;"
			"MaxCount = 5;"
			"TestCount = 1;"
		),
		CTestParams(
			"BatchLength = 1;"
			"BatchWidth = 2;"
			"ListSize = 1;"
			"Channels = 5;"
			"Depth = 2;"
			"Height = 3;"
			"Width = 5;"
			"MaxCount = 30;"
			"TestCount = 1;"
		),
		CTestParams(
			"BatchLength = (1..3);"
			"BatchWidth = (1..3);"
			"ListSize = (1..9);"
			"Channels = (1..3);"
			"Depth = (1..3);"
			"Height = (1..3);"
			"Width = (1..9);"
			"MaxCount = (1..3);"
			"TestCount = 30000"
		),
		CTestParams(
			"BatchLength = (1..5);"
			"BatchWidth = (1..25);"
			"ListSize = (1..25);"
			"Channels = (1..100);"
			"Depth = (1..5);"
			"Height = (1..5);"
			"Width = (1..25);"
			"MaxCount = (1..7);"
			"TestCount = 50"
		),
		CTestParams(
			"BatchLength = (1..3);"
			"BatchWidth = (1..3);"
			"ListSize = (1..3);"
			"Channels = (1..3);"
			"Depth = 1;"
			"Height = 1;"
			"Width = 100000;"
			"MaxCount = 10000;"
			"TestCount = 100;"
		),
		CTestParams(
			"BatchLength = (1..3);"
			"BatchWidth = (1..3);"
			"ListSize = (1..3);"
			"Channels = (1..5);"
			"Depth = 1;"
			"Height = 1;"
			"Width = 1000;"
			"MaxCount = 2000;"
			"TestCount = 10;"
		),
		CTestParams(
			"BatchLength = 1;"
			"BatchWidth = 100;"
			"ListSize = 1;"
			"Channels = 1000;"
			"Depth = 1;"
			"Height = 1;"
			"Width = 100;"
			"MaxCount = 100;"
			"TestCount = 1;"
		)
	)
);

TEST_P(CMathEngineGlobalMaxPoolingTest, Random)
{
	RUN_TEST_IMPL(blobGlobalMaxPoolingTestImpl)
}
