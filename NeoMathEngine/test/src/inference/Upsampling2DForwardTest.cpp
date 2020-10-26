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

static void upsampling2DNaive( const float* input, int inputHeight, int inputWidth, int inputChannels,
	int batchSize, int heightCopyCount, int widthCopyCount, float* output )
{
	for( int b = 0; b < batchSize; ++b ) {
		int inOffset = b * inputHeight * inputWidth * inputChannels;
		int outOffset = inOffset * heightCopyCount * widthCopyCount;
		for( int y = 0; y < inputHeight; ++y ) {
			int inOffsetY = y * inputWidth * inputChannels;
			for( int yCopy = 0; yCopy < heightCopyCount; ++yCopy ) {
				int outOffsetY = ( y * heightCopyCount + yCopy ) * inputWidth * inputChannels * widthCopyCount;
				for( int x = 0; x < inputWidth; ++x ) {
					int inOffsetX = x * inputChannels;
					for( int xCopy = 0; xCopy < widthCopyCount; ++xCopy ) {
						int outOffsetX = ( x * widthCopyCount + xCopy ) * inputChannels;
						for( int c = 0; c < inputChannels; ++c ) {
							output[outOffset + outOffsetY + outOffsetX + c] = input[inOffset + inOffsetY + inOffsetX + c];
						}
					}
				}
			}
		}
	}
}

static void upsampling2DImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval inputChannelsInterval = params.GetInterval( "InputChannels" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	const CInterval heightCopyCountInterval = params.GetInterval( "HeightCopyCount" );
	const CInterval widthCopyCountInterval = params.GetInterval( "WidthCopyCount" );

	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );
	const int inputChannels = random.UniformInt( inputChannelsInterval.Begin, inputChannelsInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int heightCopyCount = random.UniformInt( heightCopyCountInterval.Begin, heightCopyCountInterval.End );
	const int widthCopyCount = random.UniformInt( widthCopyCountInterval.Begin, widthCopyCountInterval.End );

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, batchSize * inputHeight * inputWidth * inputChannels, random )
	CFloatBlob inputBlob( MathEngine(), batchSize, inputHeight, inputWidth, 1, inputChannels );
	inputBlob.CopyFrom( inputData.data() );

	const int outHeight = inputHeight * heightCopyCount;
	const int outWidth = inputWidth * widthCopyCount;
	const int outChannels = inputChannels;
	CFloatBlob outBlob( MathEngine(), batchSize, outHeight, outWidth, 1, outChannels );

	MathEngine().Upsampling2DForward( inputBlob.GetDesc(), inputBlob.GetData(),
		heightCopyCount, widthCopyCount, outBlob.GetDesc(), outBlob.GetData() );

	std::vector<float> resultData;
	resultData.resize( batchSize * outHeight * outWidth * outChannels );
	outBlob.CopyTo( resultData.data() );

	std::vector<float> expectedData;
	expectedData.resize( batchSize * outHeight * outWidth * outChannels );
	upsampling2DNaive( inputData.data(), inputHeight, inputWidth, inputChannels, batchSize, heightCopyCount, widthCopyCount, expectedData.data() );

	for( size_t i = 0; i < resultData.size(); i++ ) {
		ASSERT_NEAR( expectedData[i], resultData[i], 1e-3f );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineUpsampling2DForwardTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P(CMathEngineUpsampling2DForwardTestInstantiation, CMathEngineUpsampling2DForwardTest,
	::testing::Values(
		CTestParams(
			"InputHeight = (5..15);"
			"InputWidth = (5..15);"
			"InputChannels = (1..3);"
			"BatchSize = (1..5);"
			"Values = (-10..10);"
			"HeightCopyCount = (1..5);"
			"WidthCopyCount = (1..5);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineUpsampling2DForwardTest, Random )
{
	RUN_TEST_IMPL( upsampling2DImpl );
}
