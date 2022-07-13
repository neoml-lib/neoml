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

static void upsampling2DBackwardNaive( const float* input, int batchSize, int inputHeight, int inputWidth, int channels,
	int heightCopyCount, int widthCopyCount, float* result )
{
	const int outputHeight = inputHeight / heightCopyCount;
	const int outputWidth = inputWidth / widthCopyCount;

	for( int b = 0; b < batchSize; ++b ) {
		for( int h = 0; h < inputHeight; ++h ) {
			for( int w = 0; w < inputWidth; ++w ) {
				for( int c = 0; c < channels; ++c ) {
					const int inputIndex = b * inputHeight * inputWidth * channels + h * inputWidth * channels +
						w * channels + c;
					const int destH = h / heightCopyCount;
					const int destW = w / widthCopyCount;
					const int resultIndex = b * outputHeight * outputWidth * channels + destH * outputWidth * channels +
						destW * channels + c;
					result[resultIndex] += input[inputIndex];
				}
			}
		}
	}
}

static void upsampling2DBackwardImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval copyHeightInterval = params.GetInterval( "HeightCopyCount" );
	const CInterval copyWidthInterval = params.GetInterval( "WidthCopyCount" );
	const CInterval channelsInterval = params.GetInterval( "InputChannels" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int heightCopyCount = random.UniformInt( copyHeightInterval.Begin, copyHeightInterval.End );
	const int widthCopyCount = random.UniformInt( copyWidthInterval.Begin, copyWidthInterval.End );
	const int inputHeight = heightCopyCount * random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = widthCopyCount * random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );

	CFloatBlob inputBlob( MathEngine(), batchSize, inputHeight, inputWidth, 1, channels );
	CREATE_FILL_FLOAT_ARRAY( inputBuff, valuesInterval.Begin, valuesInterval.End, inputBlob.GetDataSize(), random )
		inputBlob.CopyFrom( inputBuff.data() );

	CFloatBlob resultBlob( MathEngine(), batchSize, inputHeight / heightCopyCount, inputWidth / widthCopyCount, 1, channels );

	MathEngine().Upsampling2DBackward( inputBlob.GetDesc(), inputBlob.GetData(),
		heightCopyCount, widthCopyCount, resultBlob.GetDesc(), resultBlob.GetData() );
	std::vector<float> actual( resultBlob.GetDataSize() );
	resultBlob.CopyTo( actual.data() );
	std::vector<float> expected;
	expected.insert( expected.begin(), resultBlob.GetDataSize(), 0 );

	upsampling2DBackwardNaive( inputBuff.data(), batchSize, inputHeight, inputWidth, channels, heightCopyCount, widthCopyCount, expected.data() );

	for( size_t i = 0; i < expected.size(); ++i ) {
		ASSERT_NEAR( actual[i], expected[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineUpsampling2DBackwardTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineUpsampling2DBackwardTestInstantiation, CMathEngineUpsampling2DBackwardTest,
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

TEST_P( CMathEngineUpsampling2DBackwardTest, Random )
{
	RUN_TEST_IMPL( upsampling2DBackwardImpl );
}
