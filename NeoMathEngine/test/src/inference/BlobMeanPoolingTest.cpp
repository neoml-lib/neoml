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

static void meanPoolingNaive( const float *sourceData, int filterHeight, int filterWidth, int strideHeight, int strideWidth,
	int batchSize, int height, int width, int depth, int channels, float *resultData ) 
{
	int resultHeight = ( height - filterHeight ) / strideHeight + 1;
	int resultWidth = ( width - filterWidth ) / strideWidth + 1;
	for( int b = 0; b < batchSize; ++b ) {
		for( int c = 0; c < channels * depth; ++c ) {
			for( int y = 0; y < resultHeight; ++y ) {
				for( int x = 0; x < resultWidth; ++x ) {
					int inputHStart = y * strideHeight;
					int inputHEnd = std::min( inputHStart + filterHeight, height );
					int inputWStart = x * strideWidth;
					int inputWEnd = std::min( inputWStart + filterWidth, width );

					float res = 0.f;
					for( int j = inputHStart; j < inputHEnd; ++j ) {
						for( int i = inputWStart; i < inputWEnd; ++i ) {
							int index = b * height * width * depth * channels + j * width * depth * channels + i * depth * channels + c;
							res += sourceData[index];
						}
					}
					int resultIndex = b * resultHeight * resultWidth * depth * channels + y * resultWidth * depth * channels + x * depth * channels + c;
					resultData[resultIndex] = res / filterHeight / filterWidth;
				}
			}
		}		
	}
}

static void meanPoolingTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval depthInterval = params.GetInterval( "Depth" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );

	const CInterval filterHeightInterval = params.GetInterval( "FilterHeight" );
	const CInterval filterWidthInterval = params.GetInterval( "FilterWidth" );
	const CInterval strideHeightInterval = params.GetInterval( "StrideHeight" );
	const CInterval strideWidthInterval = params.GetInterval( "StrideWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int depth = random.UniformInt( depthInterval.Begin, depthInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int filterHeight = random.UniformInt( filterHeightInterval.Begin, filterHeightInterval.End );
	const int filterWidth = random.UniformInt( filterWidthInterval.Begin, filterWidthInterval.End );
	const int strideHeight = random.UniformInt( strideHeightInterval.Begin, strideHeightInterval.End );
	const int strideWidth = random.UniformInt( strideWidthInterval.Begin, strideWidthInterval.End );

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, batchSize * height * width * channels * depth, random )

	int resultHeight = ( height - filterHeight ) / strideHeight + 1;
	int resultWidth = ( width - filterWidth ) / strideWidth + 1;
	
	std::vector<float> expected;
	expected.resize( batchSize * depth * channels * resultHeight * resultWidth );
	
	meanPoolingNaive( inputData.data(), filterHeight, filterWidth, strideHeight, strideWidth, batchSize, height, width, depth, channels, expected.data() );

	CFloatBlob inputBlob( MathEngine(), batchSize, height, width, depth, channels );
	inputBlob.CopyFrom( inputData.data() );
	CFloatBlob resultBlob( MathEngine(), batchSize, resultHeight, resultWidth, depth, channels );
	
	CMeanPoolingDesc *desc = MathEngine().InitMeanPooling(inputBlob.GetDesc(), filterHeight, filterWidth, strideHeight, strideWidth, resultBlob.GetDesc());
	MathEngine().BlobMeanPooling( *desc, inputBlob.GetData(), resultBlob.GetData() );

	std::vector<float> resultData;
	resultData.resize( resultBlob.GetDataSize() );
	resultBlob.CopyTo( resultData.data() );

	for(size_t i = 0; i < expected.size(); i++) {
		ASSERT_NEAR(expected[i], resultData[i], 1e-3);
	}
	delete desc;
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineMeanPoolingTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineMeanPoolingTestInstantiation, CMathEngineMeanPoolingTest,
	::testing::Values(
		CTestParams(
			"Height = (10..100);"
			"Width = (10..100);"
			"Channels = (1..5);"
			"Depth = (1..5);"
			"BatchSize = (1..5);"
			"FilterHeight = (1..5);"
			"FilterWidth = (1..5);"
			"StrideHeight = (1..3);"
			"StrideWidth = (1..3);"
			"Values = (-50..50);"
			"TestCount = 100;"
		),
		CTestParams(
			"Height = (3..4);"
			"Width = (3..4);"
			"Channels = (1..2);"
			"Depth = (1..1);"
			"BatchSize = (1..1);"
			"FilterHeight = (1..2);"
			"FilterWidth = (1..2);"
			"StrideHeight = (1..2);"
			"StrideWidth = (1..2);"
			"Values = (-1..10000);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineMeanPoolingTest, Random)
{
	RUN_TEST_IMPL(meanPoolingTestImpl)
}
