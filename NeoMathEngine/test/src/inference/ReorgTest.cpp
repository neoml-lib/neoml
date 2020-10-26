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

static int LegacyRepackIndex( int fromIndex, int channels, int height, int width )
{
	int x = fromIndex % width;
	fromIndex /= width;
	int y = fromIndex % height;
	fromIndex /= height;
	int c = fromIndex % channels;
	int b = fromIndex / channels;
	return c + channels * ( x + width * ( y + height * b ) );
}

template<class T>
static void ReorgFuncNaive( const T* source, int stride, bool isForward, int batchSize, int channelsCount,
	int height, int width, T* result )
{
	const int outputChannels = channelsCount / ( stride * stride );
	for( int batch = 0; batch < batchSize; ++batch ) {
		for( int channel = 0; channel < channelsCount; ++channel ) {
			for( int h = 0; h < height; ++h ) {
				for( int w = 0; w < width; ++w ) {
					int inputIndex = w + width * ( h + height * ( channel + channelsCount * batch ) );
					inputIndex = LegacyRepackIndex( inputIndex, channelsCount * stride * stride, height / stride, width / stride );
					const int outputChannelId = channel % outputChannels;
					const int offset = channel / outputChannels;
					const int outputW = w*stride + offset % stride;
					const int outputH = h*stride + offset / stride;
					int outputIndex = outputW + width * stride * ( outputH + height * stride *
						( outputChannelId + outputChannels * batch ) );
					outputIndex = LegacyRepackIndex( outputIndex, channelsCount, height, width );
					if( isForward ) {
						result[inputIndex] = source[outputIndex];
					}
					else {
						result[outputIndex] = source[inputIndex];
					}
				}
			}
		}
	}
}

template<class T>
static void reorgLayerTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval heightMultiplierInterval = params.GetInterval( "HeightMultiplier" );
	const CInterval widthMultiplierInterval = params.GetInterval( "WidthMultiplier" );
	const CInterval channelsMultiplierInterval = params.GetInterval( "ChannelsMultiplier" );
	const CInterval strideInterval = params.GetInterval( "Stride" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int stride = random.UniformInt( strideInterval.Begin, strideInterval.End );
	const int height = random.UniformInt( heightMultiplierInterval.Begin, heightMultiplierInterval.End ) * stride;
	const int width = random.UniformInt( widthMultiplierInterval.Begin, widthMultiplierInterval.End ) * stride;
	const int channels = random.UniformInt( channelsMultiplierInterval.Begin, channelsMultiplierInterval.End ) * stride * stride;

	const int blobSize = batchSize * height * width * channels;
	std::vector<T> inputData;
	inputData.resize( blobSize );
	for( int i = 0; i < blobSize; ++i ) {
		inputData[i] = static_cast<T>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) );

	}
	CBlob<T> inputBlob( MathEngine(), batchSize, height, width, channels );
	inputBlob.CopyFrom( inputData.data() );

	std::vector<T> expectedData, getData;
	expectedData.resize( blobSize );
	getData.resize( blobSize );

	CBlob<T> outputBlob( MathEngine(), batchSize, height, width, channels );

	MathEngine().Reorg( inputBlob.GetDesc(), inputBlob.GetData(), stride, true, outputBlob.GetDesc(), outputBlob.GetData() );
	outputBlob.CopyTo( getData.data() );
	ReorgFuncNaive<T>( inputData.data(), stride, true, batchSize, channels, height, width, expectedData.data() );

	for( size_t i = 0; i < expectedData.size(); ++i ) {
		ASSERT_NEAR( expectedData[i], getData[i], 1e-3 );
	}

	MathEngine().Reorg( inputBlob.GetDesc(), inputBlob.GetData(), stride, false, outputBlob.GetDesc(), outputBlob.GetData() );
	outputBlob.CopyTo( getData.data() );
	ReorgFuncNaive<T>( inputData.data(), stride, false, batchSize, channels, height, width, expectedData.data() );
	for( size_t i = 0; i < expectedData.size(); ++i ) {
		ASSERT_NEAR( expectedData[i], getData[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineReorgTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineReorgTestInstantiation, CMathEngineReorgTest,
	::testing::Values(
		CTestParams(
			"Stride = (1..5);"
			"BatchSize = (1..5);"
			"WidthMultiplier = (1..10);"
			"HeightMultiplier = (1..10);"
			"ChannelsMultiplier = (1..4);"
			"Values = (-10..10);"
			"TestCount = 10;"
		)
	)
);

TEST_P( CMathEngineReorgTest, Random )
{
	RUN_TEST_IMPL( reorgLayerTestImpl<int> );
	RUN_TEST_IMPL( reorgLayerTestImpl<float> );
}
