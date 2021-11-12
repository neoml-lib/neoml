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

static inline int convOutputSize( int imageSize, int filterSize, int stride, int padding, int dilation )
{
	const int fullImageSize = imageSize + 2 * padding;
	const int fullFilterSize = 1 + ( filterSize - 1 ) * dilation;
	return 1 + ( fullImageSize - fullFilterSize ) / stride;
}

static inline void foldNaive( const std::vector<float>& matrices, std::vector<float>& images,
	int batchSize, int imageHeight, int imageWidth, int channels, int filterHeight, int filterWidth,
	int strideHeight, int strideWidth, int paddingHeight, int paddingWidth, int dilationHeight, int dilationWidth )
{
	const int outputImageHeight = convOutputSize( imageHeight, filterHeight, strideHeight, paddingHeight, dilationHeight );
	const int outputImageWidth = convOutputSize( imageWidth, filterWidth, strideWidth, paddingWidth, dilationWidth );
	const int matrixHeight = outputImageHeight * outputImageWidth;
	const int matrixWidth = filterHeight * filterWidth * channels;

	ASSERT_EQ( matrices.size(), batchSize * matrixHeight * matrixWidth );

	images.resize( 0 );
	images.resize( batchSize * imageHeight * imageWidth * channels );

	int outIndex = 0;
	for( int b = 0; b < batchSize; ++b ) {
		for( int outY = 0; outY < outputImageHeight; ++outY ) {
			for( int outX = 0; outX < outputImageWidth; ++outX ) {
				for( int filterY = 0; filterY < filterHeight; ++filterY ) {
					const int inY = outY * strideHeight - paddingHeight + filterY * dilationHeight;
					for( int filterX = 0; filterX < filterWidth; ++filterX ) {
						const int inX = outX * strideWidth - paddingWidth + filterX * dilationWidth;
						if( inX >= 0 && inY >= 0 && inX < imageWidth && inY < imageHeight ) {
							int inIndex = ( ( b * imageHeight + inY ) * imageWidth + inX ) * channels;
							for( int c = 0; c < channels; ++c ) {
								images[inIndex++] += matrices[outIndex++];
							}
						} else {
							outIndex += channels;
						}
					}
				}
			}
		}
	}
}

static void foldTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchInterval = params.GetInterval( "InputBatch" );
	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval channelsInterval = params.GetInterval( "InputChannels" );
	const CInterval paddingHeightInterval = params.GetInterval( "PaddingHeight" );
	const CInterval paddingWidthInterval = params.GetInterval( "PaddingWidth" );
	const CInterval filterHeightInterval = params.GetInterval( "FilterHeight" );
	const CInterval filterWidthInterval = params.GetInterval( "FilterWidth" );
	const CInterval dilationHeightInterval = params.GetInterval( "DilationHeight" );
	const CInterval dilationWidthInterval = params.GetInterval( "DilationWidth" );
	const CInterval strideHeightInterval = params.GetInterval( "StrideHeight" );
	const CInterval strideWidthInterval = params.GetInterval( "StrideWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int inputBatch = random.UniformInt( batchInterval.Begin, batchInterval.End );
	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );
	const int inputChannels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int paddingHeight = random.UniformInt( paddingHeightInterval.Begin, paddingHeightInterval.End );
	const int paddingWidth = random.UniformInt( paddingWidthInterval.Begin, paddingWidthInterval.End );
	const int filterHeight = random.UniformInt( filterHeightInterval.Begin, filterHeightInterval.End );
	const int filterWidth = random.UniformInt( filterWidthInterval.Begin, filterWidthInterval.End );
	const int dilationHeight = random.UniformInt( dilationHeightInterval.Begin, dilationHeightInterval.End );
	const int dilationWidth = random.UniformInt( dilationWidthInterval.Begin, dilationWidthInterval.End );
	const int strideHeight = random.UniformInt( strideHeightInterval.Begin, strideHeightInterval.End );
	const int strideWidth = random.UniformInt( strideWidthInterval.Begin, strideWidthInterval.End );

	const int outputHeight = convOutputSize( inputHeight, filterHeight, strideHeight, paddingHeight, dilationHeight );
	const int outputWidth = convOutputSize( inputWidth, filterWidth, strideWidth, paddingWidth, dilationWidth );
	const int matrixHeight = outputHeight * outputWidth;
	const int matrixWidth = filterHeight * filterWidth * inputChannels;

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, inputBatch * matrixHeight * matrixWidth, random )
	CFloatBlob inputBlob( MathEngine(), 1, inputBatch, 1, matrixHeight, 1, 1, matrixWidth );
	inputBlob.CopyFrom( inputData.data() );

	CFloatBlob outputBlob( MathEngine(), 1, inputBatch, 1, inputHeight, inputWidth, 1, inputChannels );

	MathEngine().Fold( inputBatch, inputBlob.GetData(), filterHeight, filterWidth, strideHeight, strideWidth,
		paddingHeight, paddingWidth, dilationHeight, dilationWidth, outputBlob.GetData(),
		inputHeight, inputWidth, inputChannels );

	std::vector<float> actual( outputBlob.GetDataSize() );
	outputBlob.CopyTo( actual.data() );

	std::vector<float> expected;
	foldNaive( inputData, expected, inputBatch, inputHeight, inputWidth, inputChannels,
		filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth,
		dilationHeight, dilationWidth );

	ASSERT_EQ( expected.size(), actual.size() );
	for( size_t i = 0; i < actual.size(); ++i ) {
		ASSERT_TRUE( FloatEq( expected[i], actual[i], 1e-5f ) );
	}
}

class CMathEngineFoldTest : public CTestFixtureWithParams {
};

TEST_P( CMathEngineFoldTest, Random )
{
	RUN_TEST_IMPL( foldTestImpl );
}

INSTANTIATE_TEST_CASE_P( CMathEngineFoldTestInstantiation, CMathEngineFoldTest,
	::testing::Values(
		CTestParams(
			"InputBatch = (1..3);"
			"InputHeight = (10..15);"
			"InputWidth = (10..15);"
			"InputChannels = (1..3);"
			"FilterHeight = (2..5);"
			"FilterWidth = (2..5);"
			"PaddingHeight = (0..1);"
			"PaddingWidth = (0..1);"
			"DilationHeight = (1..2);"
			"DilationWidth = (1..2);"
			"StrideHeight = (1..2);"
			"StrideWidth = (1..2);"
			"Values = (-10..10);"
			"TestCount = 100;"
		),
		CTestParams(
			"InputBatch = 4;"
			"InputHeight = 5;"
			"InputWidth = 6;"
			"InputChannels = 3;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputBatch = 4;"
			"InputHeight = 5;"
			"InputWidth = 6;"
			"InputChannels = 4;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputBatch = 1;"
			"InputHeight = 4;"
			"InputWidth = 5;"
			"InputChannels = 3;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 3;"
			"Values = (-10..10);"
			"TestCount = 1;"
		)
	)
);
