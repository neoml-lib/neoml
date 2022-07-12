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

static void batchConvolutionBackward( float* input, const float* filter, const float* freeTerms, const float* output,
	int inputLength, int inputBatch, int inputHeight, int inputWidth, int inputDepth, int inputChannels,
	int paddingHeight, int paddingWidth, int filterCount, int filterHeight, int filterWidth,
	int dilationHeight, int dilationWidth, int strideHeight, int strideWidth )
{
	const int outputHeight = calcConvOutputSize( inputHeight, paddingHeight, filterHeight, dilationHeight, strideHeight );
	const int outputWidth = calcConvOutputSize( inputWidth, paddingWidth, filterWidth, dilationWidth, strideWidth );
	const int inputObjectSize = inputHeight * inputWidth * inputDepth * inputChannels;
	const int outputObjectSize = filterCount * outputHeight * outputWidth;
	const int channels = inputDepth * inputChannels;
	const int filterObjectSize = channels * filterHeight * filterWidth;

	for( int b = 0; b < inputLength * inputBatch; ++b ) {
		for( int h = 0; h < inputHeight; ++h ) {
			for( int w = 0; w < inputWidth; ++w ) {
				for( int c = 0; c < channels; ++c ) {
					input[b * inputObjectSize + h * inputWidth * channels + w * channels + c] = freeTerms[c];
				}
			}
		}
	}

	for( int b = 0; b < inputLength * inputBatch; ++b ) {
		for( int h = 0; h < outputHeight; ++h ) {
			for( int w = 0; w < outputWidth; ++w ) {
				for( int outChannel = 0; outChannel < filterCount; ++outChannel ) {
					const int outputIndex = b * outputObjectSize + h * outputWidth * filterCount + w * filterCount + outChannel;

					for( int filterH = 0; filterH < filterHeight; ++filterH ) {
						for( int filterW = 0; filterW < filterWidth; ++filterW ) {
							for( int inChannel = 0; inChannel < channels; ++inChannel ) {
								const int inputH = h * strideHeight - paddingHeight + filterH * dilationHeight;
								const int inputW = w * strideWidth - paddingWidth + filterW * dilationWidth;

								if( inputH >= 0 && inputW >= 0 && inputH < inputHeight && inputW < inputWidth ) {
									const int inputIndex = b * inputObjectSize + inputH * inputWidth * channels + inputW * channels + inChannel;
									const int filterIndex = outChannel * filterObjectSize + filterH * filterWidth * channels + filterW * channels + inChannel;

									input[inputIndex] += filter[filterIndex] * output[outputIndex];
								}
							}
						}
					}
				}
			}
		}
	}
}

static void blobConvolutionBackwardImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval lengthInterval = params.GetInterval( "InputLength" );
	const CInterval batchInterval = params.GetInterval( "InputBatch" );
	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval inputDepthInterval = params.GetInterval( "InputDepth" );
	const CInterval channelsInterval = params.GetInterval( "InputChannels" );
	const CInterval paddingHeightInterval = params.GetInterval( "PaddingHeight" );
	const CInterval paddingWidthInterval = params.GetInterval( "PaddingWidth" );
	const CInterval filterCountInterval = params.GetInterval( "FilterCount" );
	const CInterval filterHeightInterval = params.GetInterval( "FilterHeight" );
	const CInterval filterWidthInterval = params.GetInterval( "FilterWidth" );
	const CInterval dilationHeightInterval = params.GetInterval( "DilationHeight" );
	const CInterval dilationWidthInterval = params.GetInterval( "DilationWidth" );
	const CInterval strideHeightInterval = params.GetInterval( "StrideHeight" );
	const CInterval strideWidthInterval = params.GetInterval( "StrideWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	bool isZeroFreeTerm = params.GetValue<int>( "IsZeroFreeTerm" ) == 1;

	const int inputLength = random.UniformInt( lengthInterval.Begin, lengthInterval.End );
	const int inputBatch = random.UniformInt( batchInterval.Begin, batchInterval.End );
	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );
	const int inputDepth = random.UniformInt( inputDepthInterval.Begin, inputDepthInterval.End );
	const int inputChannels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int paddingHeight = random.UniformInt( paddingHeightInterval.Begin, paddingHeightInterval.End );
	const int paddingWidth = random.UniformInt( paddingWidthInterval.Begin, paddingWidthInterval.End );
	const int filterCount = random.UniformInt( filterCountInterval.Begin, filterCountInterval.End );
	const int filterHeight = random.UniformInt( filterHeightInterval.Begin, filterHeightInterval.End );
	const int filterWidth = random.UniformInt( filterWidthInterval.Begin, filterWidthInterval.End );
	const int dilationHeight = random.UniformInt( dilationHeightInterval.Begin, dilationHeightInterval.End );
	const int dilationWidth = random.UniformInt( dilationWidthInterval.Begin, dilationWidthInterval.End );
	const int strideHeight = random.UniformInt( strideHeightInterval.Begin, strideHeightInterval.End );
	const int strideWidth = random.UniformInt( strideWidthInterval.Begin, strideWidthInterval.End );
	const int outputHeight = calcConvOutputSize( inputHeight, paddingHeight, filterHeight, dilationHeight, strideHeight );
	const int outputWidth = calcConvOutputSize( inputWidth, paddingWidth, filterWidth, dilationWidth, strideWidth );

	CREATE_FILL_FLOAT_ARRAY( outputData, valuesInterval.Begin, valuesInterval.End,
		inputLength * inputBatch * outputHeight * outputWidth * filterCount, random )
	CFloatBlob outputBlob( MathEngine(), inputLength, inputBatch, 1, outputHeight, outputWidth, 1, filterCount );
	outputBlob.CopyFrom( outputData.data() );

	CREATE_FILL_FLOAT_ARRAY( filterData, valuesInterval.Begin, valuesInterval.End,
		filterCount * filterHeight * filterWidth * inputDepth * inputChannels, random )
	CFloatBlob filterBlob( MathEngine(), filterCount, filterHeight, filterWidth, inputDepth, inputChannels );
	filterBlob.CopyFrom( filterData.data() );

	CREATE_FILL_FLOAT_ARRAY( freeTermData, valuesInterval.Begin, valuesInterval.End, inputDepth * inputChannels, random )
	CFloatBlob freeTermBlob( MathEngine(), 1, 1, 1, inputDepth * inputChannels );
	freeTermBlob.CopyFrom( freeTermData.data() );
	if( isZeroFreeTerm ) {
		freeTermData.clear();
		freeTermData.insert( freeTermData.begin(), inputDepth * inputChannels, 0 );
	}

	CFloatBlob inputBlob( MathEngine(), inputLength, inputBatch, 1, inputHeight, inputWidth, inputDepth, inputChannels );

	CConvolutionDesc* convDesc = MathEngine().InitBlobConvolution( inputBlob.GetDesc(),
		paddingHeight, paddingWidth, strideHeight, strideWidth,
		dilationHeight, dilationWidth, filterBlob.GetDesc(), outputBlob.GetDesc() );

	CConstFloatHandle freeTermDataPtr = freeTermBlob.GetData();

	MathEngine().BlobConvolutionBackward( *convDesc, outputBlob.GetData(), filterBlob.GetData(),
		isZeroFreeTerm ? 0 : &freeTermDataPtr, inputBlob.GetData() );
	delete convDesc;

	const int inputSize = inputLength * inputBatch * inputHeight * inputWidth * inputDepth * inputChannels;
	std::vector<float> expectedData( inputSize );
	std::vector<float> actualData( inputSize );
	inputBlob.CopyTo( actualData.data() );

	batchConvolutionBackward( expectedData.data(), filterData.data(), freeTermData.data(), outputData.data(),
		inputLength, inputBatch, inputHeight, inputWidth, inputDepth, inputChannels,
		paddingHeight, paddingWidth, filterCount, filterHeight, filterWidth,
		dilationHeight, dilationWidth, strideHeight, strideWidth );

	for( int i = 0; i < inputSize; ++i ) {
		ASSERT_NEAR( expectedData[i], actualData[i], 1e-3f );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBlobConvolutionBackwardTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineBlobConvolutionBackwardTestInstantiation, CMathEngineBlobConvolutionBackwardTest,
	::testing::Values(
		CTestParams(
			"InputLength = (1..3);"
			"InputBatch = (1..3);"
			"InputHeight = (10..15);"
			"InputWidth = (10..15);"
			"InputDepth = (1..3);"
			"InputChannels = (1..3);"
			"FilterCount = (1..3);"
			"FilterHeight = (2..5);"
			"FilterWidth = (2..5);"
			"PaddingHeight = (0..1);"
			"PaddingWidth = (0..1);"
			"DilationHeight = (1..2);"
			"DilationWidth = (1..2);"
			"StrideHeight = (1..2);"
			"StrideWidth = (1..2);"
			"IsZeroFreeTerm = (0..1);"
			"Values = (-10..10);"
			"TestCount = 100;"
		),
		CTestParams(
			"InputLength = 3;"
			"InputBatch = 4;"
			"InputHeight = 5;"
			"InputWidth = 6;"
			"InputDepth = 4;"
			"InputChannels = 3;"
			"FilterCount = 4;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"IsZeroFreeTerm = 0;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 3;"
			"InputBatch = 4;"
			"InputHeight = 5;"
			"InputWidth = 6;"
			"InputDepth = 1;"
			"InputChannels = 4;"
			"FilterCount = 2;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"IsZeroFreeTerm = 0;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 1;"
			"InputBatch = 1;"
			"InputHeight = 4;"
			"InputWidth = 5;"
			"InputDepth = 1;"
			"InputChannels = 3;"
			"FilterCount = 4;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 3;"
			"IsZeroFreeTerm = 0;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 1;"
			"InputBatch = 1;"
			"InputHeight = 5;"
			"InputWidth = 5;"
			"InputDepth = 1;"
			"InputChannels = 3;"
			"FilterCount = 4;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"IsZeroFreeTerm = 1;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 1;"
			"InputBatch = 1;"
			"InputHeight = 10;"
			"InputWidth = 10;"
			"InputDepth = 1;"
			"InputChannels = 2;"
			"FilterCount = 4;"
			"FilterHeight = 3;"
			"FilterWidth = 7;"
			"PaddingHeight = 2;"
			"PaddingWidth = 3;"
			"DilationHeight = 2;"
			"DilationWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 1;"
			"IsZeroFreeTerm = 0;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 1;"
			"InputBatch = 1;"
			"InputHeight = 5;"
			"InputWidth = 5;"
			"InputDepth = 1;"
			"InputChannels = 3;"
			"FilterCount = 4;"
			"FilterHeight = 3;"
			"FilterWidth = 1;"
			"PaddingHeight = 2;"
			"PaddingWidth = 10;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"IsZeroFreeTerm = 0;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 1;"
			"InputBatch = 1;"
			"InputHeight = 5;"
			"InputWidth = 5;"
			"InputDepth = 1;"
			"InputChannels = 3;"
			"FilterCount = 4;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"PaddingHeight = 2;"
			"PaddingWidth = 10;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"IsZeroFreeTerm = 0;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 1;"
			"InputBatch = 1;"
			"InputHeight = 4;"
			"InputWidth = 5;"
			"InputDepth = 1;"
			"InputChannels = 3;"
			"FilterCount = 4;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 3;"
			"IsZeroFreeTerm = 1;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 1;"
			"InputBatch = 1;"
			"InputHeight = 5;"
			"InputWidth = 5;"
			"InputDepth = 1;"
			"InputChannels = 3;"
			"FilterCount = 4;"
			"FilterHeight = 2;"
			"FilterWidth = 3;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 4;"
			"IsZeroFreeTerm = 0;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 1;"
			"InputBatch = 1;"
			"InputHeight = 4096;"
			"InputWidth = 1;"
			"InputDepth = 1;"
			"InputChannels = 3;"
			"FilterCount = 3;"
			"FilterHeight = 2;"
			"FilterWidth = 3;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 1;"
			"IsZeroFreeTerm = 1;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 1;"
			"InputBatch = 1;"
			"InputHeight = 4096;"
			"InputWidth = 1;"
			"InputDepth = 1;"
			"InputChannels = 3;"
			"FilterCount = 3;"
			"FilterHeight = 3;"
			"FilterWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 2;"
			"IsZeroFreeTerm = 0;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 2;"
			"InputBatch = 3;"
			"InputHeight = 7;"
			"InputWidth = 8;"
			"InputDepth = 3;"
			"InputChannels = 2;"
			"FilterCount = 2;"
			"FilterHeight = 3;"
			"FilterWidth = 4;"
			"PaddingHeight = 2;"
			"PaddingWidth = 3;"
			"DilationHeight = 3;"
			"DilationWidth = 2;"
			"StrideHeight = 2;"
			"StrideWidth = 1;"
			"IsZeroFreeTerm = 1;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 2;"
			"InputBatch = 3;"
			"InputHeight = 8;"
			"InputWidth = 7;"
			"InputDepth = 3;"
			"InputChannels = 2;"
			"FilterCount = 2;"
			"FilterHeight = 4;"
			"FilterWidth = 3;"
			"PaddingHeight = 3;"
			"PaddingWidth = 2;"
			"DilationHeight = 2;"
			"DilationWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 2;"
			"IsZeroFreeTerm = 0;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 2;"
			"InputBatch = 3;"
			"InputHeight = 32;"
			"InputWidth = 32;"
			"InputDepth = 4;"
			"InputChannels = 2;"
			"FilterCount = 2;"
			"FilterHeight = 1;"
			"FilterWidth = 4;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"IsZeroFreeTerm = 1;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 3;"
			"InputBatch = 2;"
			"InputHeight = 32;"
			"InputWidth = 32;"
			"InputDepth = 2;"
			"InputChannels = 4;"
			"FilterCount = 2;"
			"FilterHeight = 4;"
			"FilterWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"IsZeroFreeTerm = 0;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 1;"
			"InputBatch = 1;"
			"InputHeight = 6;"
			"InputWidth = 5;"
			"InputDepth = 1;"
			"InputChannels = 3;"
			"FilterCount = 4;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"IsZeroFreeTerm = 1;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputLength = 1;"
			"InputBatch = 1;"
			"InputHeight = 256;"
			"InputWidth = 16;"
			"InputDepth = 1;"
			"InputChannels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"PaddingHeight = 16;"
			"PaddingWidth = 16;"
			"DilationHeight = 16;"
			"DilationWidth = 16;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"IsZeroFreeTerm = 0;"
			"Values = (-1..1);"
			"TestCount = 1;"
		)
	)
);

TEST_P( CMathEngineBlobConvolutionBackwardTest, Random )
{
	RUN_TEST_IMPL( blobConvolutionBackwardImpl );
}
