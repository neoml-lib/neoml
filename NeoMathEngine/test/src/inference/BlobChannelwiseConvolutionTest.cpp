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

static void blobChannelwiseConvolutionNaive( int inputCount, int inputChannels, int inputHeight, int inputWidth,
	int filterHeight, int filterWidth,
	int paddingHeight, int paddingWidth,
	int strideHeight, int strideWidth,
	const float *input, const float *filter, const float *freeTerm, float *result )
{
	const int outHeight = 1 + ( inputHeight + 2 * paddingHeight - filterHeight ) / strideHeight;
	const int outWidth = 1 + ( inputWidth + 2 * paddingWidth - filterWidth ) / strideWidth;
	const int outChannels = inputChannels;

	for( int b = 0; b < inputCount; b++ ) {
		const int inputOffset = b * inputChannels * inputHeight * inputWidth;
		const int outOffset = b * outChannels * outHeight * outWidth;

		for( int y = 0; y < outHeight; y++ ) {
			for( int x = 0; x < outWidth; x++ ) {
				for( int c = 0; c < outChannels; c++ ) {
					const int inputHStart = y * strideHeight - paddingHeight;
					const int inputHEnd = inputHStart + filterHeight;
					const int inputWStart = x * strideWidth - paddingWidth;
					const int inputWEnd = inputWStart + filterWidth;

					float resultVal = freeTerm[c];
					int filterY = 0;
					for( int j = inputHStart; j < inputHEnd; j++ ) {
						if( j < 0 || j >= inputHeight ) {
							filterY++;
							continue;
						}
						int filterX = 0;

						for( int i = inputWStart; i < inputWEnd; i++ ) {
							if( i < 0 || i >= inputWidth ) {
								filterX++;
								continue;
							}
							
							const float srcVal = input[inputOffset + j * inputWidth * inputChannels + i * inputChannels + c];
							const float fltVal = filter[filterY * filterWidth * inputChannels + filterX * inputChannels + c];
							resultVal = fma( srcVal, fltVal, resultVal );
							filterX++;
						}
						filterY++;
					}
					result[outOffset + y * outWidth * outChannels + x * outChannels + c] = resultVal;
				}
			}
		}
	}
}

static void blobChannelwiseConvolutionTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval listSizeInterval = params.GetInterval( "ListSize" );
	const CInterval filterHeightInterval = params.GetInterval( "FilterHeight" );
	const CInterval filterWidthInterval = params.GetInterval( "FilterWidth" );
	const CInterval strideHeightInterval = params.GetInterval( "StrideHeight" );
	const CInterval strideWidthInterval = params.GetInterval( "StrideWidth" );
	const CInterval paddingHeightInterval = params.GetInterval( "PaddingHeight" );
	const CInterval paddingWidthInterval = params.GetInterval( "PaddingWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int listSize = random.UniformInt( listSizeInterval.Begin, listSizeInterval.End );
	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int filterHeight = random.UniformInt( filterHeightInterval.Begin, filterHeightInterval.End );
	const int filterWidth = random.UniformInt( filterWidthInterval.Begin, filterWidthInterval.End );
	const int paddingHeight = random.UniformInt( paddingHeightInterval.Begin, paddingHeightInterval.End );
	const int paddingWidth = random.UniformInt( paddingWidthInterval.Begin, paddingWidthInterval.End );
	const int strideHeight = random.UniformInt( strideHeightInterval.Begin, strideHeightInterval.End );
	const int strideWidth = random.UniformInt( strideWidthInterval.Begin, strideWidthInterval.End );

	const int blobSize = batchLength * batchWidth * listSize * inputHeight * inputWidth * channels;

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, blobSize, random )
	CFloatBlob inputBlob( MathEngine(), batchLength, batchWidth, listSize, inputHeight, inputWidth, 1, channels );
	inputBlob.CopyFrom( inputData.data() );

	CREATE_FILL_FLOAT_ARRAY( filterData, valuesInterval.Begin, valuesInterval.End, channels * filterHeight * filterWidth, random )
	CFloatBlob filterBlob( MathEngine(), 1, filterHeight, filterWidth, 1, channels );
	filterBlob.CopyFrom( filterData.data() );

	std::vector<float> freeTermData;
	freeTermData.resize( channels );
	for( size_t i = 0; i < freeTermData.size(); i++ ) {
		freeTermData[i] = static_cast<float>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) );
	}
	CFloatBlob freeTermBlob( MathEngine(), 1, 1, 1, channels );
	freeTermBlob.CopyFrom( freeTermData.data() );

	const int outHeight = 1 + ( inputHeight + 2 * paddingHeight - filterHeight ) / strideHeight;
	const int outWidth = 1 + ( inputWidth + 2 * paddingWidth - filterWidth ) / strideWidth;
	std::vector<float> expectedData;
	expectedData.resize( batchLength * batchWidth * listSize * outHeight * outWidth * channels );

	blobChannelwiseConvolutionNaive( batchLength * batchWidth * listSize, channels, inputHeight, inputWidth,
		filterHeight, filterWidth,
		paddingHeight, paddingWidth,
		strideHeight, strideWidth,
		inputData.data(), filterData.data(), freeTermData.data(), expectedData.data() );

	CFloatBlob outBlob( MathEngine(), batchLength, batchWidth, listSize, outHeight, outWidth, 1, channels );
	CConstFloatHandle ft = freeTermBlob.GetData();

	CChannelwiseConvolutionDesc* convDesc = MathEngine().InitBlobChannelwiseConvolution( inputBlob.GetDesc(),
		paddingHeight, paddingWidth, strideHeight, strideWidth, filterBlob.GetDesc(), &freeTermBlob.GetDesc(), outBlob.GetDesc() );
	MathEngine().BlobChannelwiseConvolution( *convDesc, inputBlob.GetData(),
		filterBlob.GetData(), &ft, outBlob.GetData() );
	delete convDesc;

	std::vector<float> resultData;
	resultData.resize( batchLength * batchWidth * listSize * outHeight * outWidth * channels );
	outBlob.CopyTo( resultData.data() );

	for( size_t i = 0; i < resultData.size(); i++ ) {
		ASSERT_NEAR( expectedData[i], resultData[i], 1e-3f );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBlobChannelwiseConvolutionTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineBlobChannelwiseConvolutionTestInstantiation, CMathEngineBlobChannelwiseConvolutionTest,
	::testing::Values(
		CTestParams(
			"InputHeight = (5..15);"
			"InputWidth = (5..15);"
			"Channels = 16;"
			"BatchLength = (1..3);"
			"BatchWidth = (1..3);"
			"ListSize = (1..3);"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"StrideHeight = (1..2);"
			"StrideWidth = (1..2);"
			"Values = (-10..10);"
			"TestCount = 100;"
		),
		CTestParams(
			"InputHeight = (5..15);"
			"InputWidth = (5..15);"
			"Channels = (1..16);"
			"BatchLength = (1..3);"
			"BatchWidth = (1..3);"
			"ListSize = (1..3);"
			"FilterHeight = (3..5);"
			"FilterWidth = (3..5);"
			"PaddingHeight = (0..2);"
			"PaddingWidth = (0..2);"
			"StrideHeight = (1..2);"
			"StrideWidth = (1..2);"
			"Values = (-10..10);"
			"TestCount = 100;"
		),
		CTestParams(
			"InputHeight = 5;"
			"InputWidth = 6;"
			"Channels = 5;"
			"BatchLength = 4;"
			"BatchWidth = 3;"
			"ListSize = 2;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"PaddingHeight = 0;"
			"PaddingWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputHeight = 7;"
			"InputWidth = 8;"
			"Channels = 3;"
			"BatchLength = 2;"
			"BatchWidth = 3;"
			"ListSize = 2;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"PaddingHeight = 0;"
			"PaddingWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputHeight = 1;"
			"InputWidth = 1;"
			"Channels = 4;"
			"BatchLength = 2;"
			"BatchWidth = 3;"
			"ListSize = 2;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputHeight = 1;"
			"InputWidth = 1;"
			"Channels = 4;"
			"BatchLength = 2;"
			"BatchWidth = 3;"
			"ListSize = 2;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"InputHeight = 2;"
			"InputWidth = 2;"
			"Channels = 4;"
			"BatchLength = 1;"
			"BatchWidth = 1;"
			"ListSize = 1;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"Values = (-10..10);"
			"TestCount = 1;"
		)
	)
);

TEST_P(CMathEngineBlobChannelwiseConvolutionTest, Random)
{
	RUN_TEST_IMPL(blobChannelwiseConvolutionTestImpl)
}
