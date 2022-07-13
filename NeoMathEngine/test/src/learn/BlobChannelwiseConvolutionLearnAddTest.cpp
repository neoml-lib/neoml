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

static void blobChannelwiseConvolutionLearnAddNaive(
	const float* input, float* filter, float* freeTerm, const float* output,
	int batchSize, int channels, int inputHeight, int inputWidth, int filterHeight, int filterWidth,
	int paddingHeight, int paddingWidth, int strideHeight, int strideWidth )
{
	const int outputHeight = calcConvOutputSize( inputHeight, paddingHeight, filterHeight, 1, strideHeight );
	const int outputWidth = calcConvOutputSize( inputWidth, paddingWidth, filterWidth, 1, strideWidth );
	const int outputObjectSize = outputWidth * outputHeight * channels;
	const int inputObjectSize = inputHeight * inputWidth * channels;

	for( int b = 0; b < batchSize; ++b ) {
		for( int c = 0; c < channels; ++c ) {
			for( int h = 0; h < outputHeight; ++h ) {
				for( int w = 0; w < outputWidth; ++w ) {
					const int outputIndex = b * outputObjectSize + h * outputWidth * channels + w * channels + c;
					freeTerm[c] += output[outputIndex];

					for( int filterH = 0; filterH < filterHeight; ++filterH ) {
						const int inputH = h * strideHeight - paddingHeight + filterH;
						if( inputH < 0 || inputH >= inputHeight ) {
							continue;
						}
						for( int filterW = 0; filterW < filterWidth; ++filterW ) {
							const int inputW = w * strideWidth - paddingWidth + filterW;
							if( inputW < 0 || inputW >= inputWidth ) {
								continue;
							}
							const int inputIndex = b * inputObjectSize + inputH * inputWidth * channels +
								inputW * channels + c;
							const int filterIndex = filterH * filterWidth * channels + filterW * channels + c;
							filter[filterIndex] += input[inputIndex] * output[outputIndex];
						}
					}
				}
			}
		}
	}
}

static void blobChannelwiseConvolutionLearnAddImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval filterHeightInterval = params.GetInterval( "FilterHeight" );
	const CInterval filterWidthInterval = params.GetInterval( "FilterWidth" );
	const CInterval strideHeightInterval = params.GetInterval( "StrideHeight" );
	const CInterval strideWidthInterval = params.GetInterval( "StrideWidth" );
	const CInterval paddingHeightInterval = params.GetInterval( "PaddingHeight" );
	const CInterval paddingWidthInterval = params.GetInterval( "PaddingWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const bool addFreeTerm = params.GetValue<int>( "AddFreeTerm" ) == 1;

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int paddingHeight = random.UniformInt( paddingHeightInterval.Begin, paddingHeightInterval.End );
	const int paddingWidth = random.UniformInt( paddingWidthInterval.Begin, paddingWidthInterval.End );
	const int strideHeight = random.UniformInt( strideHeightInterval.Begin, strideHeightInterval.End );
	const int strideWidth = random.UniformInt( strideWidthInterval.Begin, strideWidthInterval.End );
	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );
	const int filterHeight = random.UniformInt( filterHeightInterval.Begin, filterHeightInterval.End );
	const int filterWidth = random.UniformInt( filterWidthInterval.Begin, filterWidthInterval.End );

	const int outputHeight = calcConvOutputSize( inputHeight, paddingHeight, filterHeight, 1, strideHeight );
	const int outputWidth = calcConvOutputSize( inputWidth, paddingWidth, filterWidth, 1, strideWidth );

	CREATE_FILL_FLOAT_ARRAY( outputDiffData, valuesInterval.Begin, valuesInterval.End, batchSize * outputHeight * outputWidth * channels, random )
	CFloatBlob outputDiffBlob( MathEngine(), batchSize, outputHeight, outputWidth, 1, channels );
	outputDiffBlob.CopyFrom( outputDiffData.data() );

	CREATE_FILL_FLOAT_ARRAY( filterData, valuesInterval.Begin, valuesInterval.End, filterHeight * filterWidth * channels, random )
	CFloatBlob filterBlob( MathEngine(), 1, filterHeight, filterWidth, 1, channels );
	filterBlob.CopyFrom( filterData.data() );
	std::vector<float> expectedFilterData = filterData;

	CREATE_FILL_FLOAT_ARRAY( freeTermData, valuesInterval.Begin, valuesInterval.End, channels, random )
	CFloatBlob freeTermBlob( MathEngine(), 1, 1, 1, 1, channels );
	freeTermBlob.CopyFrom( freeTermData.data() );
	std::vector<float> expectedFreeTermData = freeTermData;

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, batchSize * inputHeight * inputWidth * channels, random )
	CFloatBlob inputBlob( MathEngine(), batchSize, inputHeight, inputWidth, 1, channels );
	inputBlob.CopyFrom( inputData.data() );

	CChannelwiseConvolutionDesc* convDesc = MathEngine().InitBlobChannelwiseConvolution(
		inputBlob.GetDesc(), paddingHeight, paddingWidth, strideHeight, strideWidth, filterBlob.GetDesc(), &freeTermBlob.GetDesc(), outputDiffBlob.GetDesc() );
	CFloatHandle freeTermDiffHandle = freeTermBlob.GetData();
	MathEngine().BlobChannelwiseConvolutionLearnAdd( *convDesc, inputBlob.GetData(), outputDiffBlob.GetData(),
		filterBlob.GetData(), addFreeTerm ? &freeTermDiffHandle : 0 );
	delete convDesc;

	blobChannelwiseConvolutionLearnAddNaive(
		inputData.data(), expectedFilterData.data(), expectedFreeTermData.data(), outputDiffData.data(),
		batchSize, channels, inputHeight, inputWidth, filterHeight, filterWidth,
		paddingHeight, paddingWidth, strideHeight, strideWidth );

	filterBlob.CopyTo( filterData.data() );
	for( size_t i = 0; i < filterData.size(); ++i ) {
		ASSERT_NEAR( expectedFilterData[i], filterData[i], 1e-2f );
	}

	if( addFreeTerm ) {
		freeTermBlob.CopyTo( freeTermData.data() );
		for( size_t i = 0; i < freeTermData.size(); ++i ) {
			ASSERT_NEAR( expectedFreeTermData[i], freeTermData[i], 1e-2f );
		}
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBlobChannelwiseConvolutionLearnAddTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineBlobChannelwiseConvolutionLearnAddTestInstantiation, CMathEngineBlobChannelwiseConvolutionLearnAddTest,
	::testing::Values(
		CTestParams(
			"InputHeight = (5..16);"
			"InputWidth = (5..16);"
			"Channels = (1..4);"
			"BatchSize = (1..4);"
			"FilterHeight = (2..5);"
			"FilterWidth = (2..5);"
			"PaddingHeight = (0..1);"
			"PaddingWidth = (0..1);"
			"StrideHeight = (1..4);"
			"StrideWidth = (1..4);"
			"Values = (-10..10);"
			"AddFreeTerm = (0..1);"
			"TestCount = 1000;"
		),
		CTestParams(
			"InputHeight = 7;"
			"InputWidth = 7;"
			"Channels = (1..4);"
			"BatchSize = (1..4);"
			"FilterHeight = 8;"
			"FilterWidth = 8;"
			"PaddingHeight = 3;"
			"PaddingWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"Values = (-10..10);"
			"AddFreeTerm = (0..1);"
			"TestCount = 10;"
		)
	)
);

TEST_P( CMathEngineBlobChannelwiseConvolutionLearnAddTest, Random )
{
	RUN_TEST_IMPL( blobChannelwiseConvolutionLearnAddImpl );
}
