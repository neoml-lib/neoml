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

struct CConv3dBackwardTestParams {
	int InputCount;
	int InputHeight;
	int InputWidth;
	int InputDepth;
	int InputChannels;

	int PaddingHeight;
	int PaddingWidth;
	int PaddingDepth;

	int FilterCount;
	int FilterHeight;
	int FilterWidth;
	int FilterDepth;

	int StrideHeight;
	int StrideWidth;
	int StrideDepth;

	CConv3dBackwardTestParams( int inputCount, int inputHeight, int inputWidth, int inputDepth, int inputChannels,
		int paddingHeight, int paddingWidth, int paddingDepth, int filterCount, int filterHeight, int filterWidth, int filterDepth,
		int strideHeight, int strideWidth, int strideDepth ) :
		InputCount( inputCount ),
		InputHeight( inputHeight ),
		InputWidth( inputWidth ),
		InputDepth( inputDepth ),
		InputChannels( inputChannels ),
		PaddingHeight( paddingHeight ),
		PaddingWidth( paddingWidth ),
		PaddingDepth( paddingDepth ),
		FilterCount( filterCount ),
		FilterHeight( filterHeight ),
		FilterWidth( filterWidth ),
		FilterDepth( filterDepth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth ),
		StrideDepth( strideDepth )
	{
	}
};

static inline int calcConvOutputSize( int input, int padding, int filter, int stride )
{
	return  1 + ( input + 2 * padding - filter ) / stride;
}

static inline float* getInputElem( const CConv3dBackwardTestParams& params, int h, int w, int d, int ch, float* input )
{
	const int channels = params.InputChannels;

	if( h < 0 || w < 0 || d < 0 || h >= params.InputHeight || w >= params.InputWidth || d >= params.InputDepth ) {
		return 0;
	}

	return input + ( h * params.InputWidth * params.InputDepth * channels + w * params.InputDepth * channels + d * channels + ch );
}

static void calcOneStepConvolutionBackward( const CConv3dBackwardTestParams& params, int fromH, int fromW, int fromD, float* input,
	float* filter, const float& output )
{
	const int channels = params.InputChannels;

	for( int h = 0; h < params.FilterHeight; ++h ) {
		for( int w = 0; w < params.FilterWidth; ++w ) {
			for( int d = 0; d < params.FilterDepth; ++d ) {
				for( int ch = 0; ch < params.InputChannels; ++ch ) {
					float* inputPtr = getInputElem( params, fromH + h, fromW + w, fromD + d, ch, input );
					int filterIndex = h * params.FilterWidth * params.FilterDepth * channels + w * params.FilterDepth * channels + d * channels + ch;
					if( inputPtr != 0 ) {
						*inputPtr += output * filter[filterIndex];
					}
				}
			}
		}
	}
}

static void convolutionBackward( const CConv3dBackwardTestParams& params,
	float *input, float *filter, float *freeTerm, float *output )
{
	const int outputHeight = calcConvOutputSize( params.InputHeight, params.PaddingHeight, params.FilterHeight, params.StrideHeight );
	const int outputWidth = calcConvOutputSize( params.InputWidth, params.PaddingWidth, params.FilterWidth, params.StrideWidth );
	const int outputDepth = calcConvOutputSize( params.InputDepth, params.PaddingDepth, params.FilterDepth, params.StrideDepth );
	const int filterObjectSize = params.InputChannels * params.FilterHeight * params.FilterWidth * params.FilterDepth;

	for( int h = 0; h < outputHeight; ++h ) {
		for( int w = 0; w < outputWidth; ++w ) {
			for( int d = 0; d < outputDepth; ++d ) {
				for( int ch = 0; ch < params.FilterCount; ++ch ) {
					const int outputIndex = h * outputWidth * outputDepth * params.FilterCount + w * outputDepth * params.FilterCount + d * params.FilterCount + ch;
					calcOneStepConvolutionBackward( params, h * params.StrideHeight - params.PaddingHeight,
						w * params.StrideWidth - params.PaddingWidth, d * params.StrideDepth - params.PaddingDepth,
						input, filter + filterObjectSize * ch, output[outputIndex] );
				}
			}
		}
	}

	if( freeTerm != 0 ) {
		for( int h = 0; h < params.InputHeight; ++h ) {
			for( int w = 0; w < params.InputWidth; ++w ) {
				for( int d = 0; d < params.InputDepth; ++d ) {
					for( int ch = 0; ch < params.InputChannels; ++ch ) {
						*input++ += freeTerm[ch];
					}
				}
			}
		}
	}
}

static void batchConvolutionBackward( const CConv3dBackwardTestParams& params,
	float *input, float *filter, float *freeTerm, float *output )
{
	const int outputHeight = calcConvOutputSize( params.InputHeight, params.PaddingHeight, params.FilterHeight, params.StrideHeight );
	const int outputWidth = calcConvOutputSize( params.InputWidth, params.PaddingWidth, params.FilterWidth, params.StrideWidth );
	const int outputDepth = calcConvOutputSize( params.InputDepth, params.PaddingDepth, params.FilterDepth, params.StrideDepth );

	const int inputObjectSize = params.InputHeight * params.InputWidth * params.InputDepth * params.InputChannels;
	const int outputObjectSize = params.FilterCount * outputHeight * outputWidth * outputDepth;

	for( int b = 0; b < params.InputCount; ++b ) {
		convolutionBackward( params, input + b * inputObjectSize, filter, freeTerm,
			output + b * outputObjectSize );
	}
}

static CConv3dBackwardTestParams getConv3dParams( const CTestParams& params, CRandom& random )
{
	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval inputDepthInterval = params.GetInterval( "InputDepth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval filterCountInterval = params.GetInterval( "FilterCount" );
	const CInterval filterHeightInterval = params.GetInterval( "FilterHeight" );
	const CInterval filterWidthInterval = params.GetInterval( "FilterWidth" );
	const CInterval filterDepthInterval = params.GetInterval( "FilterDepth" );
	const CInterval strideHeightInterval = params.GetInterval( "StrideHeight" );
	const CInterval strideWidthInterval = params.GetInterval( "StrideWidth" );
	const CInterval strideDepthInterval = params.GetInterval( "StrideDepth" );
	const CInterval paddingHeightInterval = params.GetInterval( "PaddingHeight" );
	const CInterval paddingWidthInterval = params.GetInterval( "PaddingWidth" );
	const CInterval paddingDepthInterval = params.GetInterval( "PaddingDepth" );

	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );
	const int inputDepth = random.UniformInt( inputDepthInterval.Begin, inputDepthInterval.End );

	const int inputChannels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int filterCount = random.UniformInt( filterCountInterval.Begin, filterCountInterval.End );

	const int filterHeight = random.UniformInt( filterHeightInterval.Begin, filterHeightInterval.End );
	const int filterWidth = random.UniformInt( filterWidthInterval.Begin, filterWidthInterval.End );
	const int filterDepth = random.UniformInt( filterDepthInterval.Begin, filterDepthInterval.End );

	const int strideHeight = random.UniformInt( strideHeightInterval.Begin, strideHeightInterval.End );
	const int strideWidth = random.UniformInt( strideWidthInterval.Begin, strideWidthInterval.End );
	const int strideDepth = random.UniformInt( strideDepthInterval.Begin, strideDepthInterval.End );

	const int paddingHeight = random.UniformInt( paddingHeightInterval.Begin, paddingHeightInterval.End );
	const int paddingWidth = random.UniformInt( paddingWidthInterval.Begin, paddingWidthInterval.End );
	const int paddingDepth = random.UniformInt( paddingDepthInterval.Begin, paddingDepthInterval.End );

	return CConv3dBackwardTestParams( batchSize, inputHeight, inputWidth, inputDepth, inputChannels,
		paddingHeight, paddingWidth, paddingDepth,
		filterCount, filterHeight, filterWidth, filterDepth,
		strideHeight, strideWidth, strideDepth );
}

static void blob3dConvolutionBackwardImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	CConv3dBackwardTestParams convParams = getConv3dParams( params, random );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const bool addFreeTerm = random.UniformInt( 0, 2 ) != 0 ? true : false;

	const int outHeight = calcConvOutputSize( convParams.InputHeight, convParams.PaddingHeight, convParams.FilterHeight, convParams.StrideHeight );
	const int outWidth = calcConvOutputSize( convParams.InputWidth, convParams.PaddingWidth, convParams.FilterWidth, convParams.StrideWidth );
	const int outDepth = calcConvOutputSize( convParams.InputDepth, convParams.PaddingDepth, convParams.FilterDepth, convParams.StrideDepth );

	CREATE_FILL_FLOAT_ARRAY( outputData, valuesInterval.Begin, valuesInterval.End, convParams.InputCount * outHeight * outWidth * outDepth * convParams.FilterCount, random )
	CFloatBlob outBlob( MathEngine(), convParams.InputCount, outHeight, outWidth, outDepth, convParams.FilterCount );
	outBlob.CopyFrom( outputData.data() );

	CREATE_FILL_FLOAT_ARRAY( filterData, valuesInterval.Begin, valuesInterval.End,
		convParams.FilterCount * convParams.InputChannels * convParams.FilterHeight * convParams.FilterWidth * convParams.FilterDepth, random )
	CFloatBlob filterBlob( MathEngine(), convParams.FilterCount, convParams.FilterHeight, convParams.FilterWidth, convParams.FilterDepth, convParams.InputChannels );
	filterBlob.CopyFrom( filterData.data() );

	CREATE_FILL_FLOAT_ARRAY( freeTermData, valuesInterval.Begin, valuesInterval.End, convParams.InputChannels, random );
	CFloatBlob freeTermBlob( MathEngine(), 1, 1, 1, convParams.InputChannels );
	freeTermBlob.CopyFrom( freeTermData.data() );

	CFloatBlob inputBlob( MathEngine(), convParams.InputCount, convParams.InputHeight, convParams.InputWidth, convParams.InputDepth, convParams.InputChannels );

	CConstFloatHandle ft = freeTermBlob.GetData();

	C3dConvolutionDesc *convDesc = MathEngine().InitBlob3dConvolution( inputBlob.GetDesc(),
		convParams.PaddingHeight, convParams.PaddingWidth, convParams.PaddingDepth, convParams.StrideHeight, convParams.StrideWidth, convParams.StrideDepth,
		filterBlob.GetDesc(), outBlob.GetDesc() );
	MathEngine().Blob3dConvolutionBackward( *convDesc, outBlob.GetData(),
		filterBlob.GetData(), addFreeTerm ? &ft : nullptr, inputBlob.GetData() );
	delete convDesc;

	std::vector<float> resultData( convParams.InputCount * convParams.InputHeight * convParams.InputWidth * convParams.InputDepth * convParams.InputChannels );
	inputBlob.CopyTo( resultData.data() );

	std::vector<float> inputData;
	inputData.insert( inputData.begin(), convParams.InputCount * convParams.InputHeight * convParams.InputWidth * convParams.InputDepth * convParams.InputChannels, 0 );
	batchConvolutionBackward( convParams, inputData.data(), filterData.data(), addFreeTerm ? freeTermData.data() : 0, outputData.data() );

	for( size_t i = 0; i < resultData.size(); i++ ) {
		ASSERT_NEAR( inputData[i], resultData[i], 1e-2 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CBlob3dConvolutionBackwardTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CBlob3dConvolutionBackwardTestInstantiation, CBlob3dConvolutionBackwardTest,
	::testing::Values(
		CTestParams(
			"InputHeight = (4..7);"
			"InputWidth = (4..7);"
			"InputDepth = (4..7);"
			"Channels = (1..4);"
			"BatchSize = (1..4);"
			"FilterCount = (1..4);"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"FilterDepth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"PaddingDepth = 0;"
			"StrideHeight = (1..2);"
			"StrideWidth = (1..2);"
			"StrideDepth = (1..2);"
			"Values = (-1..1);"
			"TestCount = 50;"
		),
		CTestParams(
			"InputHeight = (5..15);"
			"InputWidth = (5..15);"
			"InputDepth = (5..15);"
			"Channels = (1..3);"
			"BatchSize = (1..3);"
			"FilterCount = (1..3);"
			"FilterHeight = (2..5);"
			"FilterWidth = (2..5);"
			"FilterDepth = (2..5);"
			"PaddingHeight = (0..1);"
			"PaddingWidth = (0..1);"
			"PaddingDepth = (0..1);"
			"StrideHeight = (1..2);"
			"StrideWidth = (1..2);"
			"StrideDepth = (1..2);"
			"Values = (-1..1);"
			"TestCount = 200;"
		),
		CTestParams(
			"InputHeight = (10..15);"
			"InputWidth = (10..15);"
			"InputDepth = (10..15);"
			"Channels = (3..5);"
			"BatchSize = (1..3);"
			"FilterCount = (3..5);"
			"FilterHeight = (5..7);"
			"FilterWidth = (5..7);"
			"FilterDepth = (5..7);"
			"PaddingHeight = (1..3);"
			"PaddingWidth = (1..3);"
			"PaddingDepth = (1..3);"
			"StrideHeight = (1..3);"
			"StrideWidth = (1..3);"
			"StrideDepth = (1..3);"
			"Values = (-1..1);"
			"TestCount = 10;"
		)
	)
);

TEST_P( CBlob3dConvolutionBackwardTest, Random )
{
	RUN_TEST_IMPL( blob3dConvolutionBackwardImpl );
}
