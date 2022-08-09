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

static void calcOneStepConvolutionLearnAdd( const CConv3dTestParams& params, int fromH, int fromW, int fromD, float* input,
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
						filter[filterIndex] += *inputPtr * output;
					}
				}
			}
		}
	}
}

static void convolutionLearnAdd( const CConv3dTestParams& params,
	float *input, float *filter, float *freeTerm, float *output )
{
	const int outputHeight = calcConvOutputSize( params.InputHeight, params.PaddingHeight, params.FilterHeight, 1, params.StrideHeight );
	const int outputWidth = calcConvOutputSize( params.InputWidth, params.PaddingWidth, params.FilterWidth, 1, params.StrideWidth );
	const int outputDepth = calcConvOutputSize( params.InputDepth, params.PaddingDepth, params.FilterDepth, 1, params.StrideDepth );
	const int filterObjectSize = params.InputChannels * params.FilterHeight * params.FilterWidth * params.FilterDepth;

	for( int h = 0; h < outputHeight; ++h ) {
		for( int w = 0; w < outputWidth; ++w ) {
			for( int d = 0; d < outputDepth; ++d ) {
				for( int ch = 0; ch < params.FilterCount; ++ch ) {
					const int outputIndex = h * outputWidth * outputDepth * params.FilterCount + w * outputDepth * params.FilterCount + d * params.FilterCount + ch;
					calcOneStepConvolutionLearnAdd( params, h * params.StrideHeight - params.PaddingHeight,
						w * params.StrideWidth - params.PaddingWidth, d * params.StrideDepth - params.PaddingDepth,
						input, filter + filterObjectSize * ch, output[outputIndex] );
					freeTerm[ch] += output[outputIndex];
				}
			}
		}
	}
}

static void batchConvolutionLearnAdd( const CConv3dTestParams& params,
	float *input, float *filter, float *freeTerm, float *output )
{
	const int outputHeight = calcConvOutputSize( params.InputHeight, params.PaddingHeight, params.FilterHeight, 1, params.StrideHeight );
	const int outputWidth = calcConvOutputSize( params.InputWidth, params.PaddingWidth, params.FilterWidth, 1, params.StrideWidth );
	const int outputDepth = calcConvOutputSize( params.InputDepth, params.PaddingDepth, params.FilterDepth, 1, params.StrideDepth );

	const int inputObjectSize = params.InputHeight * params.InputWidth * params.InputDepth * params.InputChannels;
	const int outputObjectSize = params.FilterCount * outputHeight * outputWidth * outputDepth;

	for( int b = 0; b < params.InputCount; ++b ) {
		convolutionLearnAdd( params, input + b * inputObjectSize, filter, freeTerm,
			output + b * outputObjectSize );
	}
}

static void blob3dConvolutionLearnAddImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	CConv3dTestParams convParams = getConv3dParams( params, random );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const bool addFreeTerm = random.UniformInt( 0, 2 ) != 0 ? true : false;

	const int outHeight = calcConvOutputSize( convParams.InputHeight, convParams.PaddingHeight, convParams.FilterHeight, 1, convParams.StrideHeight );
	const int outWidth = calcConvOutputSize( convParams.InputWidth, convParams.PaddingWidth, convParams.FilterWidth, 1, convParams.StrideWidth );
	const int outDepth = calcConvOutputSize( convParams.InputDepth, convParams.PaddingDepth, convParams.FilterDepth, 1, convParams.StrideDepth );

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, convParams.InputCount * convParams.InputHeight *
		convParams.InputWidth * convParams.InputDepth * convParams.InputChannels, random )
	CFloatBlob inputBlob( MathEngine(), convParams.InputCount, convParams.InputHeight, convParams.InputWidth, convParams.InputDepth, convParams.InputChannels );
	inputBlob.CopyFrom( inputData.data() );

	CREATE_FILL_FLOAT_ARRAY( outputDiff, valuesInterval.Begin, valuesInterval.End, convParams.InputCount * outHeight * outWidth * outDepth * convParams.FilterCount, random )
	CFloatBlob outDiffBlob( MathEngine(), convParams.InputCount, outHeight, outWidth, outDepth, convParams.FilterCount );
	outDiffBlob.CopyFrom( outputDiff.data() );

	CREATE_FILL_FLOAT_ARRAY( freeTermDiff, valuesInterval.Begin, valuesInterval.End, convParams.FilterCount, random );
	CFloatBlob freeTermDiffBlob( MathEngine(), 1, 1, 1, convParams.FilterCount );
	freeTermDiffBlob.CopyFrom( freeTermDiff.data() );
	CFloatHandle ft = freeTermDiffBlob.GetData();
	std::vector<float> expectedFreeTermDiff = freeTermDiff;

	const int filterSize = convParams.FilterCount * convParams.FilterHeight * convParams.FilterWidth * convParams.FilterDepth * convParams.InputChannels;

	CREATE_FILL_FLOAT_ARRAY( filterDiff, valuesInterval.Begin, valuesInterval.End, filterSize, random );
	CFloatBlob filterDiffBlob( MathEngine(), convParams.FilterCount, convParams.FilterHeight, convParams.FilterWidth, convParams.FilterDepth, convParams.InputChannels );
	filterDiffBlob.CopyFrom( filterDiff.data() );
	std::vector<float> expectedFilterDiff = filterDiff;

	C3dConvolutionDesc *convDesc = MathEngine().InitBlob3dConvolution( inputBlob.GetDesc(),
		convParams.PaddingHeight, convParams.PaddingWidth, convParams.PaddingDepth, convParams.StrideHeight, convParams.StrideWidth, convParams.StrideDepth,
		filterDiffBlob.GetDesc(), outDiffBlob.GetDesc() );
	MathEngine().Blob3dConvolutionLearnAdd( *convDesc, inputBlob.GetData(), outDiffBlob.GetData(),
		filterDiffBlob.GetData(), addFreeTerm ? &ft : nullptr, false );
	delete convDesc;

	filterDiffBlob.CopyTo( filterDiff.data() );

	batchConvolutionLearnAdd( convParams, inputData.data(), expectedFilterDiff.data(), expectedFreeTermDiff.data(), outputDiff.data() );

	for( size_t i = 0; i < expectedFilterDiff.size(); i++ ) {
		ASSERT_NEAR( expectedFilterDiff[i], filterDiff[i], 1e-2 );
	}

	if( addFreeTerm ) {
		freeTermDiffBlob.CopyTo( freeTermDiff.data() );

		for( size_t i = 0; i < expectedFreeTermDiff.size(); i++ ) {
			ASSERT_NEAR( expectedFreeTermDiff[i], freeTermDiff[i], 1e-2 );
		}
	}
}

//------------------------------------------------------------------------------------------------------------

class CBlob3dConvolutionLearnAddTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CBlob3dConvolutionLearnAddTestInstantiation, CBlob3dConvolutionLearnAddTest,
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

TEST_P( CBlob3dConvolutionLearnAddTest, Random )
{
	RUN_TEST_IMPL( blob3dConvolutionLearnAddImpl );
}
