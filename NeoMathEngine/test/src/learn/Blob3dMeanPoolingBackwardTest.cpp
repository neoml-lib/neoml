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

static void mean3dPoolingBackwardNaive( const C3dPoolingTestParams& params, const float *outputDiff, float *inputDiff )
{
	const int filterSize = params.FilterHeight * params.FilterWidth * params.FilterDepth;
	int resultHeight = calcConvOutputSize( params.InputHeight, 0, params.FilterHeight, 1, params.StrideHeight );
	int resultWidth = calcConvOutputSize( params.InputWidth, 0, params.FilterWidth, 1, params.StrideWidth );
	int resultDepth = calcConvOutputSize( params.InputDepth, 0, params.FilterDepth, 1, params.StrideDepth );

	for( int b = 0; b < params.InputCount; ++b ) {
		for( int c = 0; c < params.InputChannels; ++c ) {
			for( int y = 0; y < resultHeight; ++y ) {
				for( int x = 0; x < resultWidth; ++x ) {
					for( int z = 0; z < resultDepth; ++z ) {
						const float res = outputDiff[b * resultHeight * resultWidth * resultDepth * params.InputChannels +
							y * resultWidth * resultDepth * params.InputChannels + x * resultDepth * params.InputChannels + z * params.InputChannels + c];

						int inputHStart = y * params.StrideHeight;
						int inputHEnd = inputHStart + params.FilterHeight;
						int inputWStart = x * params.StrideWidth;
						int inputWEnd = inputWStart + params.FilterWidth;
						int inputDStart = z * params.StrideDepth;
						int inputDEnd = inputDStart + params.FilterDepth;
						for( int h = inputHStart; h < inputHEnd; ++h ) {
							for( int w = inputWStart; w < inputWEnd; ++w ) {
								for( int d = inputDStart; d < inputDEnd; ++d ) {
									int index = b * params.InputHeight * params.InputWidth * params.InputDepth * params.InputChannels + 
										h * params.InputWidth * params.InputDepth * params.InputChannels + w * params.InputDepth * params.InputChannels + 
										d * params.InputChannels + c;
									inputDiff[index] += res / filterSize;
								}
							}
						}
					}
				}
			}
		}
	}
}

static void blob3dMeanPoolingBackwardTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	auto poolingParams = get3dPoolingParams( params, random );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int outHeight = calcConvOutputSize( poolingParams.InputHeight, 0, poolingParams.FilterHeight, 1, poolingParams.StrideHeight );
	const int outWidth = calcConvOutputSize( poolingParams.InputWidth, 0, poolingParams.FilterWidth, 1, poolingParams.StrideWidth );
	const int outDepth = calcConvOutputSize( poolingParams.InputDepth, 0, poolingParams.FilterDepth, 1, poolingParams.StrideDepth );

	CREATE_FILL_FLOAT_ARRAY( outputDiffData, valuesInterval.Begin, valuesInterval.End, poolingParams.InputCount * outHeight * outWidth * outDepth * poolingParams.InputChannels, random )
	CFloatBlob outputDiffBlob( MathEngine(), poolingParams.InputCount, outHeight, outWidth, outDepth, poolingParams.InputChannels );
	outputDiffBlob.CopyFrom( outputDiffData.data() );

	CFloatBlob inputDiffBlob( MathEngine(), poolingParams.InputCount, poolingParams.InputHeight, 
		poolingParams.InputWidth, poolingParams.InputDepth, poolingParams.InputChannels );

	auto poolingDesc = MathEngine().Init3dMeanPooling( inputDiffBlob.GetDesc(), poolingParams.FilterHeight, poolingParams.FilterWidth, poolingParams.FilterDepth,
		poolingParams.StrideHeight, poolingParams.StrideWidth, poolingParams.StrideDepth, outputDiffBlob.GetDesc() );
	MathEngine().Blob3dMeanPoolingBackward( *poolingDesc, outputDiffBlob.GetData(), inputDiffBlob.GetData() );
	delete poolingDesc;

	const int inputDiffSize = poolingParams.InputCount * poolingParams.InputHeight * poolingParams.InputWidth * poolingParams.InputDepth * poolingParams.InputChannels;
	std::vector<float> actualDiff, expectedDiff;
	actualDiff.resize( inputDiffSize );
	inputDiffBlob.CopyTo( actualDiff.data() );
	expectedDiff.insert( expectedDiff.begin(), inputDiffSize, 0 );

	mean3dPoolingBackwardNaive( poolingParams, outputDiffData.data(), expectedDiff.data() );

	for( int i = 0; i < inputDiffSize; ++i ) {
		ASSERT_NEAR( expectedDiff[i], actualDiff[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBlob3dMeanPoolingBackwardTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineBlob3dMeanPoolingBackwardTestInstantiation, CMathEngineBlob3dMeanPoolingBackwardTest,
	::testing::Values(
		CTestParams(
			"InputHeight = (5..15);"
			"InputWidth = (5..15);"
			"InputDepth = (5..15);"
			"Channels = (1..3);"
			"BatchSize = (1..3);"
			"FilterHeight = (1..5);"
			"FilterWidth = (1..5);"
			"FilterDepth = (1..5);"
			"StrideHeight = (1..2);"
			"StrideWidth = (1..2);"
			"StrideDepth = (1..2);"
			"Values = (-10..10);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CMathEngineBlob3dMeanPoolingBackwardTest, Random )
{
	RUN_TEST_IMPL( blob3dMeanPoolingBackwardTestImpl )
}
