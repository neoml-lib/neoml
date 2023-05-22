/* Copyright © 2017-2023 ABBYY

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

static void mean3dPoolingBackwardNaive( const C3dPoolingTestParams& params, const float* resultDiff, float* sourceDiff )
{
	const int filterSize = params.FilterHeight * params.FilterWidth * params.FilterDepth;
	const int resultHeight = calcConvOutputSize( params.InputHeight, 0, params.FilterHeight, 1, params.StrideHeight );
	const int resultWidth = calcConvOutputSize( params.InputWidth, 0, params.FilterWidth, 1, params.StrideWidth );
	const int resultDepth = calcConvOutputSize( params.InputDepth, 0, params.FilterDepth, 1, params.StrideDepth );

	for( int b = 0; b < params.InputCount; ++b ) {
		for( int c = 0; c < params.InputChannels; ++c ) {
			for( int y = 0; y < resultHeight; ++y ) {
				for( int x = 0; x < resultWidth; ++x ) {
					for( int z = 0; z < resultDepth; ++z ) {
						const float res = resultDiff[b * resultHeight * resultWidth * resultDepth * params.InputChannels +
							y * resultWidth * resultDepth * params.InputChannels + x * resultDepth * params.InputChannels + z * params.InputChannels + c];

						const int sourceHStart = y * params.StrideHeight;
						const int sourceHEnd = sourceHStart + params.FilterHeight;
						const int sourceWStart = x * params.StrideWidth;
						const int sourceWEnd = sourceWStart + params.FilterWidth;
						const int sourceDStart = z * params.StrideDepth;
						const int sourceDEnd = sourceDStart + params.FilterDepth;
						for( int h = sourceHStart; h < sourceHEnd; ++h ) {
							for( int w = sourceWStart; w < sourceWEnd; ++w ) {
								for( int d = sourceDStart; d < sourceDEnd; ++d ) {
									int index = b * params.InputHeight * params.InputWidth * params.InputDepth * params.InputChannels +
										h * params.InputWidth * params.InputDepth * params.InputChannels + w * params.InputDepth * params.InputChannels +
										d * params.InputChannels + c;
									sourceDiff[index] += res / filterSize;
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

	CREATE_FILL_FLOAT_ARRAY( resultDiffData, valuesInterval.Begin, valuesInterval.End, poolingParams.InputCount * outHeight * outWidth * outDepth * poolingParams.InputChannels, random )
	CFloatBlob resultDiffBlob( MathEngine(), poolingParams.InputCount, outHeight, outWidth, outDepth, poolingParams.InputChannels );
	resultDiffBlob.CopyFrom( resultDiffData.data() );

	CFloatBlob sourceDiffBlob( MathEngine(), poolingParams.InputCount, poolingParams.InputHeight,
		poolingParams.InputWidth, poolingParams.InputDepth, poolingParams.InputChannels );

	const auto poolingDesc = MathEngine().Init3dMeanPooling( sourceDiffBlob.GetDesc(), poolingParams.FilterHeight, poolingParams.FilterWidth, poolingParams.FilterDepth,
		poolingParams.StrideHeight, poolingParams.StrideWidth, poolingParams.StrideDepth, resultDiffBlob.GetDesc() );
	MathEngine().Blob3dMeanPoolingBackward( *poolingDesc, resultDiffBlob.GetData(), sourceDiffBlob.GetData() );
	delete poolingDesc;

	const int sourceDiffSize = poolingParams.InputCount * poolingParams.InputHeight * poolingParams.InputWidth * poolingParams.InputDepth * poolingParams.InputChannels;
	std::vector<float> actualDiff, expectedDiff;
	actualDiff.resize( sourceDiffSize );
	sourceDiffBlob.CopyTo( actualDiff.data() );
	expectedDiff.insert( expectedDiff.begin(), sourceDiffSize, 0 );

	mean3dPoolingBackwardNaive( poolingParams, resultDiffData.data(), expectedDiff.data() );

	for( int i = 0; i < sourceDiffSize; ++i ) {
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
