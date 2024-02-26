/* Copyright © 2017-2024 ABBYY

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

static void maxPoolingBackwardNaive( const CPoolingTestParams& params, const float* resultDiff, const int* maxIndices, float* sourceDiff )
{
	const int sourceObjectSize = params.InputHeight * params.InputWidth * params.InputDepth * params.InputChannels;
	const int channels = params.InputDepth * params.InputChannels;

	for( int b = 0; b < params.InputCount; ++b ) {
		for( int y = 0; y < params.OutputHeight; ++y ) {
			for( int x = 0; x < params.OutputWidth; ++x ) {
				for( int c = 0; c < channels; ++c ) {
					const int index = b * params.OutputHeight * params.OutputWidth * channels +
						y * params.OutputWidth * channels + x * channels + c;
					const int maxIndex = maxIndices[index];
					sourceDiff[b * sourceObjectSize + maxIndex] += resultDiff[index];
				}
			}
		}
	}
}

static void blobMaxPoolingBackwardTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const auto poolingParams = getPoolingParams( params, random );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	const int sourceSize = poolingParams.InputCount * poolingParams.InputHeight * poolingParams.InputWidth
		* poolingParams.InputDepth * poolingParams.InputChannels;
	const int resultSize = poolingParams.InputCount * poolingParams.OutputHeight * poolingParams.OutputWidth
		* poolingParams.InputDepth * poolingParams.InputChannels;

	CREATE_FILL_FLOAT_ARRAY( resultDiffData, valuesInterval.Begin, valuesInterval.End, resultSize, random )
	CFloatBlob resultDiffBlob( MathEngine(),
		poolingParams.InputCount, poolingParams.OutputHeight, poolingParams.OutputWidth, poolingParams.InputDepth, poolingParams.InputChannels );
	resultDiffBlob.CopyFrom( resultDiffData.data() );
	CFloatBlob resultBlob( MathEngine(),
		poolingParams.InputCount, poolingParams.OutputHeight, poolingParams.OutputWidth, poolingParams.InputDepth, poolingParams.InputChannels );

	CREATE_FILL_FLOAT_ARRAY( sourceData, valuesInterval.Begin, valuesInterval.End, sourceSize, random )
	CFloatBlob sourceDataBlob( MathEngine(), poolingParams.InputCount, poolingParams.InputHeight,
		poolingParams.InputWidth, poolingParams.InputDepth, poolingParams.InputChannels );
	sourceDataBlob.CopyFrom( sourceData.data() );

	CFloatBlob sourceDiffBlob( MathEngine(), poolingParams.InputCount, poolingParams.InputHeight,
		poolingParams.InputWidth, poolingParams.InputDepth, poolingParams.InputChannels );

	CIntBlob indexBlob( MathEngine(),
		poolingParams.InputCount, poolingParams.OutputHeight, poolingParams.OutputWidth, poolingParams.InputDepth, poolingParams.InputChannels );
	CIntHandle indexBlobPtr = indexBlob.GetData();

	auto poolingDesc = MathEngine().InitMaxPooling( sourceDataBlob.GetDesc(), poolingParams.FilterHeight, poolingParams.FilterWidth,
		poolingParams.StrideHeight, poolingParams.StrideWidth, resultDiffBlob.GetDesc() );
	MathEngine().BlobMaxPooling( *poolingDesc, sourceDataBlob.GetData(), &indexBlobPtr, resultBlob.GetData() );

	MathEngine().BlobMaxPoolingBackward( *poolingDesc, resultDiffBlob.GetData(), indexBlobPtr, sourceDiffBlob.GetData() );
	delete poolingDesc;

	std::vector<int> maxIndices;
	maxIndices.resize( resultSize );
	indexBlob.CopyTo( maxIndices.data() );

	std::vector<float> actualDiff, expectedDiff;
	actualDiff.resize( sourceSize );
	sourceDiffBlob.CopyTo( actualDiff.data() );
	expectedDiff.insert( expectedDiff.begin(), sourceSize, 0 );

	maxPoolingBackwardNaive( poolingParams, resultDiffData.data(), maxIndices.data(), expectedDiff.data() );

	for( int i = 0; i < sourceSize; ++i ) {
		ASSERT_NEAR( expectedDiff[i], actualDiff[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMathEngineBlobMaxPoolingBackwardTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineBlobMaxPoolingBackwardTestInstantiation, CMathEngineBlobMaxPoolingBackwardTest,
	::testing::Values(
		CTestParams(
			"InputHeight = (5..15);"
			"InputWidth = (5..15);"
			"InputDepth = (5..15);"
			"Channels = (1..3);"
			"BatchSize = (1..32);"
			"FilterHeight = (1..5);"
			"FilterWidth = (1..5);"
			"StrideHeight = (1..2);"
			"StrideWidth = (1..2);"
			"Values = (-10..10);"
			"TestCount = 100;"
		)
	)
);


TEST_P( CMathEngineBlobMaxPoolingBackwardTest, Random )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		GTEST_LOG_(INFO) << "Skipped rest of test for MathEngine type=" << int(met) << " because no implementation.\n";
		return;
	}

	RUN_TEST_IMPL( blobMaxPoolingBackwardTestImpl )
}
