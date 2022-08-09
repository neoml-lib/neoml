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

static void maxPoolingBackwardNaive( const CPoolingTestParams& params, const float *outputDiff, const int *maxIndices, float *inputDiff )
{
	const int inputObjectSize = params.InputHeight * params.InputWidth * params.InputDepth * params.InputChannels;
	const int channels = params.InputDepth * params.InputChannels;

	for( int b = 0; b < params.InputCount; ++b ) {
		for( int y = 0; y < params.OutputHeight; ++y ) {
			for( int x = 0; x < params.OutputWidth; ++x ) {
				for( int c = 0; c < channels; ++c ) {
					const int index = b * params.OutputHeight * params.OutputWidth * channels +
						y * params.OutputWidth * channels + x * channels + c;
					const int maxIndex = maxIndices[index];
					const float diff = outputDiff[index];
					inputDiff[b * inputObjectSize + maxIndex] += diff;
				}
			}
		}
	}
}

static void blobMaxPoolingBackwardTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	auto poolingParams = getPoolingParams( params, random );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	const int inputSize = poolingParams.InputCount * poolingParams.InputHeight * poolingParams.InputWidth
		* poolingParams.InputDepth * poolingParams.InputChannels;
	const int outputSize = poolingParams.InputCount * poolingParams.OutputHeight * poolingParams.OutputWidth
		* poolingParams.InputDepth * poolingParams.InputChannels;

	CREATE_FILL_FLOAT_ARRAY( outputDiffData, valuesInterval.Begin, valuesInterval.End, outputSize, random )
	CFloatBlob outputDiffBlob( MathEngine(),
		poolingParams.InputCount, poolingParams.OutputHeight, poolingParams.OutputWidth, poolingParams.InputDepth, poolingParams.InputChannels );
	outputDiffBlob.CopyFrom( outputDiffData.data() );
	CFloatBlob outputBlob( MathEngine(),
		poolingParams.InputCount, poolingParams.OutputHeight, poolingParams.OutputWidth, poolingParams.InputDepth, poolingParams.InputChannels );

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, inputSize, random )
	CFloatBlob inputDataBlob( MathEngine(), poolingParams.InputCount, poolingParams.InputHeight,
		poolingParams.InputWidth, poolingParams.InputDepth, poolingParams.InputChannels );
	inputDataBlob.CopyFrom( inputData.data() );

	CFloatBlob inputDiffBlob( MathEngine(), poolingParams.InputCount, poolingParams.InputHeight,
		poolingParams.InputWidth, poolingParams.InputDepth, poolingParams.InputChannels );

	CIntBlob indexBlob( MathEngine(),
		poolingParams.InputCount, poolingParams.OutputHeight, poolingParams.OutputWidth, poolingParams.InputDepth, poolingParams.InputChannels );
	CIntHandle indexBlobPtr = indexBlob.GetData();

	auto poolingDesc = MathEngine().InitMaxPooling( inputDataBlob.GetDesc(), poolingParams.FilterHeight, poolingParams.FilterWidth,
		poolingParams.StrideHeight, poolingParams.StrideWidth, outputDiffBlob.GetDesc() );
	MathEngine().BlobMaxPooling( *poolingDesc, inputDataBlob.GetData(), &indexBlobPtr, outputBlob.GetData() );

	MathEngine().BlobMaxPoolingBackward( *poolingDesc, outputDiffBlob.GetData(), indexBlobPtr, inputDiffBlob.GetData() );
	delete poolingDesc;

	std::vector<int> maxIndices;
	maxIndices.resize( outputSize );
	indexBlob.CopyTo( maxIndices.data() );

	std::vector<float> actualDiff, expectedDiff;
	actualDiff.resize( inputSize );
	inputDiffBlob.CopyTo( actualDiff.data() );
	expectedDiff.insert( expectedDiff.begin(), inputSize, 0 );

	maxPoolingBackwardNaive( poolingParams, outputDiffData.data(), maxIndices.data(), expectedDiff.data() );

	for( int i = 0; i < inputSize; ++i ) {
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
	RUN_TEST_IMPL( blobMaxPoolingBackwardTestImpl )
}
