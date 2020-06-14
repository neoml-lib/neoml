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

struct C3dPoolingTestParams {
	int InputCount;
	int InputHeight;
	int InputWidth;
	int InputDepth;
	int InputChannels;

	int FilterHeight;
	int FilterWidth;
	int FilterDepth;

	int StrideHeight;
	int StrideWidth;
	int StrideDepth;

	bool IsMaxPooing = false;

	C3dPoolingTestParams( int inputCount, int inputHeight, int inputWidth, int inputDepth, int inputChannels,
		int filterHeight, int filterWidth, int filterDepth, int strideHeight, int strideWidth, int strideDepth ) :
		InputCount( inputCount ),
		InputHeight( inputHeight ),
		InputWidth( inputWidth ),
		InputDepth( inputDepth ),
		InputChannels( inputChannels ),
		FilterHeight( filterHeight ),
		FilterWidth( filterWidth ),
		FilterDepth( filterDepth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth ),
		StrideDepth( strideDepth )
	{}
};

static int calcConvDimSize( int input, int filter, int stride ) 
{
	return ( input - filter ) / stride + 1;
}

static void max3dPoolingBackwardNaive( const C3dPoolingTestParams& params, const float *outputDiff, const int *maxIndices, float *inputDiff )
{
	const int inputObjectSize = params.InputHeight * params.InputWidth * params.InputDepth * params.InputChannels;
	int resultHeight = calcConvDimSize( params.InputHeight, params.FilterHeight, params.StrideHeight );
	int resultWidth = calcConvDimSize( params.InputWidth, params.FilterWidth, params.StrideWidth );
	int resultDepth = calcConvDimSize( params.InputDepth, params.FilterDepth, params.StrideDepth );

	for( int b = 0; b < params.InputCount; ++b ) {
		for( int y = 0; y < resultHeight; ++y ) {
			for( int x = 0; x < resultWidth; ++x ) {
				for( int z = 0; z < resultDepth; ++z ) {
					for( int c = 0; c < params.InputChannels; ++c ) {
						const int index = b * resultHeight * resultWidth * resultDepth * params.InputChannels +
							y * resultWidth * resultDepth * params.InputChannels + x * resultDepth * params.InputChannels + z * params.InputChannels + c;
						const int maxIndex = maxIndices[index];
						const float diff = outputDiff[index];
						inputDiff[b * inputObjectSize + maxIndex + c] += diff;
					}
				}
			}
		}
	}
}

static C3dPoolingTestParams getParams( const CTestParams& params, CRandom& random )
{
	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval inputDepthInterval = params.GetInterval( "InputDepth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval filterHeightInterval = params.GetInterval( "FilterHeight" );
	const CInterval filterWidthInterval = params.GetInterval( "FilterWidth" );
	const CInterval filterDepthInterval = params.GetInterval( "FilterDepth" );
	const CInterval strideHeightInterval = params.GetInterval( "StrideHeight" );
	const CInterval strideWidthInterval = params.GetInterval( "StrideWidth" );
	const CInterval strideDepthInterval = params.GetInterval( "StrideDepth" );

	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );
	const int inputDepth = random.UniformInt( inputDepthInterval.Begin, inputDepthInterval.End );

	const int inputChannels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );

	const int filterHeight = random.UniformInt( filterHeightInterval.Begin, filterHeightInterval.End );
	const int filterWidth = random.UniformInt( filterWidthInterval.Begin, filterWidthInterval.End );
	const int filterDepth = random.UniformInt( filterDepthInterval.Begin, filterDepthInterval.End );

	const int strideHeight = random.UniformInt( strideHeightInterval.Begin, strideHeightInterval.End );
	const int strideWidth = random.UniformInt( strideWidthInterval.Begin, strideWidthInterval.End );
	const int strideDepth = random.UniformInt( strideDepthInterval.Begin, strideDepthInterval.End );

	return C3dPoolingTestParams( batchSize, inputHeight, inputWidth, inputDepth, inputChannels,
		filterHeight, filterWidth, filterDepth, strideHeight, strideWidth, strideDepth );
}

static void blob3dMaxPoolingBackwardTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	auto poolingParams = getParams( params, random );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int outHeight = calcConvDimSize( poolingParams.InputHeight, poolingParams.FilterHeight, poolingParams.StrideHeight );
	const int outWidth = calcConvDimSize( poolingParams.InputWidth, poolingParams.FilterWidth, poolingParams.StrideWidth );
	const int outDepth = calcConvDimSize( poolingParams.InputDepth, poolingParams.FilterDepth, poolingParams.StrideDepth );

	CREATE_FILL_FLOAT_ARRAY( outputDiffData, valuesInterval.Begin, valuesInterval.End, poolingParams.InputCount * outHeight * outWidth * outDepth * poolingParams.InputChannels, random )
	CFloatBlob outputDiffBlob( MathEngine(), poolingParams.InputCount, outHeight, outWidth, outDepth, poolingParams.InputChannels );
	outputDiffBlob.CopyFrom( outputDiffData.data() );
	CFloatBlob outputBlob( MathEngine(), poolingParams.InputCount, outHeight, outWidth, outDepth, poolingParams.InputChannels );

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, poolingParams.InputCount * poolingParams.InputHeight * poolingParams.InputWidth
		* poolingParams.InputDepth * poolingParams.InputChannels, random )
	CFloatBlob inputDataBlob( MathEngine(), poolingParams.InputCount, poolingParams.InputHeight,
		poolingParams.InputWidth, poolingParams.InputDepth, poolingParams.InputChannels );
	inputDataBlob.CopyFrom( inputData.data() );

	CFloatBlob inputDiffBlob( MathEngine(), poolingParams.InputCount, poolingParams.InputHeight,
		poolingParams.InputWidth, poolingParams.InputDepth, poolingParams.InputChannels );

	CIntBlob indexBlob( MathEngine(), poolingParams.InputCount, outHeight, outWidth, outDepth, poolingParams.InputChannels );
	CIntHandle indexBlobPtr = indexBlob.GetData();

	auto poolingDesc = MathEngine().Init3dMaxPooling( inputDataBlob.GetDesc(), poolingParams.FilterHeight, poolingParams.FilterWidth, poolingParams.FilterDepth,
		poolingParams.StrideHeight, poolingParams.StrideWidth, poolingParams.StrideDepth, outputDiffBlob.GetDesc() );
	MathEngine().Blob3dMaxPooling( *poolingDesc, inputDataBlob.GetData(), &indexBlobPtr, outputBlob.GetData()  );

	MathEngine().Blob3dMaxPoolingBackward( *poolingDesc, outputDiffBlob.GetData(), indexBlobPtr, inputDiffBlob.GetData() );
	delete poolingDesc;

	std::vector<int> maxIndices;
	maxIndices.resize( poolingParams.InputCount * outHeight * outWidth * outDepth * poolingParams.InputChannels );
	indexBlob.CopyTo( maxIndices.data() );

	const int inputDiffSize = poolingParams.InputCount * poolingParams.InputHeight * poolingParams.InputWidth * poolingParams.InputDepth * poolingParams.InputChannels;
	std::vector<float> actualDiff, expectedDiff;
	actualDiff.resize( inputDiffSize );
	inputDiffBlob.CopyTo( actualDiff.data() );
	expectedDiff.insert( expectedDiff.begin(), inputDiffSize, 0 );

	max3dPoolingBackwardNaive( poolingParams, outputDiffData.data(), maxIndices.data(), expectedDiff.data() );

	for( int i = 0; i < inputDiffSize; ++i ) {
		ASSERT_NEAR( expectedDiff[i], actualDiff[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBlob3dMaxPoolingBackwardTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineBlob3dMaxPoolingBackwardTestInstantiation, CMathEngineBlob3dMaxPoolingBackwardTest,
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

TEST_P( CMathEngineBlob3dMaxPoolingBackwardTest, Random )
{
	RUN_TEST_IMPL( blob3dMaxPoolingBackwardTestImpl )
}
