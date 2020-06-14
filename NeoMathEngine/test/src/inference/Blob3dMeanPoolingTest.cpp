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

static void meanPooling3dNaive( int batchSize, int channels, int inputHeight, int inputWidth, int inputDepth, 
	int filterHeight, int filterWidth, int filterDepth, int strideHeight, int strideWidth, int strideDepth, 
	const float *source, float *result ) 
{
	const int outHeight = ( inputHeight - filterHeight ) / strideHeight + 1;
	const int outWidth = ( inputWidth - filterWidth ) / strideWidth + 1;
	const int outDepth = ( inputDepth - filterDepth ) / strideDepth + 1;

	int sourceDepthSize = inputDepth * channels;
	int sourceRowSize = inputWidth * sourceDepthSize;
	int sourceObjectSize = inputHeight * sourceRowSize;

	int resultDepthSize = outDepth * channels;
	int resultRowSize = outWidth * resultDepthSize;

	int filterGeom = filterDepth * filterHeight * filterWidth;

	const float* sourceObject = source;
	float* resultJStart = result;

	for(int b = 0; b < batchSize; ++b) {
		for(int j = 0; j < outHeight; ++j) {
			int sourceJIndex = j * strideHeight * sourceRowSize;
			for(int filterJ = 0; filterJ < filterHeight; ++filterJ) {
				float* resultIStart = resultJStart;

				for(int i = 0; i < outWidth; ++i) {
					int sourceIIndex = sourceJIndex + i * strideWidth * sourceDepthSize;
					for(int filterI = 0; filterI < filterWidth; ++filterI) {
						float* resultData = resultIStart;

						for(int k = 0; k < outDepth; ++k) {
							int sourceIndex = sourceIIndex + k * strideDepth * channels;
							for(int filterK = 0; filterK < filterDepth; ++filterK) {

								bool isFirstItem = (filterJ == 0) && (filterI == 0) && (filterK == 0);
								const float* sourceData = sourceObject + sourceIndex;
								float* resultDataChannel = resultData;
								for( int c = 0; c < channels; c++ ) {
									if( isFirstItem ) {
										*resultDataChannel = *sourceData / filterGeom;
									} else {
										*resultDataChannel += *sourceData / filterGeom;
									}
									resultDataChannel++;
									sourceData++;
								}

								sourceIndex += channels;
							}
							resultData += channels;
						}
						sourceIIndex += sourceDepthSize;
					}
					resultIStart += resultDepthSize;
				}
				sourceJIndex += sourceRowSize;
			}
			resultJStart += resultRowSize;
		}

		sourceObject += sourceObjectSize;
	}
}

static void test3dMeanPoolingImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

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
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );
	const int inputDepth = random.UniformInt( inputDepthInterval.Begin, inputDepthInterval.End );

	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	
	const int filterHeight = random.UniformInt( filterHeightInterval.Begin, filterHeightInterval.End );
	const int filterWidth = random.UniformInt( filterWidthInterval.Begin, filterWidthInterval.End );
	const int filterDepth = random.UniformInt( filterDepthInterval.Begin, filterDepthInterval.End );
	
	const int strideHeight = random.UniformInt( strideHeightInterval.Begin, strideHeightInterval.End );
	const int strideWidth = random.UniformInt( strideWidthInterval.Begin, strideWidthInterval.End );
	const int strideDepth = random.UniformInt( strideDepthInterval.Begin, strideDepthInterval.End );
	
	const int geometrySize = inputHeight * inputWidth * inputDepth;
	const int blobSize = batchSize * geometrySize * channels;

	const int outHeight = ( inputHeight - filterHeight ) / strideHeight + 1;
	const int outWidth = ( inputWidth - filterWidth ) / strideWidth + 1;
	const int outDepth = ( inputDepth - filterDepth ) / strideDepth + 1;

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, blobSize, random )
	CFloatBlob inputBlob( MathEngine(), batchSize, inputHeight, inputWidth, inputDepth, channels );
	inputBlob.CopyFrom( inputData.data() );

	std::vector<float> expectedData;
	expectedData.resize( batchSize * outHeight * outWidth * outDepth * channels );

	meanPooling3dNaive( batchSize, channels, 
		inputHeight, inputWidth, inputDepth, filterHeight, filterWidth, filterDepth, strideHeight, strideWidth, strideDepth, 
		inputData.data(), expectedData.data() );
	
	CFloatBlob outBlob( MathEngine(), batchSize, outHeight, outWidth, outDepth, channels );

	C3dMeanPoolingDesc *desc = MathEngine().Init3dMeanPooling(inputBlob.GetDesc(), filterHeight, filterWidth, filterDepth, strideHeight, strideWidth, strideDepth, outBlob.GetDesc());
	MathEngine().Blob3dMeanPooling(*desc, inputBlob.GetData(), outBlob.GetData());
	delete desc;

	std::vector<float> resultData;
	resultData.resize( batchSize * outHeight * outWidth * outDepth * channels );
	outBlob.CopyTo(resultData.data());
	std::vector<int> maxIndicesData;

	for( int i = 0; i < batchSize * outHeight * outWidth * outDepth * channels; i++ ) {
		ASSERT_NEAR( expectedData[i], resultData[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngine3dMeanPoolingTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngine3dMeanPoolingTestInstantiation, CMathEngine3dMeanPoolingTest,
	::testing::Values(
		CTestParams(
			"InputHeight = (10..100);"
			"InputWidth = (10..100);"
			"InputDepth = (10..50);"
			"Channels = (3..10);"
			"BatchSize = (1..5);"
			"FilterHeight = (1..7);"
			"FilterWidth = (1..7);"
			"FilterDepth = (1..7);"
			"StrideHeight = (1..7);"
			"StrideWidth = (1..7);"
			"StrideDepth = (1..7);"
			"Values = (-50..50);"
			"TestCount = 10;"
		)
	)
);

TEST_P(CMathEngine3dMeanPoolingTest, Random)
{
	RUN_TEST_IMPL(test3dMeanPoolingImpl)
}
