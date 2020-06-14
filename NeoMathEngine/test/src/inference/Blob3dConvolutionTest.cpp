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

static void blob3dConvolutionNaiveWorker( int inputChannels, int inputHeight, int inputWidth, int inputDepth, 
	int filterCount, int filterHeight, int filterWidth, int filterDepth, 
	int paddingHeight, int paddingWidth, int paddingDepth,
	int strideHeight, int strideWidth, int strideDepth, 
	const float *input, const float *filter, const float *freeTerm, float *result,
	int b, int y, int x, int z, int c ) 
{
    const int outHeight = 1 + ( inputHeight + 2 * paddingHeight - filterHeight ) / strideHeight;
    const int outWidth = 1 + ( inputWidth + 2 * paddingWidth - filterWidth ) / strideWidth;
    const int outDepth = 1 + ( inputDepth + 2 * paddingDepth - filterDepth ) / strideDepth;
	const int outChannels = filterCount;

    input += b * inputChannels * inputDepth * inputHeight * inputWidth;
    result += b * outChannels * outDepth * outHeight * outWidth + y * outWidth * outDepth * outChannels + x * outDepth * outChannels + z * outChannels + c;
    filter += c * inputChannels * filterDepth * filterWidth * filterHeight;

    const int inputDStart = z * strideDepth - paddingDepth;
    const int inputDEnd = inputDStart + filterDepth;
    const int inputHStart = y * strideHeight - paddingHeight;
    const int inputHEnd = inputHStart + filterHeight;
    const int inputWStart = x * strideWidth - paddingWidth;
    const int inputWEnd = inputWStart + filterWidth;

    float resultVal = freeTerm != 0 ? freeTerm[c] : 0.0f;
    for( int k = 0; k < inputChannels; k++) {
        int filterY = 0;
        for( int j = inputHStart; j < inputHEnd; j++ ) {
            if( j < 0 || j >= inputHeight ) {
                filterY++;
                continue;
            }
            int filterX = 0;
            int sourceHOffset = j * inputWidth * inputDepth * inputChannels;
            int filterHOffset = filterY * filterWidth * filterDepth * inputChannels;

            for( int i = inputWStart; i < inputWEnd; i++ ) {
                if( i < 0 || i >= inputWidth ) {
                    filterX++;
                    continue;
                }
                int filterZ = 0;
                int sourceWOffset = sourceHOffset + i * inputDepth * inputChannels;
                int filterWOffset = filterHOffset + filterX * filterDepth * inputChannels;
                for( int l = inputDStart; l < inputDEnd; l++ ) {
                    if( l < 0 || l >= inputDepth ) {
                        filterZ++;
                        continue;
                    }
                    const float srcVal = input[sourceWOffset + l * inputChannels];
                    const float fltVal = filter[filterWOffset + filterZ * inputChannels];
                    resultVal = fma(srcVal, fltVal, resultVal );
                    filterZ++;
                }
                filterX++;
            }
            filterY++;
        }
        input++;
        filter++;
    }

    *result = resultVal;
}

static void blob3dConvolutionNaive( int inputCount, int inputChannels, int inputHeight, int inputWidth, int inputDepth, 
	int filterCount, int filterHeight, int filterWidth, int filterDepth, 
	int paddingHeight, int paddingWidth, int paddingDepth,
	int strideHeight, int strideWidth, int strideDepth, 
	const float *input, const float *filter, const float *freeTerm, float *result)
{
    const int outHeight = 1 + ( inputHeight + 2 * paddingHeight - filterHeight ) / strideHeight;
    const int outWidth = 1 + ( inputWidth + 2 * paddingWidth - filterWidth ) / strideWidth;
    const int outDepth = 1 + ( inputDepth + 2 * paddingDepth - filterDepth ) / strideDepth;
	const int outChannels = filterCount;

	for( int b = 0; b < inputCount; b++ ) {
		for( int y = 0; y < outHeight; y++ ) {
			for( int x = 0; x < outWidth; x++ ) {
				for( int z = 0; z < outDepth; z++ ) {
					for( int c = 0; c < outChannels; c++ ) {
						blob3dConvolutionNaiveWorker( inputChannels, inputHeight, inputWidth, inputDepth,
							filterCount, filterHeight, filterWidth, filterDepth,
							paddingHeight, paddingWidth, paddingDepth,
							strideHeight, strideWidth, strideDepth, 
							input, filter, freeTerm, result, b, y, x, z, c);
					}
				}
			}
		}
	}
}

static void blob3dConvolutionImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

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
	const CInterval valuesInterval = params.GetInterval( "Values" );

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
    
    const bool addFreeTerm = random.UniformInt( 0, 1 ) == 1 ? true : false;

	const int geometrySize = inputHeight * inputWidth * inputDepth;
	const int blobSize = batchSize * geometrySize * inputChannels;

	const int outHeight = 1 + ( inputHeight + 2 * paddingHeight - filterHeight ) / strideHeight;
	const int outWidth = 1 + ( inputWidth + 2 * paddingWidth - filterWidth ) / strideWidth;
	const int outDepth = 1 + ( inputDepth + 2 * paddingDepth - filterDepth ) / strideDepth;

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, blobSize, random )
	CFloatBlob inputBlob( MathEngine(), batchSize, inputHeight, inputWidth, inputDepth, inputChannels );
	inputBlob.CopyFrom( inputData.data() );

	CREATE_FILL_FLOAT_ARRAY( filterData, valuesInterval.Begin, valuesInterval.End, filterCount * inputChannels * filterHeight * filterWidth * filterDepth, random )
	CFloatBlob filterBlob( MathEngine(), filterCount, filterHeight, filterWidth, filterDepth, inputChannels );
	filterBlob.CopyFrom( filterData.data() );

    std::vector<float> freeTermData;
    freeTermData.resize( filterCount );
    for( size_t i = 0; i < freeTermData.size(); i++ ) {
        freeTermData[i] = static_cast<float>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) );
    }
    CFloatBlob freeTermBlob( MathEngine(), CT_Float, 1, 1, 1, filterCount );
    freeTermBlob.CopyFrom( freeTermData.data() );

	std::vector<float> expectedData;
	expectedData.resize( batchSize * outHeight * outWidth * outDepth * filterCount );

	blob3dConvolutionNaive( batchSize, inputChannels, inputHeight, inputWidth, inputDepth,
		filterCount, filterHeight, filterWidth, filterDepth, 
		paddingHeight, paddingWidth, paddingDepth,
		strideHeight, strideWidth, strideDepth, 
		inputData.data(), filterData.data(), addFreeTerm ? freeTermData.data() : 0, expectedData.data() );
	
	CFloatBlob outBlob( MathEngine(), batchSize, outHeight, outWidth, outDepth, filterCount );

	CFloatHandle ft = freeTermBlob.GetData();

	C3dConvolutionDesc *convDesc = MathEngine().InitBlob3dConvolution( inputBlob.GetDesc(),
		paddingHeight, paddingWidth, paddingDepth, strideHeight, strideWidth, strideDepth, filterBlob.GetDesc(), outBlob.GetDesc() );
	MathEngine().Blob3dConvolution( *convDesc, inputBlob.GetData(), filterBlob.GetData(), addFreeTerm ? &ft : nullptr, outBlob.GetData() );
	delete convDesc;

	std::vector<float> resultData;
	resultData.resize( batchSize * outHeight * outWidth * outDepth * filterCount );
	outBlob.CopyTo(const_cast<float*>(resultData.data()));

	for( size_t i = 0; i < resultData.size(); i++ ) {
		ASSERT_NEAR( expectedData[i], resultData[i], 1e-3f );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBlob3dConvolutionTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineBlob3dConvolutionTestInstantiation, CMathEngineBlob3dConvolutionTest,
	::testing::Values(
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
			"Values = (-5..5);"
			"TestCount = 100;"
		),
        CTestParams(
            "InputHeight = (5..15);"
            "InputWidth = (5..15);"
            "InputDepth = (5..15);"
            "Channels = (1..3);"
            "BatchSize = (1..3);"
            "FilterCount = (1..3);"
            "FilterHeight = (1..1);"
            "FilterWidth = (1..1);"
            "FilterDepth = (1..1);"
            "PaddingHeight = (0..0);"
            "PaddingWidth = (0..0);"
            "PaddingDepth = (0..0);"
            "StrideHeight = (1..2);"
            "StrideWidth = (1..2);"
            "StrideDepth = (1..2);"
			"Values = (-5..5);"
            "TestCount = 10;"
        )
    )
);

TEST_P(CMathEngineBlob3dConvolutionTest, Random )
{
	RUN_TEST_IMPL( blob3dConvolutionImpl )
}