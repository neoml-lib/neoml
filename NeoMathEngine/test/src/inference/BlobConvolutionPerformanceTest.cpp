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

#include <chrono>

using namespace NeoML;
using namespace NeoMLTest;
using namespace std::chrono;

static void blobConvolutionPerformanceTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	std::vector<int> convParams;
	params.GetArray( "ConvParams", convParams );
	ASSERT_EQ( convParams.size(), 13 );

	// PaddingHeight | PaddingWidth | StrideHeight | StrideWidth | DilationHeight | DilationWidth | ObjectHeight | ObjectWidth | NumChannels | ObjectCount | FiltHeight | FiltWidth | IsFreeTerm
	const int paddingHeight = convParams[0];
	const int paddingWidth = convParams[1];
	const int strideHeight = convParams[2];
	const int strideWidth = convParams[3];
	const int dilationHeight = convParams[4];
	const int dilationWidth = convParams[5];
	const int inputHeight = convParams[6];
	const int inputWidth = convParams[7];
	const int inputChannels = convParams[8];
	const int objectCount = convParams[9];
	const int filterHeight = convParams[10];
	const int filterWidth = convParams[11];
	const bool isZeroFreeTerm = convParams[12] == 0 ? false : true;

	const int inputLength = objectCount;
	const int inputBatch = 1;
	const int inputDepth = 1;
	const int filterCount = inputChannels;
	const int outputHeight = calcConvOutputSize( inputHeight, paddingHeight, filterHeight, dilationHeight, strideHeight );
	const int outputWidth = calcConvOutputSize( inputWidth, paddingWidth, filterWidth, dilationWidth, strideWidth );

	const CInterval valuesInterval = { -10, 10 };

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End,
		inputLength * inputBatch * inputHeight * inputWidth * inputDepth * inputChannels, random )
	CFloatBlob inputBlob( MathEngine(), inputLength, inputBatch, 1, inputHeight, inputWidth, inputDepth, inputChannels );
	inputBlob.CopyFrom( inputData.data() );

	CREATE_FILL_FLOAT_ARRAY( filterData, valuesInterval.Begin, valuesInterval.End,
		filterCount * filterHeight * filterWidth * inputDepth * inputChannels, random )
	CFloatBlob filterBlob( MathEngine(), filterCount, filterHeight, filterWidth, inputDepth, inputChannels );
	filterBlob.CopyFrom( filterData.data() );

	CREATE_FILL_FLOAT_ARRAY( freeTermData, valuesInterval.Begin, valuesInterval.End, filterCount, random )
	CFloatBlob freeTermBlob( MathEngine(), 1, 1, 1, filterCount );
	freeTermBlob.CopyFrom( freeTermData.data() );
	if( isZeroFreeTerm ) {
		freeTermData.clear();
		freeTermData.insert( freeTermData.begin(), filterCount, 0 );
	}

	CFloatBlob outputBlob( MathEngine(), inputLength, inputBatch, 1, outputHeight, outputWidth, 1, filterCount );

	CConvolutionDesc* convDesc = MathEngine().InitBlobConvolution( inputBlob.GetDesc(),
		paddingHeight, paddingWidth, strideHeight, strideWidth,
		dilationHeight, dilationWidth, filterBlob.GetDesc(), outputBlob.GetDesc() );

	CConstFloatHandle freeTermDataPtr = freeTermBlob.GetData();
	
	const int outputSize = inputLength * inputBatch * outputHeight * outputWidth * 1 * filterCount;
	std::vector<float> actualData( outputSize );
	
	auto startTime = high_resolution_clock::now();
	
	MathEngine().BlobConvolution( *convDesc, inputBlob.GetData(), filterBlob.GetData(),
		isZeroFreeTerm ? 0 : &freeTermDataPtr, outputBlob.GetData() );
	outputBlob.CopyTo( actualData.data() );
	
	auto stopTime = high_resolution_clock::now();
	
	GTEST_LOG_( INFO ) << "ConvParams: " << params.GetStrValue( "ConvParams" ) << std::endl <<
		"BlobConvolution time: " << std::setprecision(3) << ( stopTime - startTime ).count() / 1e6 << " ms.";

	delete convDesc;
	
	std::vector<float> expectedData( outputSize );

	batchConvolutionForward( inputData.data(), filterData.data(), freeTermData.data(), expectedData.data(),
		inputLength, inputBatch, inputHeight, inputWidth, inputDepth, inputChannels,
		paddingHeight, paddingWidth, filterCount, filterHeight, filterWidth,
		dilationHeight, dilationWidth, strideHeight, strideWidth );

	for( int i = 0; i < outputSize; ++i ) {
		ASSERT_TRUE( FloatEq( expectedData[i], actualData[i] ) );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBlobConvolutionPerformanceTest : public CTestFixtureWithParams {
public:
	void SetUp() override { MathEngine().CleanUp(); }
};
// PaddingHeight | PaddingWidth | StrideHeight | StrideWidth | DilationHeight | DilationWidth | ObjectHeight | ObjectWidth | NumChannels | ObjectCount | FiltHeight | FiltWidth | IsFreeTerm
CTestParams TestParams[] = {
	CTestParams( "ConvParams = { 0, 0, 1, 1, 1, 1, 12, 12, 16, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 0, 0, 1, 1, 1, 1, 25, 25, 1, 1, 5, 5, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 0, 0, 1, 1, 1, 1, 9, 9, 24, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 0, 0, 2, 2, 1, 1, 21, 21, 12, 1, 5, 5, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 0, 0, 2, 2, 1, 1, 25, 25, 1, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 0, 0, 2, 2, 1, 1, 31, 31, 1, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 0, 0, 2, 3, 1, 1, 5, 7, 3, 2, 1, 1, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 0, 0, 2, 3, 1, 1, 5, 7, 3, 2, 3, 5, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 0, 0, 3, 3, 1, 1, 12, 12, 16, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 0, 0, 3, 3, 1, 1, 15, 15, 16, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 128, 128, 128, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 128, 128, 256, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 16, 16, 1024, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 16, 16, 256, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 16, 16, 512, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 16, 32, 50, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 16, 32, 60, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 16, 64, 20, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 256, 256, 128, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 256, 256, 16, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 256, 256, 192, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 256, 256, 216, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 256, 256, 24, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 256, 256, 32, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 256, 256, 64, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 256, 256, 96, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 32, 32, 3, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 32, 32, 512, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 32, 64, 20, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 4, 4, 256, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 4, 64, 60, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 512, 512, 16, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 512, 512, 24, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 512, 512, 3, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 512, 512, 64, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 64, 64, 256, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 64, 64, 512, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 8, 32, 60, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 8, 32, 80, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 8, 64, 50, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 8, 8, 128, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 1, 1, 1, 1, 8, 8, 512, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 2, 2, 1, 1, 1024, 1024, 1, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 2, 2, 1, 1, 1024, 1024, 3, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 2, 2, 1, 1, 224, 224, 3, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 2, 2, 1, 1, 32, 128, 1, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 2, 2, 1, 1, 512, 512, 16, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 2, 2, 1, 1, 512, 512, 24, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 2, 2, 1, 1, 512, 512, 3, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 2, 2, 1, 1, 64, 128, 1, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 1, 1, 2, 2, 1, 1, 768, 512, 3, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 16, 16, 1, 1, 16, 16, 256, 256, 144, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 16, 16, 1, 1, 16, 16, 256, 256, 24, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 16, 16, 1, 1, 16, 16, 256, 256, 64, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 2, 2, 1, 1, 2, 2, 256, 256, 128, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 2, 2, 1, 1, 2, 2, 256, 256, 24, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 2, 2, 1, 1, 2, 2, 256, 256, 64, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 2, 2, 2, 2, 1, 1, 1024, 1024, 3, 1, 5, 5, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 3, 3, 1, 1, 1, 1, 256, 256, 128, 1, 7, 7, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 4, 4, 1, 1, 4, 4, 256, 256, 128, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 4, 4, 1, 1, 4, 4, 256, 256, 24, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 4, 4, 1, 1, 4, 4, 256, 256, 48, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 4, 4, 1, 1, 4, 4, 256, 256, 64, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 8, 8, 1, 1, 8, 8, 256, 256, 16, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 8, 8, 1, 1, 8, 8, 256, 256, 24, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 8, 8, 1, 1, 8, 8, 256, 256, 64, 1, 3, 3, 1 }; TestCount = 1;" ),
	CTestParams( "ConvParams = { 8, 8, 1, 1, 8, 8, 256, 256, 96, 1, 3, 3, 1 }; TestCount = 1;" )
};
// PaddingHeight | PaddingWidth | StrideHeight | StrideWidth | DilationHeight | DilationWidth | ObjectWidth | ObjectHeight | NumChannels | ObjectCount | FiltWidth | FiltHeight | IsFreeTerm
INSTANTIATE_TEST_CASE_P( CMathEngineBlobConvolutionPerformanceTestInstantiation, CMathEngineBlobConvolutionPerformanceTest,
	::testing::ValuesIn( TestParams )
);

TEST_P( CMathEngineBlobConvolutionPerformanceTest, Random )
{
	RUN_TEST_IMPL( blobConvolutionPerformanceTestImpl );
}
