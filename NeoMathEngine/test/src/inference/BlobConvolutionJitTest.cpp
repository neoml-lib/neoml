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

static void blobConvolutionJitTestImpl( const CTestParams& params, int seed )
{
    CRandom random( seed );

    std::vector<int> convParams;
    params.GetArray( "MainParams", convParams );
    ASSERT_EQ( convParams.size(), 12 );
    const CInterval channelINterval = params.GetInterval( "ChCount" );

    // FC    FW  FH  DW  DH  SW  SH  PW  PH SrcW SrcH FT
    const int filterCount = convParams[0];
    const int filterWidth = convParams[1];
    const int filterHeight = convParams[2];
    const int dilationWidth = convParams[3];
    const int dilationHeight = convParams[4];
    const int strideWidth = convParams[5];
    const int strideHeight = convParams[6];
    const int paddingWidth = convParams[7];
    const int paddingHeight = convParams[8];
    const int inputWidth = convParams[9];
    const int inputHeight = convParams[10];
    const bool isZeroFreeTerm = convParams[11] == 0 ? false : true;
    
    const int inputLength = 1;
    const int inputBatch = 1;
    const int inputDepth = 1;
    const int outputHeight = calcConvOutputSize( inputHeight, paddingHeight, filterHeight, dilationHeight, strideHeight );
    const int outputWidth = calcConvOutputSize( inputWidth, paddingWidth, filterWidth, dilationWidth, strideWidth );

    for( int channelCount = channelINterval.Begin; channelCount <= channelINterval.End; channelCount++ ) {
        const CInterval valuesInterval = { -10, 10 };

        CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End,
            inputLength * inputBatch * inputHeight * inputWidth * inputDepth * channelCount, random )
            CFloatBlob inputBlob( MathEngine(), inputLength, inputBatch, 1, inputHeight, inputWidth, inputDepth, channelCount );
        inputBlob.CopyFrom( inputData.data() );

        CREATE_FILL_FLOAT_ARRAY( filterData, valuesInterval.Begin, valuesInterval.End,
            filterCount * filterHeight * filterWidth * inputDepth * channelCount, random )
            CFloatBlob filterBlob( MathEngine(), filterCount, filterHeight, filterWidth, inputDepth, channelCount );
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

        MathEngine().BlobConvolution( *convDesc, inputBlob.GetData(), filterBlob.GetData(),
            isZeroFreeTerm ? 0 : &freeTermDataPtr, outputBlob.GetData() );
        outputBlob.CopyTo( actualData.data() );

        delete convDesc;

        std::vector<float> expectedData( outputSize );

        batchConvolutionForward( inputData.data(), filterData.data(), freeTermData.data(), expectedData.data(),
            inputLength, inputBatch, inputHeight, inputWidth, inputDepth, channelCount,
            paddingHeight, paddingWidth, filterCount, filterHeight, filterWidth,
            dilationHeight, dilationWidth, strideHeight, strideWidth );

        for( int i = 0; i < outputSize; ++i ) {
            
            bool res = FloatEq( expectedData[i], actualData[i], 1e-5 );
            if( !res ) {
                GTEST_LOG_( ERROR ) << "\n                FC  FW  FH  DW  DH  SW  SH  PW  PH SrcW SrcH FT\n" <<
                    "ConvParams: " << params.GetStrValue( "MainParams" ) << std::endl <<
                    "Channel count: " << channelCount;
            }
            ASSERT_TRUE( res );
        }
    }
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBlobConvolutionJitTest : public CTestFixtureWithParams {
public:
	void SetUp() override { MathEngine().CleanUp(); }
};

CTestParams JitTestParams[] = {
    // FC = 3
    // Kernels: wide - 1x24; narrow - 3x8
    // channelCount: batch - 1; single - 4
    //                            FC  FW  FH  DW  DH  SW  SH  PW  PH SrcW SrcH FT
    CTestParams( "MainParams = {   3,  3,  3,  1,  1,  1,  1,  1,  1,  26,   3, 1 }; ChCount = (2..2); TestCount = 1;" ),
    CTestParams( "MainParams = {   3,  3,  3,  1,  1,  1,  1,  1,  1,  10,   5, 1 }; ChCount = (2..2); TestCount = 1;" ),
    CTestParams( "MainParams = {   3,  3,  3,  1,  1,  1,  1,  1,  1,   3,   3, 1 }; ChCount = (1..9); TestCount = 1;" ),
    CTestParams( "MainParams = {   3,  3,  3,  1,  1,  1,  1,  1,  1,   3,   5, 1 }; ChCount = (1..9); TestCount = 1;" ),
    CTestParams( "MainParams = {   3,  9,  9,  1,  1,  1,  1,  4,  1,  97,  37, 1 }; ChCount = (1..9); TestCount = 1;" ),
    // FC = 6
    // Kernels: wide - 1x12; narrow - 3x4
    // channelCount: batch - 1; single - 4
    //                            FC  FW  FH  DW  DH  SW  SH  PW  PH SrcW SrcH FT
    CTestParams( "MainParams = {   6,  3,  3,  1,  1,  1,  1,  1,  1,  14,   3, 1 }; ChCount = (2..2); TestCount = 1;" ),
    CTestParams( "MainParams = {   6,  3,  3,  1,  1,  1,  1,  1,  1,   6,   5, 1 }; ChCount = (2..2); TestCount = 1;" ),
    CTestParams( "MainParams = {   6,  3,  3,  1,  1,  1,  1,  1,  1,   3,   3, 1 }; ChCount = (1..9); TestCount = 1;" ),
    CTestParams( "MainParams = {   6,  3,  3,  1,  1,  1,  1,  1,  1,   3,   5, 1 }; ChCount = (1..9); TestCount = 1;" ),
    CTestParams( "MainParams = {   6,  9,  9,  1,  1,  1,  1,  4,  1,  97,  37, 1 }; ChCount = (1..9); TestCount = 1;" ),
    // FC = 8
    // Kernels: wide - 1x7
    // channelCount: batch - 16; single - 15
    //                            FC  FW  FH  DW  DH  SW  SH  PW  PH SrcW SrcH FT
    CTestParams( "MainParams = {   8,  3,  3,  1,  1,  1,  1,  1,  1,   9,   3, 1 }; ChCount = (1..17); TestCount = 1;" ),
    CTestParams( "MainParams = {   8,  3,  3,  1,  1,  1,  1,  1,  1,   3,   3, 1 }; ChCount = (1..29); TestCount = 1;" ),
    CTestParams( "MainParams = {   8,  9,  9,  1,  1,  1,  1,  4,  1,  97,  37, 1 }; ChCount = (1..17); TestCount = 1;" ),
    // FC = 16
    // Kernels: wide - 1x5
    // channelCount: batch - 1; single - 4
    //                            FC  FW  FH  DW  DH  SW  SH  PW  PH SrcW SrcH FT
    CTestParams( "MainParams = {  16,  3,  3,  1,  1,  1,  1,  1,  1,   7,   3, 1 }; ChCount = (1..2); TestCount = 1;" ),
    CTestParams( "MainParams = {  16,  3,  3,  1,  1,  1,  1,  1,  1,   3,   3, 1 }; ChCount = (1..5); TestCount = 1;" ),
    CTestParams( "MainParams = {  16,  9,  9,  1,  1,  1,  1,  4,  1,  97,  37, 1 }; ChCount = (1..5); TestCount = 1;" ),
    // FC = 18
    // Kernels: wide - 1x4
    // channelCount: batch - 1; single - 4
    //                            FC  FW  FH  DW  DH  SW  SH  PW  PH SrcW SrcH FT
    CTestParams( "MainParams = {  18,  3,  3,  1,  1,  1,  1,  1,  1,   6,   3, 1 }; ChCount = (1..2); TestCount = 1;" ),
    CTestParams( "MainParams = {  18,  3,  3,  1,  1,  1,  1,  1,  1,   3,   3, 1 }; ChCount = (1..9); TestCount = 1;" ),
    CTestParams( "MainParams = {  18,  9,  9,  1,  1,  1,  1,  4,  1,  97,  37, 1 }; ChCount = (1..5); TestCount = 1;" ),
    // FC = 24
    // Kernels: wide - 1x3
    // channelCount: batch - 24; single - 8
    //                            FC  FW  FH  DW  DH  SW  SH  PW  PH SrcW SrcH FT
    CTestParams( "MainParams = {  24,  3,  3,  1,  1,  1,  1,  1,  1,   5,   3, 1 }; ChCount = (1..25); TestCount = 1;" ),
    CTestParams( "MainParams = {  24,  3,  3,  1,  1,  1,  1,  1,  1,   3,   3, 1 }; ChCount = (1..15); TestCount = 1;" ),
    CTestParams( "MainParams = {  24,  9,  9,  1,  1,  1,  1,  4,  1,  97,  37, 1 }; ChCount = (1..25); TestCount = 1;" ),
    // FC = 32
    // Kernels: wide - 1x2
    // channelCount: batch - 16; single - 8
    //                            FC  FW  FH  DW  DH  SW  SH  PW  PH SrcW SrcH FT
    CTestParams( "MainParams = {  32,  3,  3,  1,  1,  1,  1,  1,  1,   4,   3, 1 }; ChCount = (1..17); TestCount = 1;" ),
    CTestParams( "MainParams = {  32,  3,  3,  1,  1,  1,  1,  1,  1,   3,   3, 1 }; ChCount = (1..15); TestCount = 1;" ),
    CTestParams( "MainParams = {  32,  9,  9,  1,  1,  1,  1,  4,  1,  97,  37, 1 }; ChCount = (1..17); TestCount = 1;" ),
    // Huge JIT
    CTestParams( "MainParams = {  24, 13, 19,  2,  4,  5,  3,  1,  1, 311, 313, 1 }; ChCount = (25..25); TestCount = 1;" ),
    // Test fillPixelOffset() function
    //                            FC  FW  FH  DW  DH  SW  SH  PW  PH SrcW SrcH FT
    // Test stride/dilation
    CTestParams( "MainParams = {  18,  9,  3,  1,  1,  1,  1,  1,  1,  27,   3, 1 }; ChCount = (7..7); TestCount = 1;" ),
    CTestParams( "MainParams = {  18,  9,  3,  2,  1,  2,  1,  1,  1,  27,   3, 1 }; ChCount = (7..7); TestCount = 1;" ),
    CTestParams( "MainParams = {  18,  9,  3,  3,  1,  3,  1,  1,  1,  27,   3, 1 }; ChCount = (7..7); TestCount = 1;" ),
    CTestParams( "MainParams = {  18,  9,  3,  3,  1,  4,  1,  1,  1,  27,   3, 1 }; ChCount = (7..7); TestCount = 1;" ),
    CTestParams( "MainParams = {  18,  9,  3,  2,  1,  5,  1,  1,  1,  27,   3, 1 }; ChCount = (7..7); TestCount = 1;" ),
    CTestParams( "MainParams = {  18,  9,  3,  1,  1,  6,  1,  1,  1,  27,   3, 1 }; ChCount = (7..7); TestCount = 1;" ),
    // Test padding
    CTestParams( "MainParams = {  18,  9,  3,  3,  1,  3,  1,  2,  1,  27,   3, 1 }; ChCount = (7..7); TestCount = 1;" ),
    CTestParams( "MainParams = {  18,  9,  3,  3,  1,  3,  1,  3,  1,  27,   3, 1 }; ChCount = (7..7); TestCount = 1;" ),
    CTestParams( "MainParams = {  18,  9,  3,  3,  1,  3,  1,  4,  1,  27,   3, 1 }; ChCount = (7..7); TestCount = 1;" ),
    CTestParams( "MainParams = {  18,  9,  3,  3,  1,  3,  1,  5,  1,  27,   3, 1 }; ChCount = (7..7); TestCount = 1;" )
};

INSTANTIATE_TEST_CASE_P( CMathEngineBlobConvolutionJitTestInstantiation, CMathEngineBlobConvolutionJitTest,
    ::testing::ValuesIn( JitTestParams )
);

TEST_P( CMathEngineBlobConvolutionJitTest, Random )
{
    RUN_TEST_IMPL( blobConvolutionJitTestImpl );
}
