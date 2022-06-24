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

static void generateRleBlob( int batchCount, int height, int width, float stroke, float nonStroke, CRandom& random, std::vector<float>& resultRle, std::vector<float>& resultNative )
{
	resultRle.insert( resultRle.begin(), batchCount * height * width, 0 );
	resultNative.resize( batchCount * height * width );
	float* imageStart = resultRle.data();

	for( int b = 0; b < batchCount; b++ ) {
		CRleImage* rleImage = reinterpret_cast<CRleImage*>( imageStart );

		rleImage->Height = height;
		rleImage->Width = width;
		int i = 0;
		int pos = 0;
		while( i < height * width ) {
			CRleStroke cur = Sentinel;
			for( int j = 0; j < width; j++ ) {
				const int value = random.UniformInt( 0, 2 );
				resultNative[b * height * width + i + j] = value != 0 ? stroke : nonStroke;
				if( value != 0 ) {
					if( cur.Start == Sentinel.Start ) {
						cur.Start = ( short )( j );
					}
					cur.End = ( short )( j + 1 );
				}

				if( value == 0 || width - 1 == j ) {
					if( cur.Start != Sentinel.Start ) {
						rleImage->Lines[pos] = cur;
						pos++;
						cur = Sentinel;
					}
				}
			}
			rleImage->Lines[pos] = cur;
			pos++;
			i += width;
		}
		rleImage->StrokesCount = pos;
		imageStart += height * width;
	}
}

static void blobConvolutionLearnAddNaive( int batchSize, int inputHeight, int inputWidth, int filterCount, int filterHeight, int filterWidth,
	int dilationHeight, int dilationWidth, int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
	float* inputData, float* filterDiffData, float* freeTermData, float* outputDiffData )
{
	const int outputHeight = calcConvOutputSize( inputHeight, paddingHeight, filterHeight, dilationHeight, strideHeight );
	const int outputWidth = calcConvOutputSize( inputWidth, paddingWidth, filterWidth, dilationWidth, strideWidth );

	for( int b = 0; b < batchSize; b++ ) {
		for( int x = 0; x < outputWidth; x++ ) {
			for( int y = 0; y < outputHeight; y++ ) {
				for( int c = 0; c < filterCount; c++ ) {
					int inputX = -paddingWidth + strideWidth * x;
					int inputY = -paddingHeight + strideHeight * y;
					float* filterDataPtr = filterDiffData + c * filterHeight * filterWidth;
					float outputDiff = outputDiffData[y * outputWidth * filterCount + x * filterCount + c];
					freeTermData[c] += outputDiff;
					for( int fy = 0; fy < filterHeight; fy++ ) {
						for( int fx = 0; fx < filterWidth; fx++ ) {
							int ix = inputX + fx * dilationWidth;
							int iy = inputY + fy * dilationHeight;
							if( 0 <= ix && ix < inputWidth && 0 <= iy && iy < inputHeight ) {
								filterDataPtr[fy * filterWidth + fx] += inputData[iy * inputWidth + ix] * outputDiff;
							}
						}
					}
				}
			}
		}
		inputData += inputHeight * inputWidth;
		outputDiffData += outputWidth * outputHeight * filterCount;
	}
}

static void blobRleConvolutionLearnAddImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );

	const CInterval strideHeightInterval = params.GetInterval( "StrideHeight" );
	const CInterval strideWidthInterval = params.GetInterval( "StrideWidth" );

	const CInterval filterCountInterval = params.GetInterval( "FilterCount" );
	const CInterval filterHeightInterval = params.GetInterval( "FilterHeight" );
	const CInterval filterWidthInterval = params.GetInterval( "FilterWidth" );

	const float stroke = static_cast<float>( params.GetValue<double>( "StrokeValue" ) );
	const float nonStroke = static_cast<float>( params.GetValue<double>( "NonStrokeValue" ) );

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );

	const int strideHeight = random.UniformInt( strideHeightInterval.Begin, strideHeightInterval.End );
	const int strideWidth = random.UniformInt( strideWidthInterval.Begin, strideWidthInterval.End );

	const int filterCount = random.UniformInt( filterCountInterval.Begin, filterCountInterval.End ) * 4;
	const int filterHeight = random.UniformInt( filterHeightInterval.Begin, filterHeightInterval.End );
	const int filterWidth = random.UniformInt( filterWidthInterval.Begin, filterWidthInterval.End );

	const int outputHeight = calcConvOutputSize( inputHeight, 0, filterHeight, 1, strideHeight );
	const int outputWidth = calcConvOutputSize( inputWidth, 0, filterWidth, 1, strideWidth );

	const bool isZeroFreeTerm = random.UniformInt( 0, 2 ) != 0;

	std::vector<float> inputDataNative;
	std::vector<float> inputDataRle;
	generateRleBlob( batchSize, inputHeight, inputWidth, stroke, nonStroke, random, inputDataRle, inputDataNative );

	CFloatBlob inputBlobRle( MathEngine(), batchSize, inputHeight, inputWidth, 1 );
	inputBlobRle.CopyFrom( ( float* )inputDataRle.data() );

	CFloatBlob inputBlobNative( MathEngine(), batchSize, inputHeight, inputWidth, 1 );
	inputBlobNative.CopyFrom( inputDataNative.data() );
	
	CREATE_FILL_FLOAT_ARRAY( filterData, -1, 1, filterCount * filterHeight * filterWidth, random );
	CFloatBlob filterBlob( MathEngine(), filterCount, filterHeight, filterWidth, 1, 1 );
	filterBlob.CopyFrom( filterData.data() );

	CREATE_FILL_FLOAT_ARRAY( outputDiffData, -1, 1, batchSize * outputHeight * outputWidth * filterCount, random );
	CFloatBlob outDiffBlob( MathEngine(), batchSize, outputHeight, outputWidth, 1, filterCount );
	outDiffBlob.CopyFrom( outputDiffData.data() );

	CREATE_FILL_FLOAT_ARRAY( freeTermData, -1, 1, filterCount, random );
	CFloatBlob freeTermBlob( MathEngine(), 1, 1, 1, filterCount );
	freeTermBlob.CopyFrom( freeTermData.data() );
	CFloatHandle ft = freeTermBlob.GetData();

	blobConvolutionLearnAddNaive( batchSize, inputHeight, inputWidth, filterCount, filterHeight, filterWidth, 1, 1, 0, 0,
		strideHeight, strideWidth, inputDataNative.data(), filterData.data(), freeTermData.data(), outputDiffData.data() );

	CRleConvolutionDesc* rleConvDesc = MathEngine().InitBlobRleConvolution( inputBlobRle.GetDesc(), stroke, nonStroke,
		strideHeight, strideWidth, filterBlob.GetDesc(), outDiffBlob.GetDesc() );
	MathEngine().BlobRleConvolutionLearnAdd( *rleConvDesc, inputBlobRle.GetData(), outDiffBlob.GetData(),
		filterBlob.GetData(), isZeroFreeTerm ? 0 : &ft );
	delete rleConvDesc;

	std::vector<float> getFilterData, getFreeTermData;
	getFilterData.resize( filterCount * filterHeight * filterWidth );
	getFreeTermData.resize( filterCount );
	filterBlob.CopyTo( getFilterData.data() );
	freeTermBlob.CopyTo( getFreeTermData.data()  );

	for( size_t i = 0; i < getFilterData.size(); i++ ) {
		ASSERT_NEAR( getFilterData[i], filterData[i], 0.05 );
	}
	if( !isZeroFreeTerm ) {
		for( size_t i = 0; i < freeTermData.size(); i++ ) {
			ASSERT_NEAR( getFreeTermData[i], freeTermData[i], 0.05 );
		}
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMathEngineBlobRleConvolutionLearnAddTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineBlobRleConvolutionLearnAddTestInstantiation, CMathEngineBlobRleConvolutionLearnAddTest,
	::testing::Values(
		CTestParams(
			"BatchSize = (1..5);"
			"InputHeight = (15..64);"
			"InputWidth = (15..64);"
			"FilterCount = (1..3);"
			"FilterHeight = (1..5);"
			"FilterWidth = (1..5);"
			"StrideHeight = (1..2);"
			"StrideWidth = (1..2);"
			"StrokeValue = 1;"
			"NonStrokeValue = 0.5;"
			"TestCount = 100;"
		)
	)
);

TEST_P( CMathEngineBlobRleConvolutionLearnAddTest, Random )
{
	RUN_TEST_IMPL( blobRleConvolutionLearnAddImpl )
}
