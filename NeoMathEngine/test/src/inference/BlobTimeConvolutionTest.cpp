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

static inline void fillWithUniform( std::vector<float>& data, double min, double max, CRandom& random )
{
	for( size_t i = 0; i < data.size(); ++i ) {
		data[i] = static_cast<float>( random.Uniform( min, max ) );
	}
}

static void blobTimeConvolutionForwardTestImpl( const CTestParams& params, int seed, float precision = 1e-2 )
{
	CRandom random( seed );

	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval objectSizeInterval = params.GetInterval( "ObjectSize" );
	const CInterval filterSizeInterval = params.GetInterval( "FilterSize" );
	const CInterval filterCountInterval = params.GetInterval( "FilterCount" );
	const CInterval strideInterval = params.GetInterval( "Stride" );
	const CInterval paddingFrontInterval = params.GetInterval( "PaddingFront" );
	const CInterval paddingBackInterval = params.GetInterval( "PaddingBack" );
	const CInterval dilationInterval = params.GetInterval( "Dilation" );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int objectSize = random.UniformInt( objectSizeInterval.Begin, objectSizeInterval.End );
	int paddingFront = random.UniformInt( paddingFrontInterval.Begin, paddingFrontInterval.End );
	int paddingBack = random.UniformInt( paddingBackInterval.Begin, paddingBackInterval.End );
	const int filterSize = random.UniformInt( filterSizeInterval.Begin, std::min( filterSizeInterval.End,
		batchLength + paddingFront + paddingBack ) );
	const int filterCount = random.UniformInt( filterCountInterval.Begin, filterCountInterval.End );
	const int stride = random.UniformInt( strideInterval.Begin, strideInterval.End );
	const int maxDilation = filterSize == 1 ? 1 : ( batchLength - 1 ) / ( filterSize - 1 );
	const int dilation = random.UniformInt( dilationInterval.Begin, std::min( dilationInterval.End, maxDilation ) );
	if( paddingFront > ( filterSize - 1 ) * dilation ) {
		paddingFront = ( filterSize - 1 ) * dilation;
	}
	if( paddingBack > ( filterSize - 1 ) * dilation ) {
		paddingBack = ( filterSize - 1 ) * dilation;
	}

	const int outputSeqLen = ( batchLength - ( filterSize - 1 ) * dilation - 1 + paddingFront + paddingBack ) / stride + 1;
	CBlobDesc inputDesc( CT_Float );
	inputDesc.SetDimSize( BD_BatchLength, batchLength );
	inputDesc.SetDimSize( BD_BatchWidth, batchSize );
	inputDesc.SetDimSize( BD_ListSize, 1 );
	switch( random.Next() % 3 ) {
	case 0:
	{
		const int squareRoot = static_cast<int>( std::sqrt( objectSize ) );
		if( squareRoot * squareRoot == objectSize && random.Next() % 2 == 1 ) {
			inputDesc.SetDimSize(BD_Height, squareRoot);
			inputDesc.SetDimSize(BD_Width, squareRoot);
		} else {
			inputDesc.SetDimSize(BD_Height, objectSize);
			inputDesc.SetDimSize(BD_Width, 1);
		}
		inputDesc.SetDimSize(BD_Channels, 1);
		break;
	}
	case 1:
		inputDesc.SetDimSize(BD_Height, 1);
		inputDesc.SetDimSize(BD_Width, objectSize);
		inputDesc.SetDimSize(BD_Channels, 1);
		break;
	case 2:
		inputDesc.SetDimSize(BD_Height, 1);
		inputDesc.SetDimSize(BD_Width, 1);
		inputDesc.SetDimSize(BD_Channels, objectSize);
		break;
	default:
		ASSERT_TRUE( false );
	}

	CFloatBlob input( MathEngine(), batchLength, batchSize, 1, inputDesc.Height(), inputDesc.Width(), 1, inputDesc.Channels() );

	CFloatBlob output( MathEngine(), outputSeqLen, batchSize, 1, 1, 1, 1, filterCount );
	CFloatBlob filter( MathEngine(), filterCount, filterSize, 1, objectSize );
	CFloatBlob freeTerm( MathEngine(), 1, 1, 1, filterCount );
	std::vector<float> inputBuff;
	inputBuff.resize( input.GetDataSize() );
	std::vector<float> filterBuff;
	filterBuff.resize( filter.GetDataSize() );
	std::vector<float> freeTermBuff;
	freeTermBuff.resize( freeTerm.GetDataSize() );
	std::vector<float> expected;
	expected.resize( output.GetDataSize() );
	std::vector<float> actual;
	actual.resize( output.GetDataSize() );
	fillWithUniform( inputBuff, -1., 2., random );
	fillWithUniform( filterBuff, -1., 1., random );
	fillWithUniform( freeTermBuff, -1., -0.5, random );

	for( int outSeq = 0; outSeq < outputSeqLen; ++outSeq ) {
		for( int b = 0; b < batchSize; ++b ) {
			for( int f = 0; f < filterCount; ++f ) {
				const int outputIndex = getFlatIndex( output, outSeq, b, 0, f, 0, 0, 0 );
				expected[outputIndex] = freeTermBuff[f];
				for( int t = 0; t < filterSize; ++t ) {
					for( int ch = 0; ch < objectSize; ++ch ) {
						const int inputSeqIndex = outSeq * stride + t * dilation - paddingFront;
						if( inputSeqIndex >= 0 && inputSeqIndex < batchLength ) {
							const int inputIndex = getFlatIndex( input, inputSeqIndex, b, 0,
								ch, 0, 0, 0 );
							const int filterIndex = getFlatIndex( filter, 0, f, 0, ch, 0, t, 0 );
							expected[outputIndex] += inputBuff[inputIndex] * filterBuff[filterIndex];
						}
					}
				}
			}
		}
	}
	input.CopyFrom( inputBuff.data() );
	filter.CopyFrom( filterBuff.data() );
	freeTerm.CopyFrom( freeTermBuff.data() );

	CTimeConvolutionDesc* desc = MathEngine().InitTimeConvolution( input.GetDesc(), stride, paddingFront, paddingBack,
		dilation, filter.GetDesc(), output.GetDesc() );
	MathEngine().BlobTimeConvolution( *desc, input.GetData(), filter.GetData(),
		freeTerm.GetData(), output.GetData() );
	delete desc;

	output.CopyTo( actual.data() );
	for( size_t i = 0; i < expected.size(); ++i ) {
		ASSERT_NEAR( expected[i], actual[i], precision ) << "\nForward check failed"
			<< params;
	}
}	

//------------------------------------------------------------------------------------------------------------

class CMathEngineTimeConvolutionTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P(CMathEngineTimeConvolutionTestInstantiation, CMathEngineTimeConvolutionTest,
	::testing::Values(
		CTestParams(
			"BatchLength = 1;"
			"BatchSize = 1;"
			"ObjectSize = 1;"
			"FilterSize = 1;"
			"FilterCount = 1;"
			"Stride = 1;"
			"PaddingFront = 0;"
			"PaddingBack = 0;"
			"Dilation = 1;"
			"TestCount = 10"
		),
		CTestParams(
			"BatchLength = 7;"
			"BatchSize = 1;"
			"ObjectSize = 1;"
			"FilterSize = 1;"
			"FilterCount = 1;"
			"Stride = 1;"
			"PaddingFront = 0;"
			"PaddingBack = 0;"
			"Dilation = 1;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 1;"
			"BatchSize = 5;"
			"ObjectSize = 1;"
			"FilterSize = 1;"
			"FilterCount = 1;"
			"Stride = 1;"
			"PaddingFront = 0;"
			"PaddingBack = 0;"
			"Dilation = 1;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 1;"
			"BatchSize = 1;"
			"ObjectSize = 3;"
			"FilterSize = 1;"
			"FilterCount = 1;"
			"Stride = 1;"
			"PaddingFront = 0;"
			"PaddingBack = 0;"
			"Dilation = 1;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 1;"
			"BatchSize = 1;"
			"ObjectSize = 1;"
			"FilterSize = 1;"
			"FilterCount = 13;"
			"Stride = 1;"
			"PaddingFront = 0;"
			"PaddingBack = 0;"
			"Dilation = 1;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 11;"
			"BatchSize = 1;"
			"ObjectSize = 1;"
			"FilterSize = 11;"
			"FilterCount = 13;"
			"Stride = 1;"
			"PaddingFront = 0;"
			"PaddingBack = 0;"
			"Dilation = 1;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 12;"
			"BatchSize = 1;"
			"ObjectSize = 1;"
			"FilterSize = 11;"
			"FilterCount = 13;"
			"Stride = 1;"
			"PaddingFront = 0;"
			"PaddingBack = 0;"
			"Dilation = 1;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 13;"
			"BatchSize = 7;"
			"ObjectSize = 5;"
			"FilterSize = 3;"
			"FilterCount = 11;"
			"Stride = 2;"
			"PaddingFront = 2;"
			"PaddingBack = 2;"
			"Dilation = 2;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 13;"
			"BatchSize = 7;"
			"ObjectSize = 5;"
			"FilterSize = 3;"
			"FilterCount = 11;"
			"Stride = 2;"
			"PaddingFront = 2;"
			"PaddingBack = 0;"
			"Dilation = 2;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 13;"
			"BatchSize = 7;"
			"ObjectSize = 5;"
			"FilterSize = 3;"
			"FilterCount = 11;"
			"Stride = 2;"
			"PaddingFront = 0;"
			"PaddingBack = 2;"
			"Dilation = 2;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 113;"
			"BatchSize = 7;"
			"ObjectSize = 5;"
			"FilterSize = 5;"
			"FilterCount = 11;"
			"Stride = 3;"
			"PaddingFront = 4;"
			"PaddingBack = 4;"
			"Dilation = 3;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 5;"
			"BatchSize = 10;"
			"ObjectSize = 5;"
			"FilterSize = 2;"
			"FilterCount = 2;"
			"Stride = 1;"
			"PaddingFront = 3;"
			"PaddingBack = 3;"
			"Dilation = 3;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 5;"
			"BatchSize = 10;"
			"ObjectSize = 5;"
			"FilterSize = 2;"
			"FilterCount = 2;"
			"Stride = 1;"
			"PaddingFront = 3;"
			"PaddingBack = 0;"
			"Dilation = 3;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 5;"
			"BatchSize = 10;"
			"ObjectSize = 5;"
			"FilterSize = 2;"
			"FilterCount = 2;"
			"Stride = 1;"
			"PaddingFront = 0;"
			"PaddingBack = 3;"
			"Dilation = 3;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 2;"
			"BatchSize = 3;"
			"ObjectSize = 10000;"
			"FilterSize = 2;"
			"FilterCount = 1;"
			"Stride = 2;"
			"PaddingFront = 0;"
			"PaddingBack = 0;"
			"Dilation = 1;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 3;"
			"BatchSize = 15;"
			"ObjectSize = 3;"
			"FilterSize = 1;"
			"FilterCount = 1;"
			"Stride = 1;"
			"PaddingFront = 0;"
			"PaddingBack = 0;"
			"Dilation = 1;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = 3;"
			"BatchSize = 15;"
			"ObjectSize = 3;"
			"FilterSize = 1;"
			"FilterCount = 1;"
			"Stride = 1;"
			"PaddingFront = 0;"
			"PaddingBack = 0;"
			"Dilation = 1;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchLength = (3 .. 7);"
			"BatchSize = (1 .. 5);"
			"ObjectSize = (1 .. 7);"
			"FilterSize = (1 .. 20);"
			"FilterCount = (1 .. 3);"
			"Stride = (1 .. 3);"
			"PaddingFront = (0 .. 3);"
			"PaddingBack = (0 .. 3);"
			"Dilation = (1 .. 3);"
			"TestCount = 600"
		),
		CTestParams(
			"BatchLength = (3 .. 7);"
			"BatchSize = (1 .. 5);"
			"ObjectSize = (1000 .. 7000);"
			"FilterSize = (1 .. 20);"
			"FilterCount = (1 .. 3);"
			"Stride = (1 .. 3);"
			"PaddingFront = (0 .. 3);"
			"PaddingBack = (0 .. 3);"
			"Dilation = (1 .. 5);"
			"TestCount = 30"
		)
	)
);

TEST_P( CMathEngineTimeConvolutionTest, Random )
{
	RUN_TEST_IMPL( blobTimeConvolutionForwardTestImpl )
}
