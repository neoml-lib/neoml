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

static void blobGlobalMaxPoolingBackwardNaive( float* inputDiff, const float* outputDiff, const int* indices,
	int batchSize, int maxCount, int channels, int objectSize )
{
	for( int b = 0; b < batchSize; ++b ) {
		for( int i = 0; i < maxCount; ++i ) {
			float* inputDiffChannelData = inputDiff;
			for( int c = 0; c < channels; ++c ) {
				int index = *indices++;
				if( index >= 0 ) {
					inputDiffChannelData[index * channels] = *outputDiff;
				}
				++outputDiff;
				++inputDiffChannelData;
			}
		}
		inputDiff += objectSize;
	}
}

static void globalMaxPoolingBackwardImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval listSizeInterval = params.GetInterval( "ListSize" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval inputDepthInterval = params.GetInterval( "InputDepth" );
	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval outputDepthInterval = params.GetInterval( "OutputDepth" );
	const CInterval outputHeightInterval = params.GetInterval( "OutputHeight" );
	const CInterval outputWidthInterval = params.GetInterval( "OutputWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int listSize = random.UniformInt( listSizeInterval.Begin, std::min( listSizeInterval.End,
		( batchLengthInterval.End * batchWidthInterval.End ) / ( batchLength * batchWidth ) ) );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int inputDepth = random.UniformInt( inputDepthInterval.Begin, inputDepthInterval.End );
	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, std::min( inputWidthInterval.End,
		( inputDepthInterval.End * inputHeightInterval.End ) / ( inputDepth * inputHeight ) ) );
	const int outputDepth = random.UniformInt( outputDepthInterval.Begin, outputDepthInterval.End );
	const int outputHeight = random.UniformInt( outputHeightInterval.Begin, outputHeightInterval.End );
	const int outputWidth = random.UniformInt( outputWidthInterval.Begin, outputWidthInterval.End );

	CFloatBlob outputDiff( MathEngine(), batchLength, batchWidth, listSize, outputHeight, outputWidth, outputDepth, channels );
	CFloatBlob output( MathEngine(), batchLength, batchWidth, listSize, outputHeight, outputWidth, outputDepth, channels );
	CIntBlob indices( MathEngine(), batchLength, batchWidth, listSize, outputHeight, outputWidth, outputDepth, channels );
	CFloatBlob inputDiff( MathEngine(), batchLength, batchWidth, listSize, inputHeight, inputWidth, inputDepth, channels );
	CFloatBlob input( MathEngine(), batchLength, batchWidth, listSize, inputHeight, inputWidth, inputDepth, channels );

	CREATE_FILL_FLOAT_ARRAY( outputDiffBuff, valuesInterval.Begin, valuesInterval.End, outputDiff.GetDataSize(), random )
	outputDiff.CopyFrom( outputDiffBuff.data() );

	CREATE_FILL_FLOAT_ARRAY( outputBuff, valuesInterval.Begin, valuesInterval.End, output.GetDataSize(), random )
	output.CopyFrom( outputBuff.data() );

	CREATE_FILL_FLOAT_ARRAY( inputBuff, valuesInterval.Begin, valuesInterval.End, input.GetDataSize(), random )
	input.CopyFrom( inputBuff.data() );

	std::vector<float> expectedDiff;
	expectedDiff.insert( expectedDiff.begin(), inputDiff.GetDataSize(), 0 );
	std::vector<float> actualDiff( inputDiff.GetDataSize() );

	CGlobalMaxPoolingDesc* poolingDesc = MathEngine().InitGlobalMaxPooling( inputDiff.GetDesc(), indices.GetDesc(),
		outputDiff.GetDesc() );
	MathEngine().BlobGlobalMaxPooling( *poolingDesc, input.GetData(),
		indices.GetData(), output.GetData() );
	MathEngine().BlobGlobalMaxPoolingBackward( *poolingDesc, outputDiff.GetData(),
		indices.GetData(), inputDiff.GetData() );
	delete poolingDesc;

	inputDiff.CopyTo( actualDiff.data() );
	std::vector<int> indicesBuff( indices.GetDataSize() );
	indices.CopyTo( indicesBuff.data() );

	blobGlobalMaxPoolingBackwardNaive( expectedDiff.data(), outputDiffBuff.data(), indicesBuff.data(), batchLength * batchWidth * listSize,
		outputHeight * outputWidth * outputDepth, channels, inputHeight * inputWidth * inputDepth * channels );

	for( size_t i = 0; i < expectedDiff.size(); ++i ) {
		ASSERT_FLOAT_EQ( expectedDiff[i], actualDiff[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CGlobalMaxPoolingBackwardTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CGlobalMaxPoolingBackwardTestInstantiation, CGlobalMaxPoolingBackwardTest,
	::testing::Values(
		CTestParams(
			"BatchLength = 1;"
			"BatchWidth = 2;"
			"ListSize = 1;"
			"Channels = 5;"
			"InputDepth = 1;"
			"InputHeight = 1;"
			"InputWidth = 6;"
			"OutputDepth = 1;"
			"OutputHeight = 1;"
			"OutputWidth = 3;"
			"TestCount = 1;"
			"Values = (-10..10);"
		),
		CTestParams(
			"BatchLength = 1;"
			"BatchWidth = 1;"
			"ListSize = 1;"
			"Channels = 1;"
			"InputDepth = 1;"
			"InputHeight = 1;"
			"InputWidth = 75421;"
			"OutputDepth = 1;"
			"OutputHeight = 5;"
			"OutputWidth = 1;"
			"TestCount = 1;"
			"Values = (-10..10);"
		),
		CTestParams(
			"BatchLength = 1;"
			"BatchWidth = 2;"
			"ListSize = 1;"
			"Channels = 5;"
			"InputDepth = 2;"
			"InputHeight = 3;"
			"InputWidth = 5;"
			"OutputDepth = 5;"
			"OutputHeight = 3;"
			"OutputWidth = 2;"
			"TestCount = 1;"
			"Values = (-10..10);"
		),
		CTestParams(
			"BatchLength = (1..3);"
			"BatchWidth = (1..3);"
			"ListSize = (1..9);"
			"Channels = (1..3);"
			"InputDepth = (1..3);"
			"InputHeight = (1..3);"
			"InputWidth = (1..9);"
			"OutputDepth = (1..3);"
			"OutputHeight = (1..3);"
			"OutputWidth = (1..3);"
			"TestCount = 1000;"
			"Values = (-10..10);"
		),
		CTestParams(
			"BatchLength = (1..5);"
			"BatchWidth = (1..25);"
			"ListSize = (1..25);"
			"Channels = (1..100);"
			"InputDepth = (1..25);"
			"InputHeight = (1..5);"
			"InputWidth = (1..25);"
			"OutputDepth = (1..4);"
			"OutputHeight = (1..4);"
			"OutputWidth = (1..4);"
			"TestCount = 10;"
			"Values = (-10..10);"
		)
	)
);

TEST_P( CGlobalMaxPoolingBackwardTest, Random )
{
	RUN_TEST_IMPL( globalMaxPoolingBackwardImpl );
}
