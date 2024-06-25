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

using namespace NeoML;
using namespace NeoMLTest;

static void blobGlobalMaxPoolingBackwardNaive( float* sourceDiff, const float* resultDiff, const int* indices,
	int batchSize, int maxCount, int channels, int objectSize )
{
	for( int b = 0; b < batchSize; ++b ) {
		for( int i = 0; i < maxCount; ++i ) {
			float* sourceDiffChannelData = sourceDiff;
			for( int c = 0; c < channels; ++c ) {
				int index = *indices++;
				if( index >= 0 ) {
					sourceDiffChannelData[index * channels] = *resultDiff;
				}
				++resultDiff;
				++sourceDiffChannelData;
			}
		}
		sourceDiff += objectSize;
	}
}

static void globalMaxPoolingBackwardImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval listSizeInterval = params.GetInterval( "ListSize" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval sourceDepthInterval = params.GetInterval( "InputDepth" );
	const CInterval sourceHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval sourceWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval resultDepthInterval = params.GetInterval( "OutputDepth" );
	const CInterval resultHeightInterval = params.GetInterval( "OutputHeight" );
	const CInterval resultWidthInterval = params.GetInterval( "OutputWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int listSize = random.UniformInt( listSizeInterval.Begin, std::min( listSizeInterval.End,
		( batchLengthInterval.End * batchWidthInterval.End ) / ( batchLength * batchWidth ) ) );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int sourceDepth = random.UniformInt( sourceDepthInterval.Begin, sourceDepthInterval.End );
	const int sourceHeight = random.UniformInt( sourceHeightInterval.Begin, sourceHeightInterval.End );
	const int sourceWidth = random.UniformInt( sourceWidthInterval.Begin, std::min( sourceWidthInterval.End,
		( sourceDepthInterval.End * sourceHeightInterval.End ) / ( sourceDepth * sourceHeight ) ) );
	const int resultDepth = random.UniformInt( resultDepthInterval.Begin, resultDepthInterval.End );
	const int resultHeight = random.UniformInt( resultHeightInterval.Begin, resultHeightInterval.End );
	const int resultWidth = random.UniformInt( resultWidthInterval.Begin, resultWidthInterval.End );

	CFloatBlob resultDiff( MathEngine(), batchLength, batchWidth, listSize, resultHeight, resultWidth, resultDepth, channels );
	CFloatBlob result( MathEngine(), batchLength, batchWidth, listSize, resultHeight, resultWidth, resultDepth, channels );
	CIntBlob indices( MathEngine(), batchLength, batchWidth, listSize, resultHeight, resultWidth, resultDepth, channels );
	CFloatBlob sourceDiff( MathEngine(), batchLength, batchWidth, listSize, sourceHeight, sourceWidth, sourceDepth, channels );
	CFloatBlob source( MathEngine(), batchLength, batchWidth, listSize, sourceHeight, sourceWidth, sourceDepth, channels );

	CREATE_FILL_FLOAT_ARRAY( resultDiffBuff, valuesInterval.Begin, valuesInterval.End, resultDiff.GetDataSize(), random )
	resultDiff.CopyFrom( resultDiffBuff.data() );

	CREATE_FILL_FLOAT_ARRAY( resultBuff, valuesInterval.Begin, valuesInterval.End, result.GetDataSize(), random )
	result.CopyFrom( resultBuff.data() );

	CREATE_FILL_FLOAT_ARRAY( sourceBuff, valuesInterval.Begin, valuesInterval.End, source.GetDataSize(), random )
	source.CopyFrom( sourceBuff.data() );

	std::vector<float> expectedDiff;
	expectedDiff.insert( expectedDiff.begin(), sourceDiff.GetDataSize(), 0 );
	std::vector<float> actualDiff( sourceDiff.GetDataSize() );

	CGlobalMaxPoolingDesc* poolingDesc = MathEngine().InitGlobalMaxPooling( sourceDiff.GetDesc(), indices.GetDesc(),
		resultDiff.GetDesc() );
	MathEngine().BlobGlobalMaxPooling( *poolingDesc, source.GetData(),
		indices.GetData(), result.GetData() );
	MathEngine().BlobGlobalMaxPoolingBackward( *poolingDesc, resultDiff.GetData(),
		indices.GetData(), sourceDiff.GetData() );
	delete poolingDesc;

	sourceDiff.CopyTo( actualDiff.data() );
	std::vector<int> indicesBuff( indices.GetDataSize() );
	indices.CopyTo( indicesBuff.data() );

	blobGlobalMaxPoolingBackwardNaive( expectedDiff.data(), resultDiffBuff.data(), indicesBuff.data(), batchLength * batchWidth * listSize,
		resultHeight * resultWidth * resultDepth, channels, sourceHeight * sourceWidth * sourceDepth * channels );

	for( size_t i = 0; i < expectedDiff.size(); ++i ) {
		EXPECT_FLOAT_EQ( expectedDiff[i], actualDiff[i] );
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
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}

	RUN_TEST_IMPL( globalMaxPoolingBackwardImpl );
}
