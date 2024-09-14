/* Copyright © 2024 ABBYY

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

#include <common.h>
#pragma hdrstop

#include <TestFixture.h>
#include <DnnSimpleTest.h>

using namespace NeoML;
using namespace NeoMLTest;

namespace NeoMLTest {

template<class T>
static void SpaceToDepthNaive( const CDnnBlob& source, const T* sourceData, int blockSize,
	const CDnnBlob& result, T* resultData )
{
	const int batchSize = source.GetObjectCount();
	const int sourceHeight = source.GetHeight();
	const int sourceWidth = source.GetWidth();
	const int sourceChannels = source.GetChannelsCount();
	for( int batch = 0; batch < batchSize; ++batch ) {
		for( int row = 0; row < sourceHeight; ++row ) {
			const int blockY = row / blockSize;
			const int inBlockY = row % blockSize;
			for( int col = 0; col < sourceWidth; ++col ) {
				const int blockX = col / blockSize;
				const int inBlockX = col % blockSize;
				for( int ch = 0; ch < sourceChannels; ++ch ) {
					const int resultCh = ( inBlockY * blockSize + inBlockX ) * sourceChannels + ch;
					resultData[GetFlatIndex( result, 0, batch, 0, resultCh, 0, blockY, blockX )]
						= sourceData[GetFlatIndex( source, 0, batch, 0, ch, 0, row, col )];
				}
			}
		}
	}
}

static void spaceToDepthTestFloat( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval blockSizeIntrval = params.GetInterval( "BlockSize" );
	const CInterval inputChannelsInterval = params.GetInterval( "InputChannels" );
	const CInterval outputHeightInterval = params.GetInterval( "OutputHeight" );
	const CInterval outputWidthInterval = params.GetInterval( "OutputWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int outputHeight = random.UniformInt( outputHeightInterval.Begin, outputHeightInterval.End );
	const int outputWidth = random.UniformInt( outputWidthInterval.Begin, outputWidthInterval.End );
	const int inputChannels = random.UniformInt( inputChannelsInterval.Begin, inputChannelsInterval.End );
	const int blockSize = random.UniformInt( blockSizeIntrval.Begin, blockSizeIntrval.End );
	const int inputHeight = outputHeight * blockSize;
	const int inputWidth = outputWidth * blockSize;
	const int outputChannels = inputChannels * blockSize * blockSize;

	CPtr<CDnnBlob> original = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1,
		batchSize, inputHeight, inputWidth, inputChannels );
	const int dataSize = original->GetDataSize();
	CREATE_FILL_FLOAT_ARRAY( originalData, valuesInterval.Begin, valuesInterval.End, dataSize, random );
	original->CopyFrom( originalData.GetPtr() );

	CPtr<CDnnBlob> converted = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 1,
		batchSize, outputHeight, outputWidth, outputChannels );
	CArray<float> convertedData;
	convertedData.SetSize( dataSize );
	SpaceToDepthNaive( *original, originalData.GetPtr(), blockSize, *converted, convertedData.GetPtr() );
	converted->CopyFrom( convertedData.GetPtr() );

	{
		CDnn s2dnn( random, MathEngine() );
		CPtr<CSourceLayer> data = Source( s2dnn, "data" );
		CPtr<CDnnSimpleTestDummyLearningLayer> learn = AddLayer<CDnnSimpleTestDummyLearningLayer>( "learn", { data } );
		CPtr<CSpaceToDepthLayer> s2d = SpaceToDepth( blockSize )( learn.Ptr() );
		CPtr<CSourceLayer> label = Source( s2dnn, "label" );
		CPtr<CDnnSimpleTestDummyLossLayer> loss = AddLayer<CDnnSimpleTestDummyLossLayer>( "loss", { s2d, label } );

		data->SetBlob( original );
		learn->ExpectedDiff = original->GetCopy();
		label->SetBlob( converted );
		loss->Diff = converted->GetCopy();

		s2dnn.RunAndLearnOnce();
	}

	{
		CDnn d2snn( random, MathEngine() );
		CPtr<CSourceLayer> data = Source( d2snn, "data" );
		CPtr<CDnnSimpleTestDummyLearningLayer> learn = AddLayer<CDnnSimpleTestDummyLearningLayer>( "learn", { data } );
		CPtr<CDepthToSpaceLayer> d2s = DepthToSpace( blockSize )( learn.Ptr() );
		CPtr<CSourceLayer> label = Source( d2snn, "label" );
		CPtr<CDnnSimpleTestDummyLossLayer> loss = AddLayer<CDnnSimpleTestDummyLossLayer>( "loss", { d2s, label } );

		data->SetBlob( converted );
		learn->ExpectedDiff = converted->GetCopy();
		label->SetBlob( original );
		loss->Diff = original->GetCopy();

		d2snn.RunAndLearnOnce();
	}
}

static void spaceToDepthTestInt( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval blockSizeIntrval = params.GetInterval( "BlockSize" );
	const CInterval inputChannelsInterval = params.GetInterval( "InputChannels" );
	const CInterval outputHeightInterval = params.GetInterval( "OutputHeight" );
	const CInterval outputWidthInterval = params.GetInterval( "OutputWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int outputHeight = random.UniformInt( outputHeightInterval.Begin, outputHeightInterval.End );
	const int outputWidth = random.UniformInt( outputWidthInterval.Begin, outputWidthInterval.End );
	const int inputChannels = random.UniformInt( inputChannelsInterval.Begin, inputChannelsInterval.End );
	const int blockSize = random.UniformInt( blockSizeIntrval.Begin, blockSizeIntrval.End );
	const int inputHeight = outputHeight * blockSize;
	const int inputWidth = outputWidth * blockSize;
	const int outputChannels = inputChannels * blockSize * blockSize;

	CPtr<CDnnBlob> original = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Int, 1,
		batchSize, inputHeight, inputWidth, inputChannels );
	const int dataSize = original->GetDataSize();
	CREATE_FILL_INT_ARRAY( originalData, valuesInterval.Begin, valuesInterval.End, dataSize, random );
	original->CopyFrom( originalData.GetPtr() );

	CPtr<CDnnBlob> converted = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Int, 1,
		batchSize, outputHeight, outputWidth, outputChannels );
	CArray<int> convertedData;
	convertedData.SetSize( dataSize );
	SpaceToDepthNaive( *original, originalData.GetPtr(), blockSize, *converted, convertedData.GetPtr() );
	converted->CopyFrom( convertedData.GetPtr() );

	{
		CDnn s2dnn( random, MathEngine() );
		CPtr<CSourceLayer> data = Source( s2dnn, "data" );
		CPtr<CSpaceToDepthLayer> s2d = SpaceToDepth( blockSize )( data.Ptr() );
		CPtr<CSinkLayer> sink = Sink( s2d.Ptr(), "sink" );

		data->SetBlob( original );
		s2dnn.RunOnce();

		CPtr<CDnnBlob> result = sink->GetBlob();
		int* buffer = result->GetBuffer<int>( 0, dataSize, /*exchange*/true );
		for( int i = 0; i < dataSize; ++i ) {
			EXPECT_EQ( convertedData[i], buffer[i] ) << i;
		}
		result->ReleaseBuffer( buffer, false );
	}

	{
		CDnn d2snn( random, MathEngine() );
		CPtr<CSourceLayer> data = Source( d2snn, "data" );
		CPtr<CDepthToSpaceLayer> d2s = DepthToSpace( blockSize )( data.Ptr() );
		CPtr<CSinkLayer> sink = Sink( d2s.Ptr(), "sink" );

		data->SetBlob( converted );
		d2snn.RunOnce();

		CPtr<CDnnBlob> result = sink->GetBlob();
		int* buffer = result->GetBuffer<int>( 0, dataSize, /*exchange*/true );
		for( int i = 0; i < dataSize; ++i ) {
			EXPECT_EQ( originalData[i], buffer[i] ) << i;
		}
		result->ReleaseBuffer( buffer, false );
	}
}

class CSpaceToDepthTest : public CNeoMlTestFixtureWithParams {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------

INSTANTIATE_TEST_CASE_P( CSpaceToDepthTestInstantiation, CSpaceToDepthTest,
	::testing::Values(
		CTestParams(
			"BatchSize = (1..5);"
			"BlockSize = (2..5);"
			"InputChannels = (1..10);"
			"OutputHeight = (1..10);"
			"OutputWidth = (1..10);"
			"Values = (-25..25);"
			"TestCount = 1000;"
		)
	)
);

TEST_P( CSpaceToDepthTest, Run )
{
	RUN_TEST_IMPL(spaceToDepthTestFloat)
	RUN_TEST_IMPL(spaceToDepthTestInt)
}
