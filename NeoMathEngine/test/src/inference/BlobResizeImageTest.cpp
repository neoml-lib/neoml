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

static void blobResizeImageTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	std::vector<CInterval> imageSizeInterval;
	params.GetArray( "ImageSize", imageSizeInterval );
	ASSERT_EQ( imageSizeInterval.size(), 3 );
	const CInterval deltaLeftInterval = params.GetInterval( "DeltaLeft" );
	const CInterval deltaRightInterval = params.GetInterval( "DeltaRight" );
	const CInterval deltaTopInterval = params.GetInterval( "DeltaTop" );
	const CInterval deltaBottomInterval = params.GetInterval( "DeltaBottom" );
	const CInterval defaultValueInterval = params.GetInterval( "DefaultValue" );

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int channels = random.UniformInt( imageSizeInterval[0].Begin, imageSizeInterval[0].End );
	const int height = random.UniformInt( imageSizeInterval[1].Begin, imageSizeInterval[1].End );
	const int width = random.UniformInt( imageSizeInterval[2].Begin, imageSizeInterval[2].End );
	
	const int deltaLeft = random.UniformInt( std::max( -width + 1, deltaLeftInterval.Begin ), deltaLeftInterval.End );
	const int minRight = -( width + std::min( 0, deltaLeft ) ) + 1;
	const int deltaRight = random.UniformInt( std::max( minRight, deltaRightInterval.Begin ),
		deltaRightInterval.End );

	const int deltaTop = random.UniformInt( std::max( -height + 1, deltaTopInterval.Begin ), deltaTopInterval.End );
	const int minBottom = -( height + std::min( 0, deltaTop ) ) + 1;
	const int deltaBottom = random.UniformInt( std::max( minBottom, deltaBottomInterval.Begin ),
		deltaBottomInterval.End ); 

	const float defaultValue = static_cast< float >( random.Uniform( defaultValueInterval.Begin,
		defaultValueInterval.End ) );

	CFloatBlob input( MathEngine(), batchSize, height, width, channels );
	CFloatBlob output( MathEngine(), batchSize, height + deltaTop + deltaBottom, width + deltaLeft + deltaRight, channels );
	std::vector<float> inputBuff;
	inputBuff.resize( input.GetDataSize() );
	std::vector<float> expected;
	expected.insert( expected.begin(), output.GetDataSize(), defaultValue );
	std::vector<float> actual;
	actual.resize( output.GetDataSize() );
	for( int b = 0; b < batchSize; ++b ) {
		for( int h = 0; h < height; ++h ) {
			for( int w = 0; w < width; ++w ) {
				for( int c = 0; c < channels; ++c ) {
					const int inputIndex = getFlatIndex( input, 0, b, 0, c, 0, h, w );
					inputBuff[inputIndex] = static_cast<float>( random.Uniform( 2., 5. ) );
					if( h + deltaTop >= 0 && h + deltaTop < output.GetDesc().Height()
						&& w + deltaLeft >= 0 && w + deltaLeft < output.GetDesc().Width() )
					{
						const int outputIndex = getFlatIndex( output, 0, b, 0, c, 0, h + deltaTop, w + deltaLeft );
						expected[outputIndex] = inputBuff[inputIndex];
					}
				}
			}
		}
	}
	input.CopyFrom( inputBuff.data() );
	MathEngine().BlobResizeImage( input.GetDesc(), input.GetData(), deltaLeft, deltaRight, deltaTop,
		deltaBottom, defaultValue, output.GetDesc(), output.GetData() );
	output.CopyTo( actual.data() );
	for( size_t i = 0; i < expected.size(); ++i ) {
		ASSERT_NEAR( expected[i], actual[i], 1e-3 ) << params;
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBlobResizeImageTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineBlobResizeImageTestInstantiation, CMathEngineBlobResizeImageTest,
	::testing::Values(
		CTestParams(
			"BatchSize = 1;"
			"ImageSize = { 2, 282, 282 };"
			"DeltaLeft = 2;"
			"DeltaRight = 3;"
			"DeltaTop = -4;"
			"DeltaBottom = -1;"
			"DefaultValue = 0;"
			"TestCount = 1"
		),
		CTestParams(
			"BatchSize = (1..5);"
			"ImageSize = { (1..5), (1..10), (1..10) };"
			"DeltaLeft = (-3..3);"
			"DeltaRight = (-3..3);"
			"DeltaTop = (-3..3);"
			"DeltaBottom = (-3..3);"
			"DefaultValue = -1;"
			"TestCount = 25000"
		),
		CTestParams(
			"BatchSize = (5..20);"
			"ImageSize = { (1..5), (5..50), (5..50) };"
			"DeltaLeft = (-10..10);"
			"DeltaRight = (-10..10);"
			"DeltaTop = (-10..10);"
			"DeltaBottom = (-10..10);"
			"DefaultValue = (-5..2);"
			"TestCount = 300"
		)
	)
);

TEST_P( CMathEngineBlobResizeImageTest, Random )
{
	RUN_TEST_IMPL(blobResizeImageTestImpl)
}
