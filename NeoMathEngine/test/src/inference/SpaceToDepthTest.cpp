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

template<class T>
static void SpaceToDepthNaive( const CBlobDesc& source, const T* sourceData, int blockSize,
	const CBlobDesc& result, T* resultData )
{
	const int batchSize = source.ObjectCount();
	const int sourceHeight = source.Height();
	const int sourceWidth = source.Width();
	const int sourceChannels = source.Channels();
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

template<class T>
static void DepthToSpaceNaive( const CBlobDesc& source, const T* sourceData, int blockSize,
	const CBlobDesc& result, T* resultData )
{
	const int batchSize = source.ObjectCount();
	const int sourceHeight = source.Height();
	const int sourceWidth = source.Width();
	const int sourceChannels = source.Channels();
	const int resultChannels = sourceChannels / ( blockSize * blockSize );
	for( int batch = 0; batch < batchSize; ++batch ) {
		for( int blockX = 0; blockX < sourceHeight; ++blockX ) {
			for( int blockY = 0; blockY < sourceWidth; ++blockY ) {
				for( int ch = 0; ch < sourceChannels; ++ch ) {
					const int resultCh = ch % resultChannels;
					const int inBlockX = ( ch / resultChannels ) % blockSize;
					const int inBlockY = ( ch / resultChannels ) / blockSize;
					const int row = blockY * blockSize + inBlockY;
					const int col = blockX * blockSize + inBlockX;
					resultData[GetFlatIndex( result, 0, batch, 0, resultCh, 0, row, col )]
						= sourceData[GetFlatIndex( source, 0, batch, 0, ch, 0, blockY, blockX )];
				}
			}
		}
	}
}

template<class T>
static void spaceToDepthTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval outputHeightInterval = params.GetInterval( "OutputHeight" );
	const CInterval outputWidthInterval = params.GetInterval( "OutputWidth" );
	const CInterval inputChannelsInterval = params.GetInterval( "InputChannels" );
	const CInterval blockSizeIntrval = params.GetInterval( "BlockSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int outputHeight = random.UniformInt( outputHeightInterval.Begin, outputHeightInterval.End );
	const int outputWidth = random.UniformInt( outputWidthInterval.Begin, outputWidthInterval.End );
	const int inputChannels = random.UniformInt( inputChannelsInterval.Begin, inputChannelsInterval.End );
	const int blockSize = random.UniformInt( blockSizeIntrval.Begin, blockSizeIntrval.End );
	const int inputHeight = outputHeight * blockSize;
	const int inputWidth = outputWidth * blockSize;
	const int outputChannels = inputChannels * blockSize * blockSize;

	const int blobSize = batchSize * inputHeight * inputWidth * inputChannels;
	std::vector<T> inputData;
	inputData.resize( blobSize );
	for( int i = 0; i < blobSize; ++i ) {
		inputData[i] = static_cast<T>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) );

	}
	std::vector<T> expected( blobSize );
	
	{
		CBlob<T> inputBlob( MathEngine(), batchSize, inputHeight, inputWidth, inputChannels );
		inputBlob.CopyFrom( inputData.data() );

		CBlob<T> outputBlob( MathEngine(), batchSize, outputHeight, outputWidth, outputChannels );

		CMemoryHandleStackVar<T> fillValue( MathEngine(), 1 );
		fillValue.SetValue( static_cast<T>( -1 ) );
		MathEngine().VectorFill( outputBlob.GetData(), blobSize, fillValue );

		MathEngine().SpaceToDepth( inputBlob.GetDesc(), inputBlob.GetData(), blockSize, outputBlob.GetDesc(), outputBlob.GetData() );
		std::vector<T> actual( blobSize );
		outputBlob.CopyTo( actual.data() );

		SpaceToDepthNaive<T>( inputBlob.GetDesc(), inputData.data(), blockSize, outputBlob.GetDesc(), expected.data() );

		for( size_t i = 0; i < expected.size(); ++i ) {
			EXPECT_EQ( expected[i], actual[i] ) << "forward:" << i;
		}
	}

	{
		// Testing backward
		CBlob<T> inputBlob( MathEngine(), batchSize, outputHeight, outputWidth, outputChannels );
		inputBlob.CopyFrom( expected.data() );

		CBlob<T> outputBlob( MathEngine(), batchSize, inputHeight, inputWidth, inputChannels );

		CMemoryHandleStackVar<T> fillValue( MathEngine(), 1 );
		fillValue.SetValue( static_cast<T>( -1 ) );
		MathEngine().VectorFill( outputBlob.GetData(), blobSize, fillValue );

		MathEngine().DepthToSpace( inputBlob.GetDesc(), inputBlob.GetData(), blockSize, outputBlob.GetDesc(), outputBlob.GetData() );
		std::vector<T> actual( blobSize );
		outputBlob.CopyTo( actual.data() );

		for( size_t i = 0; i < inputData.size(); ++i ) {
			EXPECT_EQ( inputData[i], actual[i] ) << "backward:" << i;
		}
	}
}

class CMathEngineSpaceToDepthTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineSpaceToDepthTestInstantiation, CMathEngineSpaceToDepthTest,
	::testing::Values(
		CTestParams(
			"BlockSize = (2..5);"
			"BatchSize = (1..5);"
			"OutputHeight = (1..10);"
			"OutputWidth = (1..10);"
			"InputChannels = (1..10);"
			"Values = (-25..25);"
			"TestCount = 1000;"
		)
	)
);

TEST_P( CMathEngineSpaceToDepthTest, Random )
{
	RUN_TEST_IMPL( spaceToDepthTestImpl<int> );
	RUN_TEST_IMPL( spaceToDepthTestImpl<float> );
}

TEST_F( CMathEngineSpaceToDepthTest, ForwardPrecalc )
{
	const int batchSize = 2;
	const int height = 6;
	const int width = 4;
	const int channels = 3;
	const int dataSize = batchSize * height * width * channels;
	const int blockSize = 2;
	std::vector<float> inputData( dataSize );
	for( int i = 0; i < dataSize; ++i ) {
		inputData[i] = static_cast<float>( i );
	}
	std::vector<float> expected = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 36.0, 37.0, 38.0,
		39.0, 40.0, 41.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
		51.0, 52.0, 53.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 66.0, 67.0, 68.0,
		69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 78.0, 79.0, 80.0,
		81.0, 82.0, 83.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 108.0, 109.0,
		110.0, 111.0, 112.0, 113.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
		120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 126.0, 127.0, 128.0, 129.0,
		130.0, 131.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0 };

	CBlob<float> inputBlob( MathEngine(), batchSize, height, width, channels );
	inputBlob.CopyFrom( inputData.data() );

	CBlob<float> outputBlob( MathEngine(), batchSize, height / blockSize, width / blockSize,
		channels * blockSize * blockSize );

	MathEngine().SpaceToDepth( inputBlob.GetDesc(), inputBlob.GetData(), blockSize, outputBlob.GetDesc(), outputBlob.GetData() );

	std::vector<float> actual( dataSize );
	outputBlob.CopyTo( actual.data() );

	for( int i = 0; i < dataSize; ++i ) {
		EXPECT_FLOAT_EQ( expected[i], actual[i] ) << i;
	}
}

TEST_F( CMathEngineSpaceToDepthTest, BackwardPrecalc )
{
	const int batchSize = 2;
	const int height = 3;
	const int width = 2;
	const int channels = 32;
	const int dataSize = batchSize * height * width * channels;
	const int blockSize = 4;
	std::vector<float> inputData( dataSize );
	for( int i = 0; i < dataSize; ++i ) {
		inputData[i] = static_cast<float>( i );
	}

	std::vector<float> expected = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0,
		39.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 16.0, 17.0,
		18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 24.0, 25.0, 26.0, 27.0,
		28.0, 29.0, 30.0, 31.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0,
		70.0, 71.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
		104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 112.0,
		113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 120.0, 121.0,
		122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 160.0, 161.0,
		162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 136.0, 137.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 168.0, 169.0,
		170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 176.0, 177.0,
		178.0, 179.0, 180.0, 181.0, 182.0, 183.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 184.0, 185.0,
		186.0, 187.0, 188.0, 189.0, 190.0, 191.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0, 198.0, 199.0, 224.0, 225.0,
		226.0, 227.0, 228.0, 229.0, 230.0, 231.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 232.0, 233.0,
		234.0, 235.0, 236.0, 237.0, 238.0, 239.0, 208.0, 209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 240.0, 241.0,
		242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 216.0, 217.0, 218.0, 219.0, 220.0, 221.0, 222.0, 223.0, 248.0, 249.0,
		250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 256.0, 257.0, 258.0, 259.0, 260.0, 261.0, 262.0, 263.0, 288.0, 289.0,
		290.0, 291.0, 292.0, 293.0, 294.0, 295.0, 264.0, 265.0, 266.0, 267.0, 268.0, 269.0, 270.0, 271.0, 296.0, 297.0,
		298.0, 299.0, 300.0, 301.0, 302.0, 303.0, 272.0, 273.0, 274.0, 275.0, 276.0, 277.0, 278.0, 279.0, 304.0, 305.0,
		306.0, 307.0, 308.0, 309.0, 310.0, 311.0, 280.0, 281.0, 282.0, 283.0, 284.0, 285.0, 286.0, 287.0, 312.0, 313.0,
		314.0, 315.0, 316.0, 317.0, 318.0, 319.0, 320.0, 321.0, 322.0, 323.0, 324.0, 325.0, 326.0, 327.0, 352.0, 353.0,
		354.0, 355.0, 356.0, 357.0, 358.0, 359.0, 328.0, 329.0, 330.0, 331.0, 332.0, 333.0, 334.0, 335.0, 360.0, 361.0,
		362.0, 363.0, 364.0, 365.0, 366.0, 367.0, 336.0, 337.0, 338.0, 339.0, 340.0, 341.0, 342.0, 343.0, 368.0, 369.0,
		370.0, 371.0, 372.0, 373.0, 374.0, 375.0, 344.0, 345.0, 346.0, 347.0, 348.0, 349.0, 350.0, 351.0, 376.0, 377.0,
		378.0, 379.0, 380.0, 381.0, 382.0, 383.0 };

	CBlob<float> inputBlob( MathEngine(), batchSize, height, width, channels );
	inputBlob.CopyFrom( inputData.data() );

	CBlob<float> outputBlob( MathEngine(), batchSize, height * blockSize, width * blockSize,
		channels / ( blockSize * blockSize ) );

	MathEngine().DepthToSpace( inputBlob.GetDesc(), inputBlob.GetData(), blockSize,  outputBlob.GetDesc(), outputBlob.GetData() );

	std::vector<float> actual( dataSize );
	outputBlob.CopyTo( actual.data() );

	for( int i = 0; i < dataSize; ++i ) {
		EXPECT_FLOAT_EQ( expected[i], actual[i] ) << i;
	}
}
