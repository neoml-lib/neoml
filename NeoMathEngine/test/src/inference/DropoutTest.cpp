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

static void dropoutNaive( int batchLength, int batchWidth, int h, int w, int d, int c, float rate, 
	bool isSpatial, bool isBatchwise, int seed, const float *input, float *output )
{
	if( rate == 0.f ) {
		for( int i = 0; i < batchLength * batchWidth * h * w * d * c; i++ ) {
			output[i] = input[i];
		}
		return;
	}

	// Create mask
	float forwardRate = 1.f - rate;

	const int objectSize = isSpatial ? c : h * w * d * c;
	const int dropoutBatchLength = isBatchwise ? batchWidth * batchLength : batchLength;
	const int dropoutBatchWidth = batchWidth * batchLength / dropoutBatchLength;

	int maskSize = dropoutBatchWidth * objectSize;

	std::vector<float> mask;
	mask.resize( maskSize );
	CExpectedRandom expectedRandom( seed );

	const unsigned int threshold = ( unsigned int ) ( ( double ) forwardRate * UINT_MAX );
	int index = 0;
	for( int i = 0; i < ( maskSize + 3 ) / 4; ++i ) {
		CIntArray<4> generated = expectedRandom.Next();
		for( int j = 0; j < 4 && index < maskSize; ++j ) {
			mask[index++] = ( generated[j] <= threshold ) ? 1.f / forwardRate : 0.f;
		}
	}

	// Do dropout
	if( !isSpatial ) {
		for( int i = 0; i < dropoutBatchLength; i++ ) {
			for( int j = 0; j < maskSize; j++ ) {
				output[i * maskSize + j] = input[i * maskSize + j] * mask[j];
			}
		}
		return;
	}

	const float *currInput = input;
	float *currOutput = output;
	for( int b = 0; b < batchLength * batchWidth; b++ ) {
		for( int i = 0; i < h * w * d * c / objectSize; i++ ) {
			for( int j = 0; j < objectSize; j++ ) {
				currOutput[i * objectSize + j] = currInput[i * objectSize + j] * mask[( b % dropoutBatchWidth) * objectSize + j];
			}
		}
		currInput += h * w * d * c;
		currOutput += h * w * d * c;
	}
}

static void dropoutTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval depthInterval = params.GetInterval( "Depth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );

	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );

	const CInterval valuesInterval = params.GetInterval( "Values" );
	const CInterval rateInterval = params.GetInterval( "Rate" );

	const CInterval isSpatialInterval = params.GetInterval( "IsSpatial" );
	const CInterval isBatchwiseInterval = params.GetInterval( "IsBatchwise" );

	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int depth = random.UniformInt( depthInterval.Begin, depthInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );

	const float rate = static_cast<float>(random.Uniform( rateInterval.Begin, rateInterval.End ));

	const bool isSpatial = random.UniformInt( isSpatialInterval.Begin, isSpatialInterval.End ) == 1;
	const bool isBatchwise = random.UniformInt( isBatchwiseInterval.Begin, isBatchwiseInterval.End ) == 1;

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, batchLength * batchWidth * height * width * depth * channels, random );
	CFloatBlob input( MathEngine(), batchLength, batchWidth, 1, height, width, depth, channels );
	input.CopyFrom( inputData.data() );

	CFloatBlob output(MathEngine(), batchLength, batchWidth, 1, height, width, depth, channels);

	// expected
	std::vector<float> expected;
	expected.resize( inputData.size() );
	dropoutNaive( batchLength, batchWidth, height, width, depth, channels, rate, isSpatial, isBatchwise, seed, inputData.data(), expected.data() );

	// actual
	CDropoutDesc *dropoutDesc = MathEngine().InitDropout( rate, isSpatial, isBatchwise, input.GetDesc(), output.GetDesc(), seed );
	MathEngine().Dropout( *dropoutDesc, input.GetData(), output.GetData() );
	delete dropoutDesc;
	std::vector<float> result;
	result.resize( output.GetDataSize() );
	output.CopyTo( result.data() );

	// check
	for( size_t i = 0; i < result.size(); i++ ) {
		ASSERT_NEAR( expected[i], result[i], 1e-3f ) << i;
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineDropoutTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineDropoutTestInstantiation, CMathEngineDropoutTest,
	::testing::Values(
		CTestParams(
			"Height = (3..20);"
			"Width = (3..20);"
			"Depth = (3..20);"
			"Channels = (3..20);"
			"BatchLength = (1..5);"
			"BatchWidth = (1..5);"
			"IsSpatial = (0..1);"
			"IsBatchwise = (0..1);"
			"Rate = (0..1);"
			"Values = (-100..100);"
			"TestCount = 10;"
		)
	)
);

TEST_P(CMathEngineDropoutTest, Random)
{
	RUN_TEST_IMPL(dropoutTestImpl);
}
