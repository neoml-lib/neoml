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

static void naiveLrn( const float* input, float* output, int vectorCount, int vectorSize, int windowSize,
	float bias, float alpha, float beta )
{
	for( int vec = 0; vec < vectorCount; ++vec ) {
		for( int ch = 0; ch < vectorSize; ++ch ) {
			double result = 0;
			const int firstC = std::max( 0, ch - ( windowSize - 1 ) / 2 );
			const int lastC = std::min( vectorSize - 1, ch + windowSize / 2 );
			for( int subC = firstC; subC <= lastC; ++subC ) {
				result += input[subC] * input[subC];
			}
			result *= alpha;
			result /= windowSize;
			result += bias;
			result = pow( result, static_cast<double>( -beta ) );
			*output++ = static_cast<float>( result * input[ch] );
		}
		input += vectorSize;
	}
}

static void testLrn( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval listSizeInterval = params.GetInterval( "ListSize" );
	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval depthInterval = params.GetInterval( "Depth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const float bias = static_cast<float>( random.Uniform( params.GetValue<double>( "MinBias" ),
		params.GetValue<double>( "MaxBias" ) ) );
	const float alpha = static_cast<float>( random.Uniform( params.GetValue<double>( "MinAlpha" ),
		params.GetValue<double>( "MaxAlpha" ) ) );
	const float beta = static_cast<float>( random.Uniform( params.GetValue<double>( "MinBeta" ),
		params.GetValue<double>( "MaxBeta" ) ) );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int listSize = random.UniformInt( listSizeInterval.Begin, listSizeInterval.End );
	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int depth = random.UniformInt( depthInterval.Begin, depthInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );

	const int windowSize = random.UniformInt( 1, channels );

	CREATE_FILL_FLOAT_ARRAY( input, valuesInterval.Begin, valuesInterval.End,
		batchLength * batchWidth * listSize * height * width * depth * channels, random );
	CFloatBlob inputBlob( MathEngine(), batchLength, batchWidth, listSize, height, width, depth, channels );
	inputBlob.CopyFrom( input.data() );

	std::vector<float> expected( inputBlob.GetDataSize() );
	naiveLrn( input.data(), expected.data(), batchLength * batchWidth * listSize * height * width * depth,
		channels, windowSize, bias, alpha, beta );

	CFloatBlob outputBlob( MathEngine(), batchLength, batchWidth, listSize, height, width, depth, channels );

	std::unique_ptr<CLrnDesc> desc( MathEngine().InitLrn( inputBlob.GetDesc(), windowSize, bias, alpha, beta ) );
	MathEngine().Lrn( *desc, inputBlob.GetData(), CFloatHandle(), CFloatHandle(), outputBlob.GetData() );

	std::vector<float> output( inputBlob.GetDataSize() );
	outputBlob.CopyTo( output.data() );

	for( int i = 0; i < inputBlob.GetDataSize(); ++i ) {
		ASSERT_NEAR( expected[i], output[i], 1e-3f ) << " at index " << i;
	}
}

class CMathEngineLrnInferenceTest : public CTestFixtureWithParams {
};

TEST_F( CMathEngineLrnInferenceTest, Precalc )
{
	const int height = 2;
	const int width = 3;
	const int channels = 7;
	const int dataSize = height * width * channels;

	std::vector<float> input = { -1.5f, -1.41463415f, -1.32926829f, -1.24390244f, -1.15853659f, -1.07317073f,
		-0.98780488f, -0.90243902f, -0.81707317f, -0.73170732f, -0.64634146f, -0.56097561f, -0.47560976f,
		-0.3902439f, -0.30487805f, -0.2195122f, -0.13414634f, -0.04878049f, 0.03658537f, 0.12195122f,
		0.20731707f, 0.29268293f, 0.37804878f, 0.46341463f, 0.54878049f, 0.63414634f, 0.7195122f, 0.80487805f,
		0.8902439f, 0.97560976f, 1.06097561f, 1.14634146f, 1.23170732f, 1.31707317f, 1.40243902f, 1.48780488f,
		1.57317073f, 1.65853659f, 1.74390244f, 1.82926829f, 1.91463415f, 2.f };

	CFloatBlob inputBlob( MathEngine(), 1, 1, 1, height, width, 1, channels );
	inputBlob.CopyFrom( input.data() );
	std::unique_ptr<CLrnDesc> desc( MathEngine().InitLrn( inputBlob.GetDesc(), 5, 3.f, 2e-2f, 0.81f ) );

	CFloatBlob outputBlob( MathEngine(), 1, 1, 1, height, width, 1, channels );
	MathEngine().Lrn( *desc, inputBlob.GetData(), CFloatHandle(), CFloatHandle(), outputBlob.GetData() );
	std::vector<float> output( dataSize );
	outputBlob.CopyTo( output.data() );

	std::vector<float> expected = { -0.6120848f, -0.57629544f, -0.5407432f, -0.50661045f, -0.4723609f, -0.43838462f,
		-0.4041842f, -0.36983216f, -0.33469746f, -0.29962757f, -0.26483864f, -0.22998759f, -0.19510205f, -0.16015589f,
		-0.12519395f, -0.0901394f, -0.05508512f, -0.02003264f, 0.01502456f, 0.05008285f, 0.08514106f, 0.12014931f,
		0.15514243f, 0.19009213f, 0.22500427f, 0.25986353f, 0.29491338f, 0.33001018f, 0.36449975f, 0.3988879f,
		0.43308502f, 0.467459f, 0.50172466f, 0.5371442f, 0.572765f, 0.60618573f, 0.6388899f, 0.67116714f, 0.7046276f,
		0.73792887f, 0.7746171f, 0.81177235f };

	for( int i = 0; i < dataSize; ++i ) {
		ASSERT_NEAR( expected[i], output[i], 1e-5f ) << " at index " << i;
	}
}

TEST_P( CMathEngineLrnInferenceTest, Random )
{
	RUN_TEST_IMPL( testLrn );
}

INSTANTIATE_TEST_CASE_P( CMathEngineLrnInferenceTestInstantiation, CMathEngineLrnInferenceTest,
	::testing::Values(
		CTestParams(
			"BatchLength = (1..3);"
			"BatchWidth = (1..3);"
			"ListSize = (1..3);"
			"Height = (10..15);"
			"Width = (10..15);"
			"Depth = (1..3);"
			"Channels = (1..3);"
			"MinBias = 0.000001;"
			"MaxBias = 10.;"
			"MinAlpha = 0.00001;"
			"MaxAlpha = 3;"
			"MinBeta = 0.000001;"
			"MaxBeta = 2;"
			"Values = (-10..10);"
			"TestCount = 1000;"
		)
	)
);
