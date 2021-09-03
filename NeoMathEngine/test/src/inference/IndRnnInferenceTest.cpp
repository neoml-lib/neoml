/* Copyright © 2017-2021 ABBYY Production LLC

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

static inline float sigmoid( float x )
{
	if( x <= 0.f ) {
		const float e = ::expf( x );
		return e / ( e + 1.f );
	} else {
		return 1.f / ( 1.f + ::expf( -x ) );
	}
}

static inline float relu( float x )
{
	return x > 0.f ? x : 0.f;
}

typedef float ( *TTestActivation ) ( float x );

static void indRnnRecurrentNaive( bool reverse, int seqLength, int batchSize, int objSize,
	TActivationFunction activation, const float* wx, const float* mask, const float* u, float* res )
{
	const int stepOffset = reverse ? -batchSize * objSize : batchSize * objSize;

	TTestActivation applyActivation = activation == AF_Sigmoid ? sigmoid : relu;

	if( reverse ) {
		const int firstElemOffset = ( seqLength - 1 ) * batchSize * objSize;
		wx += firstElemOffset;
		res += firstElemOffset;
	}

	for( int index = 0; index < batchSize * objSize; ++index ) {
		res[index] = applyActivation( wx[index] );
	}

	const float* hPrev = res;

	for( int step = 0; step < seqLength - 1; ++step ) {
		res += stepOffset;
		wx += stepOffset;
		for( int index = 0; index < batchSize * objSize; ++index ) {
			res[index] = applyActivation( wx[index] + u[index % objSize] * ( mask == nullptr ? hPrev[index] : hPrev[index] * mask[index] ) );
		}
		hPrev = res;
	}
}

static void indRnnInferenceTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );

	CMathEngineInfo meInfo;
	MathEngine().GetMathEngineInfo( meInfo );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const bool reverse = random.Next() % 2 == 1;
	const TActivationFunction activation = random.Next() % 2 == 1
		? AF_Sigmoid : AF_ReLU;

	float dropoutRate = 0.f;
	// Dropout is supported only on CPU and Cuda
	// other platforms are inference-only (that means no actual dropout)
	if( ( meInfo.Type == MET_Cpu || meInfo.Type == MET_Cuda ) && random.Next() % 2 == 1 ) {
		dropoutRate = static_cast<float>( random.Uniform( 0., 1. ) );
	}

	const int dataSize = batchLength * batchWidth * channels;

	CREATE_FILL_FLOAT_ARRAY( wxData, -1.5f, 2.f, dataSize, random );
	CFloatBlob wxBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	wxBlob.CopyFrom( wxData.data() );

	CFloatBlob maskBlob( MathEngine(), 1, batchWidth, 1, 1, 1, 1, channels );
	if( dropoutRate > 0 ) {
		const float forwardRate = 1.f - dropoutRate;
		MathEngine().VectorFillBernoulli( maskBlob.GetData(), forwardRate, maskBlob.GetDataSize(),
			1.f / forwardRate, static_cast<int>( random.Next() ) );
	}
	std::vector<float> maskData( maskBlob.GetDataSize() );
	maskBlob.CopyTo( maskData.data() );

	CREATE_FILL_FLOAT_ARRAY( uData, -2.f, 1.5f, channels, random );
	CFloatBlob uBlob( MathEngine(), 1, 1, 1, 1, 1, 1, channels );
	uBlob.CopyFrom( uData.data() );

	std::vector<float> expectedData( dataSize );
	indRnnRecurrentNaive( reverse, batchLength, batchWidth, channels, activation,
		wxData.data(), dropoutRate > 0 ? maskData.data() : nullptr, uData.data(),
		expectedData.data() );

	CFloatBlob actualBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	MathEngine().IndRnnRecurrent( reverse, batchLength, batchWidth, channels, activation,
		wxBlob.GetData(), dropoutRate > 0 ? maskBlob.GetData() : CFloatHandle(), uBlob.GetData(),
		actualBlob.GetData() );
	std::vector<float> actualData( dataSize );
	actualBlob.CopyTo( actualData.data() );

	ASSERT_EQ( expectedData.size(), actualData.size() );
	for( int i = 0; i < dataSize; ++i ) {
		EXPECT_TRUE( FloatEq( expectedData[i], actualData[i], 1e-4f ) );
	}
}

class CIndRnnInferenceTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CIndRnnInferenceTest, CIndRnnInferenceTest,
	::testing::Values(
		CTestParams(
			"BatchLength = (1..20);"
			"BatchWidth = (1..10);"
			"Channels = (1..10);"
			"TestCount = 5000;"
		),
		CTestParams(
			"BatchLength = (1..20);"
			"BatchWidth = (50..150);"
			"Channels = (50..150);"
			"TestCount = 10;"
		)
	)
);

TEST_P( CIndRnnInferenceTest, Random )
{
	RUN_TEST_IMPL( indRnnInferenceTestImpl );
}

