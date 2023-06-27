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
#include <MeTestCommon.h>

using namespace NeoML;
using namespace NeoMLTest;

static void indRnnRecurrentBackwardNaive( bool reverse, int seqLength, int batchSize, int objSize,
	TActivationFunction activation, const float* mask, const float* u, const float* out, const float* outDiff,
	float* wxDiff )
{
	const int stepOffset = reverse ? -batchSize * objSize : batchSize * objSize;

	if( reverse ) {
		const int firstElemOffset = ( seqLength - 1 ) * batchSize * objSize;
		out += firstElemOffset;
		wxDiff += firstElemOffset;
		outDiff += firstElemOffset;
	}

	TTestActivationDiffOp activationDiffOp = activation == AF_Sigmoid ? sigmoidDiffOp : reluDiffOp;

	std::vector<float> currOutDiff;
	currOutDiff.resize( batchSize * objSize );
	::memcpy( currOutDiff.data(), outDiff, objSize * batchSize * sizeof( float ) );
	for( int step = 0; step < seqLength - 1; ++step ) {
		for( int index = 0; index < batchSize * objSize; ++index ) {
			wxDiff[index] = activationDiffOp( out[index], currOutDiff[index] );
			currOutDiff[index] = outDiff[stepOffset + index]
				+ wxDiff[index] * u[index % objSize] * ( mask == nullptr ? 1.f : mask[index] );
		}
		out += stepOffset;
		wxDiff += stepOffset;
		outDiff += stepOffset;
	}

	for( int index = 0; index < batchSize * objSize; ++index ) {
		wxDiff[index] = activationDiffOp( out[index], currOutDiff[index] );
	}
}

static void indRnnBackwardTestImpl( const CTestParams& params, int seed )
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

	const float dropoutRate = random.Next() % 2 == 1 ? static_cast<float>( random.Uniform( 0., 1. ) ) : 0.f;

	const int dataSize = batchLength * batchWidth * channels;

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

	CREATE_FILL_FLOAT_ARRAY( outData, -1.5f, 2.f, dataSize, random );
	CFloatBlob outBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	outBlob.CopyFrom( outData.data() );

	CREATE_FILL_FLOAT_ARRAY( outDiffData, -1.5f, 2.f, dataSize, random );
	CFloatBlob outDiffBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	outDiffBlob.CopyFrom( outDiffData.data() );

	std::vector<float> expectedData( dataSize );
	indRnnRecurrentBackwardNaive( reverse, batchLength, batchWidth, channels, activation,
		dropoutRate > 0 ? maskData.data() : nullptr, uData.data(), outData.data(), outDiffData.data(),
		expectedData.data() );

	CFloatBlob actualBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	MathEngine().IndRnnRecurrentBackward( reverse, batchLength, batchWidth, channels, activation,
		dropoutRate > 0 ? maskBlob.GetData() : CFloatHandle(), uBlob.GetData(), outBlob.GetData(), outDiffBlob.GetData(),
		actualBlob.GetData() );
	std::vector<float> actualData( dataSize );
	actualBlob.CopyTo( actualData.data() );

	ASSERT_EQ( expectedData.size(), actualData.size() );
	for( int i = 0; i < dataSize; ++i ) {
		EXPECT_TRUE( FloatEq( expectedData[i], actualData[i], 1e-2f ) );
	}
}

class CIndRnnBackwardTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CIndRnnBackwardTest, CIndRnnBackwardTest,
	::testing::Values(
		CTestParams(
			"BatchLength = (2..2);"
			"BatchWidth = (1..1);"
			"Channels = (1..1);"
			"TestCount = 10;"
		),
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

TEST_P( CIndRnnBackwardTest, Random )
{
	RUN_TEST_IMPL( indRnnBackwardTestImpl );
}
