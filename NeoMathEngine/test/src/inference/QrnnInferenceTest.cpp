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

static void naiveFPooling( bool reverse, int seqLength, int objSize,
	const float* z, const float* f, const float* h0,
	float* res )
{
	const int objOffset = reverse ? -objSize : objSize;

	if( reverse ) {
		const int firstElemOffset = ( seqLength - 1 ) * objSize;
		z += firstElemOffset;
		f += firstElemOffset;
		res += firstElemOffset;
	}

	if( h0 == nullptr ) {
		for( int index = 0; index < objSize; ++index ) {
			res[index] = z[index] * ( 1.f - f[index] );
		}
	} else {
		for( int index = 0; index < objSize; ++index ) {
			res[index] = f[index] * h0[index] + ( 1.f - f[index] ) * z[index];
		}
	}

	const float* hPrev = res;

	for( int step = 0; step < seqLength - 1; ++step ) {
		z += objOffset;
		f += objOffset;
		res += objOffset;
		for( int index = 0; index < objSize; ++index ) {
			res[index] = f[index] * hPrev[index] + ( 1.f - f[index] ) * z[index];
		}
		hPrev += objOffset;
	}
}

static void fPoolingImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const bool hasInitialState = random.Next() % 2 == 1;
	const bool reverse = random.Next() % 2 == 1;

	const int stateSize = channels * batchWidth;
	const int dataSize = batchLength * stateSize;

	CREATE_FILL_FLOAT_ARRAY( zData, -2.f, 2.f, dataSize, random );
	CFloatBlob zBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	zBlob.CopyFrom( zData.data() );

	CREATE_FILL_FLOAT_ARRAY( fData, -2.f, 2.f, dataSize, random );
	CFloatBlob fBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	fBlob.CopyFrom( fData.data() );

	CREATE_FILL_FLOAT_ARRAY( h0Data, -2.f, 2.f, stateSize, random );
	CFloatBlob h0Blob( MathEngine(), batchWidth, 1, 1, channels );
	h0Blob.CopyFrom( h0Data.data() );

	std::vector<float> expectedData( dataSize );
	naiveFPooling( reverse, batchLength, batchWidth * channels,
		zData.data(), fData.data(),
		hasInitialState ? h0Data.data() : nullptr,
		expectedData.data() );

	CFloatBlob actualBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	MathEngine().QrnnFPooling( reverse, batchLength, batchWidth * channels,
		zBlob.GetData(), fBlob.GetData(),
		hasInitialState ? h0Blob.GetData() : CFloatHandle(),
		actualBlob.GetData() );

	std::vector<float> actualData( actualBlob.GetDataSize() );
	actualBlob.CopyTo( actualData.data() );

	ASSERT_EQ( expectedData.size(), actualData.size() );
	for( int i = 0; i < dataSize; ++i ) {
		const bool res = FloatEq( expectedData[i], actualData[i], 1e-4f );
		if( !res ) {
			// __debugbreak();
			EXPECT_TRUE( false );
			break;
		}
	}
}

//------------------------------------------------------------------------------------------------------------

static void naiveIfPooling( bool reverse, int seqLength, int objSize,
	const float* z, const float* f, const float* i, const float* h0,
	float* res )
{
	const int objOffset = reverse ? -objSize : objSize;

	if( reverse ) {
		const int firstElemOffset = ( seqLength - 1 ) * objSize;
		z += firstElemOffset;
		f += firstElemOffset;
		i += firstElemOffset;
		res += firstElemOffset;
	}

	if( h0 == nullptr ) {
		for( int index = 0; index < objSize; ++index ) {
			res[index] = i[index] * z[index];
		}
	} else {
		for( int index = 0; index < objSize; ++index ) {
			res[index] = f[index] * h0[index] + i[index] * z[index];
		}
	}

	const float* hPrev = res;

	for( int step = 0; step < seqLength - 1; ++step ) {
		z += objOffset;
		f += objOffset;
		i += objOffset;
		res += objOffset;
		for( int index = 0; index < objSize; ++index ) {
			res[index] = f[index] * hPrev[index] + i[index] * z[index];
		}
		hPrev += objOffset;
	}
}

static void ifPoolingImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );

	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const bool hasInitialState = random.Next() % 2 == 1;
	const bool reverse = random.Next() % 2 == 1;

	const int stateSize = channels * batchWidth;
	const int dataSize = batchLength * stateSize;

	CREATE_FILL_FLOAT_ARRAY( zData, -2.f, 2.f, dataSize, random );
	CFloatBlob zBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	zBlob.CopyFrom( zData.data() );

	CREATE_FILL_FLOAT_ARRAY( fData, -2.f, 2.f, dataSize, random );
	CFloatBlob fBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	fBlob.CopyFrom( fData.data() );

	CREATE_FILL_FLOAT_ARRAY( iData, -2.f, 2.f, dataSize, random );
	CFloatBlob iBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	iBlob.CopyFrom( iData.data() );

	CREATE_FILL_FLOAT_ARRAY( h0Data, -2.f, 2.f, stateSize, random );
	CFloatBlob h0Blob( MathEngine(), batchWidth, 1, 1, channels );
	h0Blob.CopyFrom( h0Data.data() );

	std::vector<float> expectedData( dataSize );
	naiveIfPooling( reverse, batchLength, batchWidth * channels,
		zData.data(), fData.data(), iData.data(),
		hasInitialState ? h0Data.data() : nullptr,
		expectedData.data() );

	CFloatBlob actualBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	MathEngine().QrnnIfPooling( reverse, batchLength, batchWidth * channels,
		zBlob.GetData(), fBlob.GetData(), iBlob.GetData(),
		hasInitialState ? h0Blob.GetData() : CFloatHandle(),
		actualBlob.GetData() );

	std::vector<float> actualData( actualBlob.GetDataSize() );
	actualBlob.CopyTo( actualData.data() );

	ASSERT_EQ( expectedData.size(), actualData.size() );
	for( int i = 0; i < dataSize; ++i ) {
		EXPECT_TRUE( FloatEq( expectedData[i], actualData[i], 1e-4f ) );
	}
}

//------------------------------------------------------------------------------------------------------------

class CQrnnInferenceTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CQrnnInferenceTest, CQrnnInferenceTest,
	::testing::Values(
		CTestParams(
			"BatchLength = (1..20);"
			"BatchWidth = (1..10);"
			"Channels = (1..10);"
			"TestCount = 500;"
		),
		CTestParams(
			"BatchLength = (1..20);"
			"BatchWidth = (50..150);"
			"Channels = (50..150);"
			"TestCount = 10;"
		)
	)
);

//------------------------------------------------------------------------------------------------------------

TEST_P( CQrnnInferenceTest, FPoolingRandom )
{
	RUN_TEST_IMPL( fPoolingImpl );
}

TEST_P( CQrnnInferenceTest, IfPoolingRandom )
{
	RUN_TEST_IMPL( ifPoolingImpl );
}
