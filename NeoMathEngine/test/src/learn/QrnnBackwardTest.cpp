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

static void naiveFPoolingBackward( bool reverse, int seqLength, int objSize,
	const float* z, const float* f, const float* h0,
	const float* out, float* outDiff,
	float* zDiff, float* fDiff )
{
	const int objOffset = reverse ? -objSize : objSize;

	if( reverse ) {
		const int firstElemOffset = ( seqLength - 1 ) * objSize;
		z += firstElemOffset;
		f += firstElemOffset;
		out += firstElemOffset;
		outDiff += firstElemOffset;
		zDiff += firstElemOffset;
		fDiff += firstElemOffset;
	}

	for( int step = 0; step < seqLength - 1; ++step ) {
		for( int index = 0; index < objSize; ++index ) {
			zDiff[index] = outDiff[index] * ( 1.f - f[index] );
			fDiff[index] = outDiff[index] * ( out[objOffset + index] - z[index] );
			outDiff[objOffset + index] += outDiff[index] * f[index];
		}
		z += objOffset;
		f += objOffset;
		out += objOffset;
		outDiff += objOffset;
		zDiff += objOffset;
		fDiff += objOffset;
	}

	if( h0 == nullptr ) {
		for( int index = 0; index < objSize; ++index ) {
			zDiff[index] = outDiff[index] * ( 1.f - f[index] );
			fDiff[index] = -z[index] * outDiff[index];
		}
	} else {
		for( int index = 0; index < objSize; ++index ) {
			zDiff[index] = outDiff[index] * ( 1.f - f[index] );
			fDiff[index] = outDiff[index] * ( h0[index] - z[index] );
		}
	}
}

static void fPoolingBackwardImpl( const CTestParams& params, int seed )
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

	CREATE_FILL_FLOAT_ARRAY( outData, -2.f, 2.f, dataSize, random );
	CFloatBlob outBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	outBlob.CopyFrom( outData.data() );

	CREATE_FILL_FLOAT_ARRAY( outDiffData, -2.f, 2.f, dataSize, random );
	CFloatBlob outDiffBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	outDiffBlob.CopyFrom( outDiffData.data() );

	std::vector<float> expectedZDiffData( dataSize );
	std::vector<float> expectedFDiffData( dataSize );
	naiveFPoolingBackward( reverse, batchLength, batchWidth * channels,
		zData.data(), fData.data(),
		hasInitialState ? h0Data.data() : nullptr,
		outData.data(), outDiffData.data(),
		expectedZDiffData.data(), expectedFDiffData.data() );

	CFloatBlob actualZDiffBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	CFloatBlob actualFDiffBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	MathEngine().QrnnFPoolingBackward( reverse, batchLength, batchWidth * channels,
		zBlob.GetData(), fBlob.GetData(),
		hasInitialState ? h0Blob.GetData() : CFloatHandle(),
		outBlob.GetData(), outDiffBlob.GetData(),
		actualZDiffBlob.GetData(), actualFDiffBlob.GetData() );

	std::vector<float> actualZDiffData( actualZDiffBlob.GetDataSize() );
	actualZDiffBlob.CopyTo( actualZDiffData.data() );
	std::vector<float> actualFDiffData( actualFDiffBlob.GetDataSize() );
	actualFDiffBlob.CopyTo( actualFDiffData.data() );

	ASSERT_EQ( expectedZDiffData.size(), actualZDiffData.size() );
	ASSERT_EQ( expectedFDiffData.size(), actualFDiffData.size() );
	for( int i = 0; i < dataSize; ++i ) {
		EXPECT_TRUE( FloatEq( expectedZDiffData[i], actualZDiffData[i], 1e-4f ) );
		EXPECT_TRUE( FloatEq( expectedFDiffData[i], actualFDiffData[i], 1e-4f ) );
	}
}

//------------------------------------------------------------------------------------------------------------

static void naiveIfPoolingBackward( bool reverse, int seqLength, int objSize,
	const float* z, const float* f, const float* i, const float* h0,
	const float* out, float* outDiff,
	float* zDiff, float* fDiff, float* iDiff )
{
	const int objOffset = reverse ? -objSize : objSize;

	if( reverse ) {
		const int firstElemOffset = ( seqLength - 1 ) * objSize;
		z += firstElemOffset;
		f += firstElemOffset;
		i += firstElemOffset;
		out += firstElemOffset;
		outDiff += firstElemOffset;
		zDiff += firstElemOffset;
		fDiff += firstElemOffset;
		iDiff += firstElemOffset;
	}

	for( int step = 0; step < seqLength - 1; ++step ) {
		for( int index = 0; index < objSize; ++index ) {
			zDiff[index] = outDiff[index] * i[index];
			fDiff[index] = outDiff[index] * out[objOffset + index];
			iDiff[index] = outDiff[index] * z[index];
			outDiff[objOffset + index] += outDiff[index] * f[index];
		}
		z += objOffset;
		f += objOffset;
		i += objOffset;
		out += objOffset;
		outDiff += objOffset;
		zDiff += objOffset;
		fDiff += objOffset;
		iDiff += objOffset;
	}

	if( h0 == nullptr ) {
		for( int index = 0; index < objSize; ++index ) {
			zDiff[index] = outDiff[index] * i[index];
			fDiff[index] = 0.f;
			iDiff[index] = outDiff[index] * z[index];
		}
	} else {
		for( int index = 0; index < objSize; ++index ) {
			zDiff[index] = outDiff[index] * i[index];
			fDiff[index] = outDiff[index] * h0[index];
			iDiff[index] = outDiff[index] * z[index];
		}
	}
}

static void ifPoolingBackwardImpl( const CTestParams& params, int seed )
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

	CREATE_FILL_FLOAT_ARRAY( outData, -2.f, 2.f, dataSize, random );
	CFloatBlob outBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	outBlob.CopyFrom( outData.data() );

	CREATE_FILL_FLOAT_ARRAY( outDiffData, -2.f, 2.f, dataSize, random );
	CFloatBlob outDiffBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	outDiffBlob.CopyFrom( outDiffData.data() );

	std::vector<float> expectedZDiffData( dataSize );
	std::vector<float> expectedFDiffData( dataSize );
	std::vector<float> expectedIDiffData( dataSize );
	naiveIfPoolingBackward( reverse, batchLength, batchWidth * channels,
		zData.data(), fData.data(), iData.data(),
		hasInitialState ? h0Data.data() : nullptr,
		outData.data(), outDiffData.data(),
		expectedZDiffData.data(), expectedFDiffData.data(), expectedIDiffData.data() );

	CFloatBlob actualZDiffBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	CFloatBlob actualFDiffBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	CFloatBlob actualIDiffBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, channels );
	MathEngine().QrnnIfPoolingBackward( reverse, batchLength, batchWidth * channels,
		zBlob.GetData(), fBlob.GetData(), iBlob.GetData(),
		hasInitialState ? h0Blob.GetData() : CFloatHandle(),
		outBlob.GetData(), outDiffBlob.GetData(),
		actualZDiffBlob.GetData(), actualFDiffBlob.GetData(), actualIDiffBlob.GetData() );

	std::vector<float> actualZDiffData( actualZDiffBlob.GetDataSize() );
	actualZDiffBlob.CopyTo( actualZDiffData.data() );
	std::vector<float> actualFDiffData( actualFDiffBlob.GetDataSize() );
	actualFDiffBlob.CopyTo( actualFDiffData.data() );
	std::vector<float> actualIDiffData( actualIDiffBlob.GetDataSize() );
	actualIDiffBlob.CopyTo( actualIDiffData.data() );

	ASSERT_EQ( expectedZDiffData.size(), actualZDiffData.size() );
	ASSERT_EQ( expectedFDiffData.size(), actualFDiffData.size() );
	ASSERT_EQ( expectedIDiffData.size(), actualIDiffData.size() );
	for( int i = 0; i < dataSize; ++i ) {
		EXPECT_TRUE( FloatEq( expectedZDiffData[i], actualZDiffData[i], 1e-4f ) );
		EXPECT_TRUE( FloatEq( expectedFDiffData[i], actualFDiffData[i], 1e-4f ) );
		EXPECT_TRUE( FloatEq( expectedIDiffData[i], actualIDiffData[i], 1e-4f ) );
	}
}

//------------------------------------------------------------------------------------------------------------

class CQrnnBackwardTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CQrnnBackwardTest, CQrnnBackwardTest,
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

TEST_P( CQrnnBackwardTest, FPoolingRandom )
{
	RUN_TEST_IMPL( fPoolingBackwardImpl );
}

TEST_P( CQrnnBackwardTest, IfPoolingRandom )
{
	RUN_TEST_IMPL( ifPoolingBackwardImpl );
}
