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

template<int size>
class CIntArray {
public:
	static const int Size = size;

	CIntArray();

	const unsigned int& operator[] (int index) const { return data[index]; }
	unsigned int& operator[] (int index) { return data[index]; }

	const unsigned int* GetPtr() const { return data; }

private:
	unsigned int data[size];
};

template<int size>
inline CIntArray<size>::CIntArray()
{
	for (int i = 0; i < size; ++i) {
		data[i] = 0;
	}
}

// ====================================================================================================================

class CExpectedRandom {
public:
	explicit CExpectedRandom( int seed );

	void Skip( unsigned long long count );

	CIntArray<4> Next();

private:
	static const unsigned int kPhiloxW32A = 0x9E3779B9;
	static const unsigned int kPhiloxW32B = 0xBB67AE85;
	static const unsigned int kPhiloxM4x32A = 0xD2511F53;
	static const unsigned int kPhiloxM4x32B = 0xCD9E8D57;

	CIntArray<4> counter;
	CIntArray<2> key;

	static void raiseKey(CIntArray<2>& key);
	static CIntArray<4> computeSingleRound(const CIntArray<4>& counter, const CIntArray<2>& key);
	void skipOne();
};

inline CExpectedRandom::CExpectedRandom(int seed)
{
	key[0] = seed;
	key[1] = seed ^ 0xBADF00D;
	counter[2] = seed ^ 0xBADFACE;
	counter[3] = seed ^ 0xBADBEEF;
}

inline void CExpectedRandom::Skip(unsigned long long count)
{
	const unsigned int countLow = static_cast<unsigned int>(count);
	unsigned int countHigh = static_cast<unsigned int>(count >> 32);

	counter[0] += countLow;
	if (counter[0] < countLow) {
		countHigh++;
	}

	counter[1] += countHigh;
	if (counter[1] < countHigh) {
		if (++counter[2] == 0) {
			++counter[3];
		}
	}
}

inline CIntArray<4> CExpectedRandom::Next()
{
	CIntArray<4> currentCounter = counter;
	CIntArray<2> currentKey = key;

	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);

	skipOne();

	return currentCounter;
}

inline void CExpectedRandom::raiseKey(CIntArray<2>& key)
{
	key[0] += kPhiloxW32A;
	key[1] += kPhiloxW32B;
}

static inline void multiplyHighLow(unsigned int x, unsigned int y, unsigned int* resultLow,
	unsigned int* resultHigh)
{
	const unsigned long long product = static_cast<unsigned long long>(x) * y;
	*resultLow = static_cast<unsigned int>(product);
	*resultHigh = static_cast<unsigned int>(product >> 32);
}

inline CIntArray<4> CExpectedRandom::computeSingleRound(const CIntArray<4>& counter, const CIntArray<2>& key)
{
	unsigned int firstLow;
	unsigned int firstHigh;
	multiplyHighLow(kPhiloxM4x32A, counter[0], &firstLow, &firstHigh);

	unsigned int secondLow;
	unsigned int secondHigh;
	multiplyHighLow(kPhiloxM4x32B, counter[2], &secondLow, &secondHigh);

	CIntArray<4> result;
	result[0] = secondHigh ^ counter[1] ^ key[0];
	result[1] = secondLow;
	result[2] = firstHigh ^ counter[3] ^ key[1];
	result[3] = firstLow;
	return result;
}

inline void CExpectedRandom::skipOne()
{
	if (++counter[0] == 0) {
		if (++counter[1] == 0) {
			if (++counter[2] == 0) {
				++counter[3];
			}
		}
	}
}

// ====================================================================================================================
// Dropout

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
