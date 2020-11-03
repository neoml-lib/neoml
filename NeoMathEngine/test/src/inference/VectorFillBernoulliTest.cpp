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

static void vectorFillBernoulliTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval valueInterval = params.GetInterval( "Value" );
	const CInterval probabilityInterval = params.GetInterval( "Prob" );

	const float value = static_cast<float>(random.Uniform( valueInterval.Begin, valueInterval.End ));
	const float prob = static_cast<float>(random.Uniform( probabilityInterval.Begin, probabilityInterval.End ));
	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	
	std::vector<float> expected;
	expected.resize( vectorSize );
	CExpectedRandom expectedRandom( seed );
	const unsigned int threshold = ( unsigned int ) ( ( double ) prob * UINT_MAX );
	int index = 0;
	for( int i = 0; i < ( vectorSize + 3 ) / 4; ++i ) {
		CIntArray<4> generated = expectedRandom.Next();
		for( int j = 0; j < 4 && index < vectorSize; ++j ) {
			expected[index++] = ( generated[j] <= threshold ) ? value : 0.f;
		}
	}

	std::vector<float> result;
	result.resize( vectorSize );
	MathEngine().VectorFillBernoulli( CARRAY_FLOAT_WRAPPER( result ), prob, vectorSize, value, seed );
	
	for( int i = 0; i < vectorSize; i++ ) {
		ASSERT_EQ( expected[i], result[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineVectorFillBernoulliTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineVectorFillBernoulliTestInstantiation, CMathEngineVectorFillBernoulliTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (10..1000);"
			"Value = (-100..100);"
			"Prob = (0..1);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineVectorFillBernoulliTest, Random)
{
	RUN_TEST_IMPL( vectorFillBernoulliTestImpl )
}
