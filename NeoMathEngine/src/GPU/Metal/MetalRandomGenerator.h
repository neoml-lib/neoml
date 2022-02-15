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

#include <metal_stdlib>

using namespace metal;

// An unsigned int array of constant size that can be copied
template<int size>
class CIntArray {
public:
	CIntArray();

	thread const unsigned int& operator[] ( int index ) const { return data[index]; }
	thread unsigned int& operator[] ( int index ) { return data[index]; }

	thread const unsigned int* GetPtr() const { return data; }

private:
	unsigned int data[size];
};

template<int size>
inline CIntArray<size>::CIntArray()
{
	for( int i = 0; i < size; ++i ) {
		data[i] = 0;
	}
}

// ====================================================================================================================
// Generator for dropout
class CMathEngineRandom {
public:
	CMathEngineRandom();
	// Initialize based on an int value
	explicit CMathEngineRandom( int seed );
	// Stop after generating count values
	void Skip( unsigned int count );
	// Gets the next 128 random bits
	CIntArray<4> Next();

private:
	constant static const unsigned int kPhiloxW32A = 0x9E3779B9;
	constant static const unsigned int kPhiloxW32B = 0xBB67AE85;
	constant static const unsigned int kPhiloxM4x32A = 0xD2511F53;
	constant static const unsigned int kPhiloxM4x32B = 0xCD9E8D57;

	CIntArray<4> counter;
	CIntArray<2> key;

	static void raiseKey( thread CIntArray<2>& key );
	static CIntArray<4> computeSingleRound( thread const CIntArray<4>& counter, thread const CIntArray<2>& key );
	void skipOne();
};

inline CMathEngineRandom::CMathEngineRandom() = default;

inline CMathEngineRandom::CMathEngineRandom( int seed )
{
	key[0] = seed;
	// Some random constants
	key[1] = seed ^ 0xBADF00D;
	counter[2] = seed ^ 0xBADFACE;
	counter[3] = seed ^ 0xBADBEEF;
}

inline void CMathEngineRandom::Skip( unsigned int count )
{
	counter[0] += count;
	if( counter[0] < count ) {
		if( ++counter[1] == 0 ) {
			if( ++counter[2] == 0 ) {
				++counter[3];
			}
		}
	}
}

CIntArray<4> CMathEngineRandom::Next()
{
	CIntArray<4> currentCounter = counter;
	CIntArray<2> currentKey = key;

	currentCounter = computeSingleRound( currentCounter, currentKey );
	raiseKey( currentKey );
	currentCounter = computeSingleRound( currentCounter, currentKey );
	raiseKey( currentKey );
	currentCounter = computeSingleRound( currentCounter, currentKey );
	raiseKey( currentKey );
	currentCounter = computeSingleRound( currentCounter, currentKey );
	raiseKey( currentKey );
	currentCounter = computeSingleRound( currentCounter, currentKey );
	raiseKey( currentKey );
	currentCounter = computeSingleRound( currentCounter, currentKey );
	raiseKey( currentKey );
	currentCounter = computeSingleRound( currentCounter, currentKey );
	raiseKey( currentKey );
	currentCounter = computeSingleRound( currentCounter, currentKey );
	raiseKey( currentKey );
	currentCounter = computeSingleRound( currentCounter, currentKey );
	raiseKey( currentKey );
	currentCounter = computeSingleRound( currentCounter, currentKey );
	raiseKey( currentKey );

	skipOne();

	return currentCounter;
}

inline void CMathEngineRandom::raiseKey( thread CIntArray<2>& key )
{
	key[0] += kPhiloxW32A;
	key[1] += kPhiloxW32B;
}

static inline void multiplyHighLow( unsigned int x, unsigned int y, thread unsigned int* resultLow,
	thread unsigned int* resultHigh )
{
	const unsigned int last16 = 0xFFFF;
	unsigned int product = ( x & last16 ) * ( y & last16 );
	*resultLow = product & last16;
	product >>= 16;
	unsigned int first = ( x >> 16 ) * ( y & last16 );
	unsigned int second = ( x & last16 ) * ( y >> 16 );
	product += ( first & last16 ) + ( second & last16 );
	*resultLow += product << 16;
	product >>= 16;
	product += ( first >> 16 ) + ( second >> 16 ) + ( x >> 16 ) * ( y >> 16 );
	*resultHigh = product;
}

inline CIntArray<4> CMathEngineRandom::computeSingleRound( thread const CIntArray<4>& counter, thread
	const CIntArray<2>& key )
{
	unsigned int firstLow;
	unsigned int firstHigh;
	multiplyHighLow( kPhiloxM4x32A, counter[0], &firstLow, &firstHigh );

	unsigned int secondLow;
	unsigned int secondHigh;
	multiplyHighLow( kPhiloxM4x32B, counter[2], &secondLow, &secondHigh );

	CIntArray<4> result;
	result[0] = secondHigh ^ counter[1] ^ key[0];
	result[1] = secondLow;
	result[2] = firstHigh ^ counter[3] ^ key[1];
	result[3] = firstLow;
	return result;
}

inline void CMathEngineRandom::skipOne()
{
	if( ++counter[0] == 0 ) {
		if( ++counter[1] == 0 ) {
			if( ++counter[2] == 0 ) {
				++counter[3];
			}
		}
	}
}
