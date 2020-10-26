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

#pragma once

#include <NeoMathEngine/CrtAllocatedObject.h>

namespace NeoML {

// An unsigned int array of constant size that can be copied
template<int size>
class CIntArray : public CCrtAllocatedObject {
public:
	static const int Size = size;

	CIntArray();

	const unsigned int& operator[] ( int index ) const { return data[index]; }
	unsigned int& operator[] ( int index ) { return data[index]; }

	const unsigned int* GetPtr() const { return data; }

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
// The generator used for dropout
class CCpuRandom : public CCrtAllocatedObject {
public:
	// Initializes the array of four unsigned int
	explicit CCpuRandom( int seed );

	// Stop after generating 'count' values
	void Skip( uint64_t count );

	// Get the next random 128 bits
	CIntArray<4> Next();

private:
	static const unsigned int kPhiloxW32A = 0x9E3779B9;
	static const unsigned int kPhiloxW32B = 0xBB67AE85;
	static const unsigned int kPhiloxM4x32A = 0xD2511F53;
	static const unsigned int kPhiloxM4x32B = 0xCD9E8D57;

	CIntArray<4> counter;
	CIntArray<2> key;

	static void raiseKey( CIntArray<2>& key );
	static CIntArray<4> computeSingleRound( const CIntArray<4>& counter, const CIntArray<2>& key );
	void skipOne();
};

inline CCpuRandom::CCpuRandom( int seed )
{
	key[0] = seed;
	// Several random constants
	key[1] = seed ^ 0xBADF00D;
	counter[2] = seed ^ 0xBADFACE;
	counter[3] = seed ^ 0xBADBEEF;
}

inline void CCpuRandom::Skip( uint64_t count )
{
	const unsigned int countLow = static_cast<unsigned int>( count );
	unsigned int countHigh  = static_cast<unsigned int>( count >> 32 );

	counter[0] += countLow;
	if( counter[0] < countLow ) {
		countHigh++;
	}

	counter[1] += countHigh;
	if( counter[1] < countHigh ) {
		if( ++counter[2] == 0 ) {
			++counter[3];
		}
	}
}

inline CIntArray<4> CCpuRandom::Next()
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

inline void CCpuRandom::raiseKey( CIntArray<2>& key )
{
	key[0] += kPhiloxW32A;
	key[1] += kPhiloxW32B;
}

static inline void multiplyHighLow( unsigned int x, unsigned int y, unsigned int* resultLow,
	unsigned int* resultHigh )
{
	const uint64_t product = static_cast<uint64_t>( x ) * y;
	*resultLow = static_cast<unsigned int>( product );
	*resultHigh = static_cast<unsigned int>( product >> 32 );
}

inline CIntArray<4> CCpuRandom::computeSingleRound( const CIntArray<4>& counter, const CIntArray<2>& key )
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

inline void CCpuRandom::skipOne()
{
	if( ++counter[0] == 0 ) {
		if( ++counter[1] == 0 ) {
			if( ++counter[2] == 0 ) {
				++counter[3];
			}
		}
	}
}

} // namespace NeoML
