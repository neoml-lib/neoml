/* Copyright © 2017-2024 ABBYY

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

// The generator used for dropout
class CCpuRandom final : public CCrtStaticOnlyAllocatedObject {
public:
	struct CCounter final : public CCrtStaticOnlyAllocatedObject {
		unsigned int Data[4]{};
	};

	// Initializes the array of four unsigned int
	explicit CCpuRandom( int seed );

	// Stop after generating 'count' values
	void Skip( uint64_t count );
	// Get the next random 128 bits
	void Next( CCounter& currentCounter );

private:
	const unsigned int seed;
	CCounter counter{};

	static void computeSingleRound( CCounter& currentCounter, const CCounter& counter, unsigned int* key );
};

//---------------------------------------------------------------------------------------------------------------------

inline CCpuRandom::CCpuRandom( int seed ) : seed( static_cast<unsigned int>( seed ) )
{
	// Several random constants
	counter.Data[2] = seed ^ 0xBADFACE;
	counter.Data[3] = seed ^ 0xBADBEEF;
}

inline void CCpuRandom::Skip( uint64_t count )
{
	const unsigned int countLow = static_cast<unsigned int>( count );
	unsigned int countHigh  = static_cast<unsigned int>( count >> 32 );

	counter.Data[0] += countLow;
	if( counter.Data[0] < countLow ) {
		countHigh++;
	}

	counter.Data[1] += countHigh;
	if( counter.Data[1] < countHigh && ++counter.Data[2] == 0 ) {
		++counter.Data[3];
	}
}

inline void CCpuRandom::Next( CCounter& currentCounter )
{
	unsigned int key[2]{ seed, seed ^ 0xBADF00D }; // random constant

	// loop is unrolled
	computeSingleRound( currentCounter, counter, key ); // 0
	computeSingleRound( currentCounter, currentCounter, key ); // 1
	computeSingleRound( currentCounter, currentCounter, key ); // 2
	computeSingleRound( currentCounter, currentCounter, key ); // 3
	computeSingleRound( currentCounter, currentCounter, key ); // 4
	computeSingleRound( currentCounter, currentCounter, key ); // 5
	computeSingleRound( currentCounter, currentCounter, key ); // 6
	computeSingleRound( currentCounter, currentCounter, key ); // 7
	computeSingleRound( currentCounter, currentCounter, key ); // 8
	computeSingleRound( currentCounter, currentCounter, key ); // 9

	// skip one
	if( ++counter.Data[0] == 0 && ++counter.Data[1] == 0 && ++counter.Data[2] == 0 ) {
		++counter.Data[3];
	}
}

inline void CCpuRandom::computeSingleRound( CCounter& currentCounter, const CCounter& counter, unsigned int* key )
{
	constexpr uint64_t kPhiloxM4x32A = 0xD2511F53;
	const uint64_t firstProduct = kPhiloxM4x32A * counter.Data[0];
	const unsigned int firstLow = static_cast<unsigned int>( firstProduct );
	const unsigned int firstHigh = static_cast<unsigned int>( firstProduct >> 32 );

	constexpr uint64_t kPhiloxM4x32B = 0xCD9E8D57;
	const uint64_t secondProduct = kPhiloxM4x32B * counter.Data[2];
	const unsigned int secondLow = static_cast<unsigned int>( secondProduct );
	const unsigned int secondHigh = static_cast<unsigned int>( secondProduct >> 32 );

	currentCounter.Data[0] = secondHigh ^ counter.Data[1] ^ key[0];
	currentCounter.Data[1] = secondLow;
	currentCounter.Data[2] = firstHigh ^ counter.Data[3] ^ key[1];
	currentCounter.Data[3] = firstLow;

	// raise key
	key[0] += 0x9E3779B9; // kPhiloxW32A;
	key[1] += 0xBB67AE85; // kPhiloxW32B;
}

} // namespace NeoML
