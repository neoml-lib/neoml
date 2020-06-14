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

#include <NeoML/NeoMLDefs.h>

namespace NeoML {

// The class to generate random values
// It uses the complementary-multiply-with-carry algorithm
// C lag-1024, multiplier(a) = 108798, initial carry(c) = 12345678
class NEOML_API CRandom {
public:
	explicit CRandom( unsigned int seed = 0xBADF00D );

	// Resets to the starting state
	void Reset( unsigned int seed );

	// Returns the next random value
	unsigned int Next();
	// Returns a double value from a uniform distribution in [ min, max ) range
	// If min == max, min is returned
	double Uniform( double min, double max );
	// Returns an int value from a uniform distribution in [ min, max ] range. Note that max return value is possible!
	// If min == max, min is returned
	int UniformInt( int min, int max );
	// Returns a double value from a normal distribution N(mean, sigma)
	double Normal( double mean, double sigma );

	friend CArchive& operator<<( CArchive& archive, const CRandom& random);
	friend CArchive& operator>>( CArchive& archive, CRandom& random);

private:
	static const int lagSize = 1024;
	static const unsigned int stdLag[]; // the standard (pre-generated) lag
	static const unsigned int multiplier = 108798;
	static const unsigned int initialCarry = 12345678;

	unsigned int lag[lagSize];
	unsigned int carry;
	int lagPosition;
};

inline CArchive& operator<<( CArchive& archive, const CRandom& random )
{
	for(int i = 0; i < CRandom::lagSize; ++i) {
		archive << (int)random.lag[i];
	}
	archive << (int)random.carry;
	archive << random.lagPosition;

	return archive;
}

inline CArchive& operator>>( CArchive& archive, CRandom& random )
{
	for(int i = 0; i < CRandom::lagSize; ++i) {
		archive >> (int&)random.lag[i];
	}
	archive >> (int&)random.carry;
	archive >> random.lagPosition;

	return archive;
}

} // namespace NeoML
