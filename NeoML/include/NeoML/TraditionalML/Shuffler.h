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
#include <NeoML/Random.h>

namespace NeoML {

// The shuffler class
// Uses the standard shuffling algorithm, not all at once but sequentially; as a result, the first N positions are only shuffled among themselves
// For example, you can use it to get the random indices in an array
class NEOML_API CShuffler {
public:
	CShuffler( CRandom& _random, int count );

	// Gets the next index
	int Next();
	// Sets the next index to the specified value (as if it was randomly generated) and returns the same value
	// If this index has been passed already, an error will occur
	int SetNext( int index );
	// Finishes shuffling and returns all indices
	const CArray<int>& GetAllIndices();

private:
	CRandom& random;
	CArray<int> indices;
	int nextIndex;

	int getSwapIndex( int swapIndex );
};

} // namespace NeoML
