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

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/Shuffler.h>

namespace NeoML {

CShuffler::CShuffler( CRandom& _random, int count ) :
	random( _random ),
	nextIndex( 0 )
{
	NeoAssert( count > 1 );
	indices.SetSize( count );
	for( int i = 0; i < indices.Size(); ++i ) {
		indices[i] = i;
	}
}

inline int CShuffler::getSwapIndex( int swapIndex )
{
	if( swapIndex != nextIndex ) {
		FObj::swap( indices[swapIndex], indices[nextIndex] );
	}
	return indices[nextIndex++];
}

int CShuffler::Next()
{
	NeoPresume( nextIndex < indices.Size() );
	const int swapIndex = random.UniformInt( nextIndex, indices.Size() - 1 );
	return getSwapIndex( swapIndex );
}

int CShuffler::SetNext( int index )
{
	int swapIndex = NotFound;
	if( indices[index] == index ) { // if the data has not been shuffled much, the index is in its place
		swapIndex = index;
		NeoAssert( swapIndex >= nextIndex );
	} else {
		// Look for the necessary index
		for( int i = nextIndex; i < indices.Size(); ++i ) {
			if( indices[i] == index ) {
				swapIndex = i;
				break;
			}
		}
		NeoAssert( swapIndex != NotFound );
	}
	return getSwapIndex( swapIndex );
}

const CArray<int>& CShuffler::GetAllIndices()
{
	while( nextIndex < indices.Size() ) {
		Next();
	}
	return indices;
}

} // namespace NeoML
