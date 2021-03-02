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

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/CrossValidationSubProblem.h>

namespace NeoML {

CCrossValidationSubProblem::CCrossValidationSubProblem( const IProblem* _problem, int _partCount,
		int _partIndex, bool _testSet ) :
	problem( _problem ),
	partsCount( _partCount ),
	partIndex( _partIndex ),
	testSet( _testSet ),
	vectorsCount( 0 )
{
	NeoAssert( problem != 0 );
	NeoAssert( partsCount > 1 );
	NeoAssert( 0 <= partIndex && partIndex < partsCount );

	const int testSizeDiv = problem->GetVectorCount() / partsCount; // the number of subsets
	const int testSizeMod = problem->GetVectorCount() % partsCount; // the number of elements in the last subset

	if( testSet ) {
		// An element from each of the subsets and possibly another one from the last subset
		vectorsCount = testSizeDiv + ( partIndex < testSizeMod ? 1 : 0 );
	} else {
		// All the subsets with one element removed from each
		vectorsCount = testSizeDiv * ( partsCount - 1 ) + testSizeMod + ( partIndex < testSizeMod ? -1 : 0 );
	}

	CSparseFloatMatrixDesc baseMatrix = problem->GetMatrix();
	matrix.Height = vectorsCount;
	matrix.Width = baseMatrix.Width;
	if( baseMatrix.Columns == nullptr ) { // dense inside
		values.SetSize( matrix.Width * vectorsCount );
		matrix.Values = values.GetPtr();
		float* ptr = matrix.Values;
		for( int i = 0; i < vectorsCount; i++, ptr += matrix.Width ) {
			int index = translateIndex( i );
			CSparseFloatVectorDesc vec = baseMatrix.GetRow( index );
			NeoPresume( vec.Size == matrix.Width );
			::memcpy( ptr, vec.Values, vec.Size * sizeof( float ) );
		}
	} else {
		pointerB.SetSize( vectorsCount );
		pointerE.SetSize( vectorsCount );
		for( int i = 0; i < vectorsCount; i++ ) {
			int index = translateIndex( i );
			pointerB[i] = baseMatrix.PointerB[index];
			pointerE[i] = baseMatrix.PointerE[index];
		}
		matrix.Columns = baseMatrix.Columns;
		matrix.Values = baseMatrix.Values;
		matrix.PointerB = pointerB.GetPtr();
		matrix.PointerE = pointerE.GetPtr();
	}
}

// Converts the index to the initial data set index
int CCrossValidationSubProblem::translateIndex( int index ) const
{
	NeoAssert( index < vectorsCount );
	if( testSet ) {
		return index * partsCount + partIndex;
	}

	// Each subset except the last has partsCount elements
	const int groupIndex = index / ( partsCount - 1 );
	int indexInGroup = index % ( partsCount - 1 );

	if( indexInGroup >= partIndex ) {
		indexInGroup++; // Skipping the text example in this subset
	}

	return indexInGroup + groupIndex * partsCount;
}

} // namespace NeoML
