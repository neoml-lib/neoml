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

#include <NeoML/TraditionalML/StratifiedCrossValidationSubProblem.h>

namespace NeoML {

CStratifiedCrossValidationSubProblem::CStratifiedCrossValidationSubProblem( const IProblem* _problem, int _partsCount, int _partIndex,
		bool _testSet ) : 
	problem( _problem ), 
	partsCount( _partsCount ), 
	partIndex( _partIndex ), 
	testSet( _testSet )
{
	NeoAssert( problem != 0 );
	NeoAssert( partsCount > 1 );
	NeoAssert( 0 <= partIndex && partIndex < partsCount );

	minPartSize = problem->GetVectorCount() / partsCount;
	buildObjectsLists();
	if( testSet ) {
		// the number of elements in the test set == the number of elements in the partIndex part
		vectorsCount = objectsPerPart[partIndex].Size();
	} else {
		// all other elements belong to the training set
		vectorsCount = problem->GetVectorCount() - objectsPerPart[partIndex].Size();
	}

	CSparseFloatMatrixDesc baseMatrix = problem->GetMatrix();
	pointerB.SetSize( vectorsCount );
	pointerE.SetSize( vectorsCount );
	for( int i = 0; i < vectorsCount; i++ ) {
		int index = translateIndex( i );
		pointerB[i] = baseMatrix.PointerB[index];
		pointerE[i] = baseMatrix.PointerE[index];
	}

	matrix.Height = vectorsCount;
	matrix.Width = baseMatrix.Width;
	matrix.Columns = baseMatrix.Columns;
	matrix.Values = baseMatrix.Values;
	matrix.PointerB = pointerB.GetPtr();
	matrix.PointerE = pointerE.GetPtr();
}

// Creates partsCount lists with each containing the list of objects of one part
void CStratifiedCrossValidationSubProblem::buildObjectsLists()
{
	// Initialize the lists of parts' elements
	objectsPerPart.SetSize( partsCount );
	// The lists of elements by class
	CArray< CArray<int> > objectsPerClass;
	objectsPerClass.SetSize( problem->GetClassCount() );

	for( int i = 0; i < problem->GetVectorCount(); ++i ) {
		int currentClass = problem->GetClass( i );
		objectsPerClass[currentClass].Add( i );
		// Once the list for a class is as long as the number of parts, its elements are distributed into part lists, one per part
		// The array is then cleared
		if( objectsPerClass[currentClass].Size() == partsCount ) {
			for( int j = 0; j < partsCount; ++j ) {
				objectsPerPart[j].Add( objectsPerClass[currentClass][j] );
			}
			objectsPerClass[currentClass].Empty();
		}
	}

	// Distribute the rest of the elements so that no two elements from the same class end up in the same part
	int currentPartIndex = 0;
	for( int i = 0; i < objectsPerClass.Size(); ++i ) {
		for( int j = 0; j < objectsPerClass[i].Size(); ++j ) {
			objectsPerPart[currentPartIndex].Add( objectsPerClass[i][j] );
			++currentPartIndex;
			currentPartIndex %= partsCount;
		}
	}

	// Calculate the number of objects before the test part
	objectsBeforeTestPart = 0;
	for( int i = 0; i < partIndex; ++i ) {
		objectsBeforeTestPart += objectsPerPart[i].Size();
	}
}

int CStratifiedCrossValidationSubProblem::translateIndex( int index ) const
{
	NeoAssert( index < vectorsCount );
	if( testSet ) {
		return objectsPerPart[partIndex][index];
	} else {
		// If the index points to an object after test part, 
		// skip over the test part by adding the part size to the index
		if( index >= objectsBeforeTestPart ) {
			index += objectsPerPart[partIndex].Size();
		}
		// Let N be the total number of objects
		// k be the number of parts
		// The number of parts of (N / k) + 1 size
		int bigPartsNum = problem->GetVectorCount() % partsCount;
		// For the index in parts of (N / k) + 1 size
		if( index < bigPartsNum * ( minPartSize + 1 ) ) {
			return objectsPerPart[index / ( minPartSize + 1 )][index % ( minPartSize + 1 )];
		} else {
			index -= bigPartsNum * ( minPartSize + 1 );
			return objectsPerPart[index / minPartSize + bigPartsNum][index % minPartSize];
		}
	}
}

} // namespace NeoML
