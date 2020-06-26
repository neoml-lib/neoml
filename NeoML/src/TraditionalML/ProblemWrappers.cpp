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

#include <ProblemWrappers.h>

namespace NeoML {

IProblem::~IProblem()
{
}

IBaseRegressionProblem::~IBaseRegressionProblem()
{
}

IRegressionProblem::~IRegressionProblem()
{
}

IMultivariateRegressionProblem::~IMultivariateRegressionProblem()
{
}

IDataAccumulator::~IDataAccumulator()
{
}

/////////////////////////////////////////////////////////////////////////////////////////
// CMultivariateRegressionOverUnivariate

CMultivariateRegressionOverUnivariate::CMultivariateRegressionOverUnivariate(
		const IRegressionProblem* _inner ) :
	inner( _inner )
{
	NeoAssert( inner != 0 );
}

// Gets the number of features
int CMultivariateRegressionOverUnivariate::GetFeatureCount() const
{
	return inner->GetFeatureCount();
}

// Gets the number of vectors in the data set
int CMultivariateRegressionOverUnivariate::GetVectorCount() const
{
	return inner->GetVectorCount();
}

// Gets all vectors from the data set as a matrix
CSparseFloatMatrixDesc CMultivariateRegressionOverUnivariate::GetMatrix() const
{
	return inner->GetMatrix();
}

// Gets the vector weight
double CMultivariateRegressionOverUnivariate::GetVectorWeight( int index ) const
{
	return inner->GetVectorWeight( index );
}

// Gets the length of the function value vector
int CMultivariateRegressionOverUnivariate::GetValueSize() const
{
	return 1;
}

// Gets the function value for the vector with the given index in the data set
CFloatVector CMultivariateRegressionOverUnivariate::GetValue( int index ) const
{
	CFloatVector result(1);
	result.SetAt( 0, float( inner->GetValue( index ) ) );
	return result;
}

/////////////////////////////////////////////////////////////////////////////////////////
// CMultivariateRegressionOverClassification

CMultivariateRegressionOverClassification::CMultivariateRegressionOverClassification(
		const IProblem* _inner ) :
	inner( _inner )
{
	NeoAssert( inner != 0 );

	const int classCount = inner->GetClassCount();
	classValues.SetBufferSize( classCount );
	for( int i = 0; i < classCount; i++ ) {
		CFloatVector classValue( classCount );
		classValue.Nullify();
		classValue.SetAt( i, 1.f );
		classValues.Add( classValue );
	}
}

// Gets the number of features
int CMultivariateRegressionOverClassification::GetFeatureCount() const
{
	return inner->GetFeatureCount();
}

// Gets the number of vectors in the data set
int CMultivariateRegressionOverClassification::GetVectorCount() const
{
	return inner->GetVectorCount();
}

// Gets all vectors from the data set as a matrix
CSparseFloatMatrixDesc CMultivariateRegressionOverClassification::GetMatrix() const
{
	return inner->GetMatrix();
}

// Gets the vector weight
double CMultivariateRegressionOverClassification::GetVectorWeight( int index ) const
{
	return inner->GetVectorWeight( index );
}

// Gets the length of the function value vector
int CMultivariateRegressionOverClassification::GetValueSize() const
{
	return classValues.Size();
}

// Gets the function value for the vector with the given index in the data set
CFloatVector CMultivariateRegressionOverClassification::GetValue( int index ) const
{
	const int classIndex = inner->GetClass( index );
	NeoAssert( classIndex >= 0 && classIndex < classValues.Size() );
	return classValues[classIndex];
}

/////////////////////////////////////////////////////////////////////////////////////////
// CMultivariateRegressionOverBinaryClassification

CMultivariateRegressionOverBinaryClassification::CMultivariateRegressionOverBinaryClassification(
		const IProblem* _inner ) :
	inner( _inner )
{
	NeoAssert( inner != 0 );
	NeoAssert( inner->GetClassCount() == 2 );

	classValues[0] = CFloatVector(1);
	classValues[0].SetAt( 0, 0.f );
	classValues[1] = CFloatVector(1);
	classValues[1].SetAt( 0, 1.f );
}

// Gets the number of features
int CMultivariateRegressionOverBinaryClassification::GetFeatureCount() const
{
	return inner->GetFeatureCount();
}

// Gets the number of vectors in the data set
int CMultivariateRegressionOverBinaryClassification::GetVectorCount() const
{
	return inner->GetVectorCount();
}

// Gets all vectors from the data set as a matrix
CSparseFloatMatrixDesc CMultivariateRegressionOverBinaryClassification::GetMatrix() const
{
	return inner->GetMatrix();
}

// Gets the vector weight
double CMultivariateRegressionOverBinaryClassification::GetVectorWeight( int index ) const
{
	return inner->GetVectorWeight( index );
}

// Gets the length of the function value vector
int CMultivariateRegressionOverBinaryClassification::GetValueSize() const
{
	return 1;
}

// Gets the function value for the vector
CFloatVector CMultivariateRegressionOverBinaryClassification::GetValue( int index ) const
{
	const int classIndex = inner->GetClass( index );
	NeoAssert( classIndex >= 0 && classIndex < static_cast<int>( _countof( classValues ) ) );
	return classValues[classIndex];
}

/////////////////////////////////////////////////////////////////////////////////////////
// CProblemNotNullWeightsView 

CProblemNotNullWeightsView::CProblemNotNullWeightsView( const IProblem* _inner ) :
	inner( _inner )
{
	NeoAssert( inner != 0 );

	int originalVectorCount = inner->GetVectorCount();
	if( originalVectorCount > 0 ) {
		// fill nullWeightElementsMap, and adjust local MatrixDesc
		viewMatrixDesc = _inner->GetMatrix();
		// we are going to remap some elements, so create our own arrays of pointers
		viewMatrixDesc.PointerB = static_cast<int*>( 
			ALLOCATE_MEMORY( CurrentMemoryManager, originalVectorCount * sizeof( int ) ) );
		viewMatrixDesc.PointerE = static_cast<int*>( 
			ALLOCATE_MEMORY( CurrentMemoryManager, originalVectorCount * sizeof( int ) ) );

		int nullWeightElmentsCount = 0;
		for( int i = 0; i < originalVectorCount - nullWeightElmentsCount; ) {
			int iScanned = i + nullWeightElmentsCount;
			if( _inner->GetVectorWeight( iScanned ) == 0 ) {
				++nullWeightElmentsCount;
			} else {
				viewMatrixDesc.PointerB[i] = _inner->GetMatrix().PointerB[iScanned];
				viewMatrixDesc.PointerE[i] = _inner->GetMatrix().PointerE[iScanned];
				nullWeightElementsMap.Add( CIndexPair( { i, iScanned } ) );
				++i;
			}
		}
		viewMatrixDesc.Height -= nullWeightElmentsCount;
	}
}

CProblemNotNullWeightsView::~CProblemNotNullWeightsView()
{
	if( inner->GetVectorCount() > 0 ) {
		CurrentMemoryManager::Free( viewMatrixDesc.PointerB );
		CurrentMemoryManager::Free( viewMatrixDesc.PointerE );
	}
}

// Gets the number of classes
int CProblemNotNullWeightsView::GetClassCount() const
{
	return inner->GetClassCount();
}

// Gets the number of features
int CProblemNotNullWeightsView::GetFeatureCount() const
{
	return inner->GetFeatureCount();
}

// Indicates if the specified feature is discrete
bool CProblemNotNullWeightsView::IsDiscreteFeature( int index ) const
{
	return inner->IsDiscreteFeature( calculateOriginalIndex( index ) );
}

// Gets the number of vectors in the data set
int CProblemNotNullWeightsView::GetVectorCount() const
{
	return viewMatrixDesc.Height;
}

// The correct class number for a vector with a given index in [0, GetClassCount())
int CProblemNotNullWeightsView::GetClass( int index ) const
{
	return inner->GetClass( calculateOriginalIndex( index ) );
}

// Gets all vectors from the data set as a matrix
CSparseFloatMatrixDesc CProblemNotNullWeightsView::GetMatrix() const
{
	return viewMatrixDesc;
}

// Gets the vector weight
double CProblemNotNullWeightsView::GetVectorWeight( int index ) const
{
	return inner->GetVectorWeight( calculateOriginalIndex( index ) );
}

// calculate the index as if we had the matrix without null weighted elements
int CProblemNotNullWeightsView::calculateOriginalIndex( int viewedIndex ) const
{
	const int pos = nullWeightElementsMap.FindInsertionPoint<
		AscendingByMember<CIndexPair, int, &CIndexPair::ViewedIndex> >( viewedIndex );
	if( pos == 0 ) {
		return viewedIndex;
	} else {
		const int originalIndexBase = nullWeightElementsMap[pos-1].OriginalIndex;
		const int indexShift = viewedIndex - nullWeightElementsMap[pos-1].ViewedIndex;
		return originalIndexBase + indexShift;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////

} // namespace NeoML

