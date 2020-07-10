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
inline int CMultivariateRegressionOverUnivariate::GetFeatureCount() const
{
	return inner->GetFeatureCount();
}

// Gets the number of vectors in the data set
inline int CMultivariateRegressionOverUnivariate::GetVectorCount() const
{
	return inner->GetVectorCount();
}

// Gets all vectors from the data set as a matrix
CSparseFloatMatrixDesc CMultivariateRegressionOverUnivariate::GetMatrix() const
{
	return inner->GetMatrix();
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
		int nullWeightElementsCount = 0;
		for( int i = 0; i < originalVectorCount - nullWeightElementsCount; ) {
			int iScanned = i + nullWeightElementsCount;
			if( _inner->GetVectorWeight( iScanned ) == 0 ) {
				++nullWeightElementsCount;
			} else {
				notNullWeightElementsIndices.Add( iScanned );
				++i;
			}
		}

		// fill nullWeightElementsMap, and then adjust local MatrixDesc if needed
		viewMatrixDesc = _inner->GetMatrix();
		viewMatrixDesc.Height -= nullWeightElementsCount;
		NeoAssert( viewMatrixDesc.Height == notNullWeightElementsIndices.Size() );

		if( nullWeightElementsCount != 0 && viewMatrixDesc.Height > 0 ) {
			// we are going to remap some elements, so let's create our own arrays of pointers
			viewMatrixDesc.PointerB = static_cast<int*>(
				ALLOCATE_MEMORY( CurrentMemoryManager, viewMatrixDesc.Height * sizeof( int ) ) );
			viewMatrixDesc.PointerE = static_cast<int*>(
				ALLOCATE_MEMORY( CurrentMemoryManager, viewMatrixDesc.Height * sizeof( int ) ) );

			for( int i = 0; i < viewMatrixDesc.Height; ++i ) {
				int originalIndex = notNullWeightElementsIndices[i];
				viewMatrixDesc.PointerB[i] = _inner->GetMatrix().PointerB[originalIndex];
				viewMatrixDesc.PointerE[i] = _inner->GetMatrix().PointerE[originalIndex];
			}
		}
	}
}

CProblemNotNullWeightsView::~CProblemNotNullWeightsView()
{
	if( GetVectorCount() > 0 && inner->GetVectorCount() != GetVectorCount() ) {
		NeoAssert( GetVectorCount() != 0 );

		CurrentMemoryManager::Free( viewMatrixDesc.PointerB );
		CurrentMemoryManager::Free( viewMatrixDesc.PointerE );
	}
}

/////////////////////////////////////////////////////////////////////////////////////////

} // namespace NeoML

