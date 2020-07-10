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

namespace NeoML {

/////////////////////////////////////////////////////////////////////////////////////////
// CMultivariateRegressionOverUnivariate

// Gets the vector weight
inline double CMultivariateRegressionOverUnivariate::GetVectorWeight( int index ) const
{
	return inner->GetVectorWeight( index );
}

// Gets the length of the function value vector
inline int CMultivariateRegressionOverUnivariate::GetValueSize() const
{
	return 1;
}

/////////////////////////////////////////////////////////////////////////////////////////
// CMultivariateRegressionOverClassification

// Gets the number of features
inline int CMultivariateRegressionOverClassification::GetFeatureCount() const
{
	return inner->GetFeatureCount();
}

// Gets the number of vectors in the data set
inline int CMultivariateRegressionOverClassification::GetVectorCount() const
{
	return inner->GetVectorCount();
}

// Gets all vectors from the data set as a matrix
inline CSparseFloatMatrixDesc CMultivariateRegressionOverClassification::GetMatrix() const
{
	return inner->GetMatrix();
}

// Gets the vector weight
inline double CMultivariateRegressionOverClassification::GetVectorWeight( int index ) const
{
	return inner->GetVectorWeight( index );
}

// Gets the length of the function value vector
inline int CMultivariateRegressionOverClassification::GetValueSize() const
{
	return classValues.Size();
}

/////////////////////////////////////////////////////////////////////////////////////////
// CMultivariateRegressionOverBinaryClassification

// Gets the number of features
inline int CMultivariateRegressionOverBinaryClassification::GetFeatureCount() const
{
	return inner->GetFeatureCount();
}

// Gets the number of vectors in the data set
inline int CMultivariateRegressionOverBinaryClassification::GetVectorCount() const
{
	return inner->GetVectorCount();
}

// Gets all vectors from the data set as a matrix
inline CSparseFloatMatrixDesc CMultivariateRegressionOverBinaryClassification::GetMatrix() const
{
	return inner->GetMatrix();
}

// Gets the vector weight
inline double CMultivariateRegressionOverBinaryClassification::GetVectorWeight( int index ) const
{
	return inner->GetVectorWeight( index );
}

// Gets the length of the function value vector
inline int CMultivariateRegressionOverBinaryClassification::GetValueSize() const
{
	return 1;
}

/////////////////////////////////////////////////////////////////////////////////////////
// CProblemNotNullWeightsView 

// Gets the number of classes
inline int CProblemNotNullWeightsView::GetClassCount() const
{
	return inner->GetClassCount();
}

// Gets the number of features
inline int CProblemNotNullWeightsView::GetFeatureCount() const
{
	return inner->GetFeatureCount();
}

// Indicates if the specified feature is discrete
inline bool CProblemNotNullWeightsView::IsDiscreteFeature( int index ) const
{
	return inner->IsDiscreteFeature( index );
}

// Gets the number of vectors in the data set
inline int CProblemNotNullWeightsView::GetVectorCount() const
{
	return viewMatrixDesc.Height;
}

// The correct class number for a vector with a given index in [0, GetClassCount())
inline int CProblemNotNullWeightsView::GetClass( int index ) const
{
	NeoPresume( index < viewMatrixDesc.Height );

	return inner->GetClass( calculateOriginalIndex( index ) );
}

// Gets all vectors from the data set as a matrix
inline CSparseFloatMatrixDesc CProblemNotNullWeightsView::GetMatrix() const
{
	return viewMatrixDesc;
}

// Gets the vector weight
inline double CProblemNotNullWeightsView::GetVectorWeight( int index ) const
{
	NeoPresume( index < viewMatrixDesc.Height );

	return inner->GetVectorWeight( calculateOriginalIndex( index ) );
}

// calculate the index as if we had the matrix without null weighted elements
inline int CProblemNotNullWeightsView::calculateOriginalIndex( int viewedIndex ) const
{
	return notNullWeightElementsIndices[viewedIndex];
}

} // namespace NeoML
