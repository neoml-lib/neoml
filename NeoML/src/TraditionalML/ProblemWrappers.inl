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
inline CSparseFloatMatrixDesc CMultivariateRegressionOverUnivariate::GetMatrix() const
{
	return inner->GetMatrix();
}

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
// CMultivariateRegressionProblemNotNullWeightsView 

// The number of features
inline int CMultivariateRegressionProblemNotNullWeightsView::GetFeatureCount() const
{
	return inner->GetFeatureCount();
}

// The number of vectors in the input data set
inline int CMultivariateRegressionProblemNotNullWeightsView::GetVectorCount() const
{
	return ViewMatrixDesc.Height;
}

// Gets all input vectors as a matrix
inline CSparseFloatMatrixDesc CMultivariateRegressionProblemNotNullWeightsView::GetMatrix() const
{
	return ViewMatrixDesc;
}

// The vector weight
inline double CMultivariateRegressionProblemNotNullWeightsView::GetVectorWeight( int index ) const
{
	NeoPresume( index < GetVectorCount() );

	return inner->GetVectorWeight( CalculateOriginalIndex( index ) );
}

// The length of the function value vector
inline int CMultivariateRegressionProblemNotNullWeightsView::GetValueSize() const
{
	return inner->GetValueSize();
}


// The value of the function on the vector with the given index
inline CFloatVector CMultivariateRegressionProblemNotNullWeightsView::GetValue( int index ) const
{
	NeoPresume( index < GetVectorCount() );

	return inner->GetValue( CalculateOriginalIndex( index ) );
}

/////////////////////////////////////////////////////////////////////////////////////////

} // namespace NeoML
