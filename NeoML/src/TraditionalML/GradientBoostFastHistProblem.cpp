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

#include <GradientBoostFastHistProblem.h>
#include <NeoMathEngine/OpenMP.h>

namespace NeoML {

CGradientBoostFastHistProblem::CGradientBoostFastHistProblem( int threadCount, int maxBins,
		const IMultivariateRegressionProblem& baseProblem,
		const CArray<int>& _usedVectors, const CArray<int>& _usedFeatures ) :
	usedVectors( _usedVectors ),
	usedFeatures( _usedFeatures )
{
	CSparseFloatMatrixDesc matrix = baseProblem.GetMatrix();
	NeoAssert( matrix.Height == baseProblem.GetVectorCount() );
	NeoAssert( matrix.Width == baseProblem.GetFeatureCount() );
	// Initialize features data
	initializeFeatureInfo( threadCount, maxBins, matrix, baseProblem );

	// Build vector data
	buildVectorData( matrix );
}

const int* CGradientBoostFastHistProblem::GetUsedVectorDataPtr( int index ) const
{
	NeoAssert( index >= 0 );
	NeoAssert( index < usedVectors.Size() );

	return vectorData.GetPtr() + vectorPtr[usedVectors[index]];
}

int CGradientBoostFastHistProblem::GetUsedVectorDataSize( int index ) const
{
	NeoAssert( index >= 0 );
	NeoAssert( index < usedVectors.Size() );

	return vectorPtr[usedVectors[index] + 1] - vectorPtr[usedVectors[index]];
}

// Initializes the feature values
void CGradientBoostFastHistProblem::initializeFeatureInfo( int threadCount, int maxBins, const CSparseFloatMatrixDesc& matrix,
	const IMultivariateRegressionProblem& baseProblem )
{
	const int vectorCount = baseProblem.GetVectorCount();
	const int featureCount = baseProblem.GetFeatureCount();

	CArray< CArray<CFeatureValue> > featureValues; // the values of all features
	featureValues.SetSize( featureCount );

	CArray<double> featureWeights; // total weight of all vectors for which the current feature is not 0
	featureWeights.Add( 0.0, featureCount );
	double totalWeight = 0.0; // total weight of all vectors
	int totalElementCount = 0; // total number of non-zero elements

	// Adding the non-zero values
	for( int i = 0; i < vectorCount; i++ ) {
		CSparseFloatVectorDesc vector;
		matrix.GetRow( i, vector );
		const double vectorWeight = baseProblem.GetVectorWeight( i );

		totalElementCount += vector.Size;
		for( int j = 0; j < vector.Size; j++ ) {
			if( featureValues[vector.Indexes[j]].IsEmpty()
				|| featureValues[vector.Indexes[j]].Last().Value != vector.Values[j] )
			{
				CFeatureValue newValue;
				newValue.Value = vector.Values[j];
				newValue.Weight = vectorWeight;
				featureValues[vector.Indexes[j]].Add( newValue );
			} else {
				featureValues[vector.Indexes[j]].Last().Weight += vectorWeight;
			}
			featureWeights[vector.Indexes[j]] += vectorWeight;
		}
		totalWeight += vectorWeight;
	}

	vectorData.SetBufferSize( totalElementCount );

	// Adding the zero values
	for( int i = 0; i < featureValues.Size(); i++ ) {
		CFeatureValue newValue;
		newValue.Value = 0;
		newValue.Weight = totalWeight - featureWeights[i];
		featureValues[i].Add( newValue );
	}

	// Sorting and merging the same values
	NEOML_OMP_FOR_NUM_THREADS( threadCount )
	for( int i = 0; i < featureValues.Size(); i++ ) {
		featureValues[i].QuickSort< AscendingByMember<CFeatureValue, float, &CFeatureValue::Value> >();
		int size = 1;
		for( int j = 1; j < featureValues[i].Size(); j++ ) {
			if( featureValues[i][j].Value == featureValues[i][size - 1].Value ) {
				featureValues[i][size - 1].Weight += featureValues[i][j].Weight;
			} else {
				size++;
				featureValues[i][size - 1] = featureValues[i][j];
			}
		}
		featureValues[i].SetSize( size );
	}

	compressFeatureValues( threadCount, maxBins, totalWeight, featureValues );

	// Initializing the internal arrays
	nullValueIds.Add( NotFound, featureValues.Size() );
	featurePos.SetBufferSize( featureValues.Size() );
	int curPos = 0;
	for( int i = 0; i < featureValues.Size(); i++ ) {
		featurePos.Add( curPos );
		featureIndexes.Add( i, featureValues[i].Size() );
		curPos += featureValues[i].Size();

		for( int j = 0; j < featureValues[i].Size(); j++ ) {
			const float next = j + 1 == featureValues[i].Size() ? featureValues[i][j].Value : featureValues[i][j + 1].Value;
			cuts.Add( ( featureValues[i][j].Value + next ) / 2 );
			if( nullValueIds[i] == NotFound && 0 <= cuts.Last() ) {
				nullValueIds[i] = cuts.Size() - 1;
			}
		}
	}
	featurePos.Add( curPos );
}

// Compresses the values of each feature so that there are no more than maxBins different values
void CGradientBoostFastHistProblem::compressFeatureValues( int threadCount, int maxBins, double totalWeight,
	CArray< CArray<CFeatureValue> >& featureValues )
{
	NeoAssert( maxBins > 1 ); // otherwise there can be no split

	NEOML_OMP_FOR_NUM_THREADS( threadCount )
	for( int i = 0; i < featureValues.Size(); i++ ) {
		CArray<CFeatureValue>& currFeatureValues = featureValues[i];

		if( currFeatureValues.Size() <= maxBins ) {
			continue;
		}

		// Always keep the minimum and the maximum value for each feature
		if( maxBins == 2 ) {
			currFeatureValues[1] = currFeatureValues.Last();
			currFeatureValues.SetSize( 2 );
			continue;
		}
		// Always keep the minimum and the maximum value for each feature
		const double weight = totalWeight - currFeatureValues.First().Weight - currFeatureValues.Last().Weight;
		const int n = maxBins - 2;
		NeoAssert( n > 0 );
		const double maxItemWeight = weight / n;

		// Grouping the rest of the values by weight
		int size = 1;
		double sumWeight = 0;
		for( int j = 1; j < currFeatureValues.Size() - 1; j++ ) {
			if( sumWeight + currFeatureValues[j].Weight >= size * maxItemWeight ) {
				currFeatureValues[size] = currFeatureValues[j];
				size++;
			}
			sumWeight += currFeatureValues[j].Weight;
		}
		currFeatureValues[size] = currFeatureValues.Last();
		size++;
		currFeatureValues.SetSize( size );
		NeoAssert( size <= maxBins );
	}
}

// Builds an array with vector data
void CGradientBoostFastHistProblem::buildVectorData( const CSparseFloatMatrixDesc& matrix )
{
	const int vectorCount = matrix.Height;
	
	vectorPtr.SetBufferSize( vectorCount + 1 );
	int curVectorPtr = 0;
	for( int i = 0; i < vectorCount; i++ ) {
		vectorPtr.Add( curVectorPtr );
		CSparseFloatVectorDesc vector;
		matrix.GetRow( i, vector );

		for( int j = 0; j < vector.Size; j++ ) {
			float* valuePtr = cuts.GetPtr() + featurePos[vector.Indexes[j]]; // the pointer to this feature values
			int valueCount = featurePos[vector.Indexes[j] + 1] - featurePos[vector.Indexes[j]]; // the number of different values for the feature
			// Now we get the bin into which the current value falls
			int pos = FindInsertionPoint<float, Ascending<float>, float>( vector.Values[j], valuePtr, valueCount );
			if( pos > 0 && *(valuePtr + pos - 1) == vector.Values[j] ) {
				pos--;
			}
			vectorData.Add( featurePos[vector.Indexes[j]] + pos );
		}
		curVectorPtr += vector.Size;
	}

	vectorPtr.Add( curVectorPtr );
}

} // namespace NeoML
