/* Copyright © 2017-2023 ABBYY

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
#include <GradientBoostThreadTask.h>
#include <NeoMathEngine/ThreadPool.h>

namespace NeoML {

namespace {

// Abstract base class
class IFeaturesThreadTask : public IGradientBoostThreadTask {
public:
	using TFeatureValue = CGradientBoostFastHistProblem::CFeatureValue;
protected:
	// Create a task
	IFeaturesThreadTask( IThreadPool& threadPool,
			CArray<CArray<TFeatureValue>>& featureValues ) :
		IGradientBoostThreadTask( threadPool ),
		FeatureValues( featureValues )
	{}
	// The size of parallelization, max number of elements to perform
	int ParallelizeSize() const override { return FeatureValues.Size(); }
	// Run the process in a separate thread
	void Run( int /*threadIndex*/, int startIndex, int count ) override final;
	// Run on each problem's element separately
	virtual void RunOnElement( int index ) = 0;

	CArray<CArray<TFeatureValue>>& FeatureValues;
};

void IFeaturesThreadTask::Run( int /*threadIndex*/, int startIndex, int count )
{
	const int endIndex = startIndex + count;
	for( int index = startIndex; index < endIndex; ++index ) {
		RunOnElement( index );
	}
}

//-------------------------------------------------------------------------------------------------------------

// Sorting and merging the same feature values
class CSortAndMergeFeaturesThreadTask : public IFeaturesThreadTask {
public:
	// Create a task
	CSortAndMergeFeaturesThreadTask( IThreadPool& threadPool,
			CArray<CArray<TFeatureValue>>& featureValues ) :
		IFeaturesThreadTask( threadPool, featureValues )
	{}
protected:
	// Run on each problem's element separately
	void RunOnElement( int index ) override;
};

void CSortAndMergeFeaturesThreadTask::RunOnElement( int index )
{
	FeatureValues[index].QuickSort<AscendingByMember<TFeatureValue, float, &TFeatureValue::Value>>();
	int size = 1;
	for( int j = 1; j < FeatureValues[index].Size(); ++j ) {
		if( FeatureValues[index][j].Value == FeatureValues[index][size - 1].Value ) {
			FeatureValues[index][size - 1].Weight += FeatureValues[index][j].Weight;
		} else {
			++size;
			FeatureValues[index][size - 1] = FeatureValues[index][j];
		}
	}
	FeatureValues[index].SetSize( size );
}

//-------------------------------------------------------------------------------------------------------------

// Compresses the values of each feature so that there are no more than maxBins different values
class CCompressFeaturesThreadTask : public IFeaturesThreadTask {
public:
	// Create a task
	CCompressFeaturesThreadTask( IThreadPool& threadPool,
			CArray<CArray<TFeatureValue>>& featureValues, int maxBins, double totalWeight ) :
		IFeaturesThreadTask( threadPool, featureValues ),
		MaxBins( maxBins ),
		TotalWeight( totalWeight )
	{ NeoAssert( MaxBins > 1 ); } // otherwise there can be no split
protected:
	// Run on each problem's element separately
	void RunOnElement( int index ) override;

	const int MaxBins;
	const double TotalWeight;
};

void CCompressFeaturesThreadTask::RunOnElement( int index )
{
	CArray<TFeatureValue>& currFeatureValues = FeatureValues[index];
	if( currFeatureValues.Size() <= MaxBins ) {
		return;
	}

	// Always keep the minimum and the maximum value for each feature
	if( MaxBins == 2 ) {
		currFeatureValues[1] = currFeatureValues.Last();
		currFeatureValues.SetSize( 2 );
		return;
	}
	// Always keep the minimum and the maximum value for each feature
	const double weight = TotalWeight - currFeatureValues.First().Weight - currFeatureValues.Last().Weight;
	const int n = MaxBins - 2;
	NeoAssert( n > 0 );
	const double maxItemWeight = weight / n;

	// Grouping the rest of the values by weight
	int size = 1;
	double sumWeight = 0;
	for( int j = 1; j < currFeatureValues.Size() - 1; ++j ) {
		if( sumWeight + currFeatureValues[j].Weight >= size * maxItemWeight ) {
			currFeatureValues[size] = currFeatureValues[j];
			++size;
		}
		sumWeight += currFeatureValues[j].Weight;
	}
	currFeatureValues[size] = currFeatureValues.Last();
	++size;
	currFeatureValues.SetSize( size );
	NeoAssert( size <= MaxBins );
}

} // namespace

//-------------------------------------------------------------------------------------------------------------

CGradientBoostFastHistProblem::CGradientBoostFastHistProblem( int threadCount, int maxBins,
		const IMultivariateRegressionProblem& baseProblem,
		const CArray<int>& _usedVectors, const CArray<int>& _usedFeatures ) :
	threadPool( CreateThreadPool( threadCount ) ),
	usedVectors( _usedVectors ),
	usedFeatures( _usedFeatures )
{
	NeoAssert( threadPool != nullptr );
	CFloatMatrixDesc matrix = baseProblem.GetMatrix();
	NeoAssert( matrix.Height == baseProblem.GetVectorCount() );
	NeoAssert( matrix.Width == baseProblem.GetFeatureCount() );
	// Initialize features data
	initializeFeatureInfo( maxBins, matrix, baseProblem );

	// Build vector data
	buildVectorData( matrix );
}

CGradientBoostFastHistProblem::~CGradientBoostFastHistProblem()
{
	delete threadPool;
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
void CGradientBoostFastHistProblem::initializeFeatureInfo( int maxBins, const CFloatMatrixDesc& matrix,
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
		CFloatVectorDesc vector;
		matrix.GetRow( i, vector );
		const double vectorWeight = baseProblem.GetVectorWeight( i );

		for( int j = 0; j < vector.Size; j++ ) {
			if( vector.Values[j] != 0.0 ) {
				++totalElementCount;
				const int index = vector.Indexes == nullptr ? j : vector.Indexes[j];
				if( featureValues[index].IsEmpty()
					|| featureValues[index].Last().Value != vector.Values[j] )
				{
					CFeatureValue newValue{};
					newValue.Value = vector.Values[j];
					newValue.Weight = vectorWeight;
					featureValues[index].Add( newValue );
				} else {
					featureValues[index].Last().Weight += vectorWeight;
				}
				featureWeights[index] += vectorWeight;
			}
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
	CSortAndMergeFeaturesThreadTask( *threadPool, featureValues ).ParallelRun();

	CCompressFeaturesThreadTask( *threadPool, featureValues, maxBins, totalWeight ).ParallelRun();

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

// Builds an array with vector data
void CGradientBoostFastHistProblem::buildVectorData( const CFloatMatrixDesc& matrix )
{
	const int vectorCount = matrix.Height;

	vectorPtr.SetBufferSize( vectorCount + 1 );
	int curVectorPtr = 0;
	for( int i = 0; i < vectorCount; i++ ) {
		vectorPtr.Add( curVectorPtr );
		CFloatVectorDesc vector;
		matrix.GetRow( i, vector );

		for( int j = 0; j < vector.Size; j++ ) {
			if( vector.Values[j] != 0.0 ) {
				++curVectorPtr;
				const int index = vector.Indexes == nullptr ? j : vector.Indexes[j];
				float* valuePtr = cuts.GetPtr() + featurePos[index]; // the pointer to this feature values
				int valueCount = featurePos[index + 1] - featurePos[index]; // the number of different values for the feature
				// Now we get the bin into which the current value falls
				int pos = FindInsertionPoint<float, Ascending<float>, float>( vector.Values[j], valuePtr, valueCount );
				if( pos > 0 && *( valuePtr + pos - 1 ) == vector.Values[j] ) {
					pos--;
				}
				vectorData.Add( featurePos[index] + pos );
			}
		}
	}

	vectorPtr.Add( curVectorPtr );
}

} // namespace NeoML
