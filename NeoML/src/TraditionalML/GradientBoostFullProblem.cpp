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

#include <GradientBoostFullProblem.h>
#include <GradientBoostThreadTask.h>
#include <NeoMathEngine/ThreadPool.h>

namespace NeoML {

namespace {

class CGBoostSortFeaturesThreadTask : public IGradientBoostThreadTask {
public:
	// Create a task
	CGBoostSortFeaturesThreadTask( IThreadPool& threadPool,
			const CArray<int>& curFeaturePos,
			const CArray<bool>& isUsedFeatureBinary,
			const CArray<int>& featureValueCount,
			const CArray<int>& featurePos,
			CArray<CFloatVectorElement>& featureValues ) :
		IGradientBoostThreadTask( threadPool ),
		CurFeaturePos( curFeaturePos ),
		IsUsedFeatureBinary( isUsedFeatureBinary ),
		FeatureValueCount( featureValueCount ),
		FeaturePos( featurePos ),
		FeatureValues( featureValues )
	{}
protected:
	// The size of parallelization, max number of elements to perform
	int ParallelizeSize() const override { return CurFeaturePos.Size(); }
	// Run the process in a separate thread
	void Run( int threadIndex, int startIndex, int count ) override;

	const CArray<int>& CurFeaturePos;
	const CArray<bool>& IsUsedFeatureBinary;
	const CArray<int>& FeatureValueCount;
	const CArray<int>& FeaturePos;
	CArray<CFloatVectorElement>& FeatureValues;
};

void CGBoostSortFeaturesThreadTask::Run( int /*threadIndex*/, int startIndex, int count )
{
	const int endIndex = startIndex + count;
	for( int index = startIndex; index < endIndex; ++index ) {
		if( CurFeaturePos[index] != NotFound && !IsUsedFeatureBinary[index] ) {
			FeatureValues[CurFeaturePos[index]].Index = NotFound;
			FeatureValues[CurFeaturePos[index]].Value = 0;
			// Sorting by feature value
			AscendingByMember<CFloatVectorElement, float, &CFloatVectorElement::Value> comparator;
			QuickSort( &FeatureValues[FeaturePos[index]], FeatureValueCount[index], &comparator );
		}
	}
}

} // namespace

//-------------------------------------------------------------------------------------------------------------

CGradientBoostFullProblem::CGradientBoostFullProblem( int _threadCount,
		const IMultivariateRegressionProblem* _baseProblem,
		const CArray<int>& _usedVectors, const CArray<int>& _usedFeatures, const CArray<int>& _featureNumbers ) :
	threadPool( CreateThreadPool( _threadCount ) ),
	baseProblem( _baseProblem ),
	usedVectors( _usedVectors ),
	usedFeatures( _usedFeatures ),
	featureNumbers( _featureNumbers )
{
	NeoAssert( threadPool != nullptr );
	NeoAssert( baseProblem != nullptr );
}

CGradientBoostFullProblem::~CGradientBoostFullProblem()
{
	delete threadPool;
}

void CGradientBoostFullProblem::Update()
{
	featureValueCount.DeleteAll();
	featureValueCount.Add( 0, usedFeatures.Size() );
	isUsedFeatureBinary.DeleteAll();
	isUsedFeatureBinary.Add( true, usedFeatures.Size() );

	CFloatMatrixDesc matrix = baseProblem->GetMatrix();
	NeoAssert( matrix.Height == baseProblem->GetVectorCount() );
	NeoAssert( matrix.Width == baseProblem->GetFeatureCount() );

	for( int i = 0; i < usedVectors.Size(); i++ ) {
		CFloatVectorDesc vector;
		matrix.GetRow( usedVectors[i], vector );
		for( int j = 0; j < vector.Size; j++ ) {
			const int link = featureNumbers[vector.Indexes == nullptr ? j : vector.Indexes[j]];
			if( link != NotFound && vector.Values[j] != 0.0 ) {
				if( vector.Values[j] != 1.0 ) {
					isUsedFeatureBinary[link] = false;
				}
				featureValueCount[link]++;
			}
		}
	}

	for( int i = 0; i < isUsedFeatureBinary.Size(); i++ ) {
		if( !isUsedFeatureBinary[i] ) {
			featureValueCount[i]++; // there's also the zero value
		}
	}

	featurePos.Empty();
	featurePos.Add( NotFound, usedFeatures.Size() );

	int curDataSize = 0;
	int curBinaryDataSize = 0;
	for( int i = 0; i < usedFeatures.Size(); i++ ) {
		if( isUsedFeatureBinary[i] ) {
			featurePos[i] = curBinaryDataSize;
			curBinaryDataSize += featureValueCount[i];
		} else {
			featurePos[i] = curDataSize;
			curDataSize += featureValueCount[i];
		}
	}

	featureValues.SetSize( curDataSize );
	binaryFeatureValues.SetSize( curBinaryDataSize );

	CArray<int> curFeaturePos;
	featurePos.CopyTo( curFeaturePos );

	for( int i = 0; i < usedVectors.Size(); i++ ) {
		CFloatVectorDesc vector;
		matrix.GetRow( usedVectors[i], vector );
		for( int j = 0; j < vector.Size; j++ ) {
			const int link = featureNumbers[vector.Indexes == nullptr ? j : vector.Indexes[j]];
			if( link != NotFound && vector.Values[j] != 0.0 ) {
				if( isUsedFeatureBinary[link] ) {
					// A binary feature
					binaryFeatureValues[curFeaturePos[link]] = i;
					curFeaturePos[link]++;
				} else {
					// A continuous feature
					featureValues[curFeaturePos[link]].Index = i;
					featureValues[curFeaturePos[link]].Value = static_cast<float>( vector.Values[j] );
					curFeaturePos[link]++;
				}
			}
		}
	}

	CGBoostSortFeaturesThreadTask( *threadPool, curFeaturePos, isUsedFeatureBinary,
		featureValueCount, featurePos, featureValues ).ParallelRun();
}

int CGradientBoostFullProblem::GetTotalVectorCount() const
{
	return baseProblem->GetVectorCount();
}

int CGradientBoostFullProblem::GetTotalFeatureCount() const
{
	return baseProblem->GetFeatureCount();
}

bool CGradientBoostFullProblem::IsUsedFeatureBinary( int feature ) const
{
	return isUsedFeatureBinary[feature];
}

const void* CGradientBoostFullProblem::GetUsedFeatureDataPtr( int feature ) const
{
	if( featureValueCount[feature] == 0 ) {
		// No values for this feature
		return 0;
	}

	if( isUsedFeatureBinary[feature] ) {
		return &binaryFeatureValues[featurePos[feature]];
	}
	return &featureValues[featurePos[feature]];
}

int CGradientBoostFullProblem::GetUsedFeatureDataSize( int feature ) const
{
	return featureValueCount[feature];
}

} // namespace NeoML
