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

#pragma once

#include <NeoML/TraditionalML/DecisionTree.h>
#include <DecisionTreeNodeStatisticBase.h>
#include <DecisionTreeNodeBase.h>

namespace NeoML {

// The statistics for a vector set
class CVectorSetClassificationStatistic {
public:
	explicit CVectorSetClassificationStatistic( int classCount );
	explicit CVectorSetClassificationStatistic( const CVectorSetClassificationStatistic& other );

	// Adds vectors
	void AddVectorSet( int count, int classIndex, double weight );

	// Deletes vectors
	void SubVectorSet( int count, int classIndex, double weight );

	// Deletes all accumulated data
	void Erase();

	// Calculates the specified criterion according to the current data
	double CalcCriterion( CDecisionTree::TSplitCriterion criterion ) const;

	// The number of vectors in the set
	int TotalCount() const { return totalCount; }

	// Total weight of the vectors
	double TotalWeight() const { return totalWeight; }

	// The vector weights for each class
	const CArray<double>& Weights() const { return weights; }

	// The number of vectors for each class
	const CArray<int>& Counts() const { return counts; }

	// Object size
	size_t GetSize() const { return sizeof(CVectorSetClassificationStatistic) + weights.BufferSize() * sizeof(double); }

private:
	int totalCount; // the total number of vectors
	double totalWeight; // the total vector weight
	CArray<double> weights; // the vector weights for each class
	CArray<int> counts; // the number of vectors for each class
};

inline CVectorSetClassificationStatistic::CVectorSetClassificationStatistic( int classCount ) :
	totalCount( 0 ),
	totalWeight( 0.0 )
{
	weights.Add( 0.0, classCount );
	counts.Add( 0, classCount );
}

inline CVectorSetClassificationStatistic::CVectorSetClassificationStatistic( const CVectorSetClassificationStatistic& other ) :
	totalCount( other.totalCount ),
	totalWeight( other.totalWeight )
{
	other.weights.CopyTo( weights );
	other.counts.CopyTo( counts );
}

inline void CVectorSetClassificationStatistic::AddVectorSet( int valueCount, int valueClass, double valueWeight )
{
	totalCount += valueCount;
	totalWeight += valueWeight;
	weights[valueClass] += valueWeight;
	counts[valueClass] += valueCount;
}

inline void CVectorSetClassificationStatistic::SubVectorSet( int valueCount, int valueClass, double valueWeight )
{
	totalCount -= valueCount;
	totalWeight -= valueWeight;
	weights[valueClass] -= valueWeight;
	counts[valueClass] -= valueCount;
}

inline void CVectorSetClassificationStatistic::Erase()
{
	totalCount = 0;
	totalWeight = 0;
	int size = weights.Size();
	weights.Empty();
	weights.Add( 0.0, size );
	counts.Empty();
	counts.Add( 0, size );
}

inline double CVectorSetClassificationStatistic::CalcCriterion( CDecisionTree::TSplitCriterion criterion ) const
{
	double res = 0;
	if( criterion == CDecisionTree::SC_GiniImpurity ) {
		for( int i = 0; i < weights.Size(); i++ ) {
			double fraction = weights[i] / totalWeight;
			res += fraction * ( 1 - fraction );
		}
		return res;
	} else if( criterion == CDecisionTree::SC_InformationGain ) {
		for( int i = 0; i < weights.Size(); i++ ) {
			if( weights[i] > 0 ) {
				double p = weights[i] / totalWeight;
				res += p * log( p );
			}
		}
		return -res;
	} else {
		NeoAssert( false );
		return 0;
	}
}

//---------------------------------------------------------------------------------------------------------

// The statistics accumulated in a node
class CClassificationStatistics : public CDecisionTreeNodeStatisticBase {
public:
	explicit CClassificationStatistics( CDecisionTreeNodeBase* node, const IProblem& problem, const CArray<int>& usedFeatures );

	// CDecisionTreeNodeStatisticBase interface methods
	virtual void AddVector( int index, const CSparseFloatVectorDesc& vector );
	virtual void Finish();
	virtual size_t GetSize() const;
	virtual bool GetSplit( CDecisionTree::CParams param,
		bool& isDiscrete, int& featureIndex, CArray<double>& values, double& criterioValue ) const;
	virtual double GetPredictions( CArray<double>& predictions ) const;
	virtual int GetVectorsCount() const { return totalStatistics.TotalCount(); }
	virtual CDecisionTreeNodeBase& GetNode() const { return *node; }

private:
	// Sampling interval
	struct CInterval {
		double Begin;
		double End;
		int Class;
		int Count;
		double Weight;
	};

	typedef CFastArray<CInterval, 20> CIntervalArray; // 20 is used for binary classification

	const int classCount; // the number of classes
	const CPtr<CDecisionTreeNodeBase> node; // the node for which statistics are accumulated
	const CPtr<const IProblem> problem; // the problem
	CArray<int> usedFeatures; // the features used
	CArray<int> usedFeatureNumber; // the number of the current feature
	CVectorSetClassificationStatistic totalStatistics; // the whole subset statistics
	CArray<CVectorSetClassificationStatistic> featureStatistics; // the statistics for each feature
	CArray<CIntervalArray> discretizationIntervals; // the sampling intervals

	void addValue( int index, double value, int count, int classIndex, double weight );
	void mergeIntervals( int discretizationValue, CIntervalArray& intervals );
	void mergeOverlappingIntervals( CIntervalArray& intervals );
	void mergeIntervalsByWeight( int left, int right, int resultIntervalCount, CIntervalArray& intervals );
	void closeIntervals( double begin, double end, CArray<int>& link, CIntervalArray& intervals );
	void calcSplitCriterion( CDecisionTree::CParams param, bool isDiscrete,
		const CIntervalArray& curItems, CArray<double>& splitValues );
	double calcContinuousSplitCriterion( CDecisionTree::CParams param,
		const CIntervalArray& intervals, const CVectorSetClassificationStatistic& total, CArray<double>& splitValues ) const;
	double calcDiscreteSplitCriterion( CDecisionTree::CParams param,
		const CIntervalArray& intervals, const CVectorSetClassificationStatistic& total, CArray<double>& splitValues ) const;
	static bool isEqual( const CInterval& interval1, const CInterval& interval2 );
	static double sumWeight( const CIntervalArray& intervals, int left, int right );

	CClassificationStatistics( const CClassificationStatistics& );
	CClassificationStatistics& operator=( const CClassificationStatistics& );
};

} // namespace NeoML
