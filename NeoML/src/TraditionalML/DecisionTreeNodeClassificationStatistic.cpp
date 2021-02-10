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

#include <DecisionTreeNodeClassificationStatistic.h>

namespace NeoML {

// The statistics gathering coefficients
const int SmallCoef = 4;
const int BigCoef = 10;

CClassificationStatistics::CClassificationStatistics( CDecisionTreeNodeBase* _node, const IProblem& _problem, const CArray<int>& _usedFeatures ) :
	classCount( _problem.GetClassCount() ),
	node( _node ),
	problem( &_problem ),
	totalStatistics( _problem.GetClassCount() )
{
	_usedFeatures.CopyTo( usedFeatures );
	usedFeatureNumber.Add( NotFound, problem->GetFeatureCount() );

	featureStatistics.SetBufferSize( usedFeatures.Size() );
	for( int i = 0; i < usedFeatures.Size(); i++ ) {
		usedFeatureNumber[usedFeatures[i]] = i;
		featureStatistics.Add( CVectorSetClassificationStatistic( problem->GetClassCount() ) );
	}
	discretizationIntervals.SetSize( usedFeatures.Size() );
}

void CClassificationStatistics::AddVector( int index, const CFloatVectorDesc& vector )
{
	NeoAssert( problem != 0 );
	const double weight = problem->GetVectorWeight( index );
	const int classIndex = problem->GetClass( index );

	for( int i = 0; i < vector.Size; i++ ) {
		if( vector.Values[i] != 0.0 ) {
			const int curIndex = vector.GetPosIndex( i );
			if( usedFeatureNumber[curIndex] != NotFound ) {
				addValue( usedFeatureNumber[curIndex], vector.Values[i], 1, classIndex, weight );
				featureStatistics[usedFeatureNumber[curIndex]].AddVectorSet( 1, classIndex, weight );
			}
		}
	}

	totalStatistics.AddVectorSet( 1, classIndex, weight );
}

void CClassificationStatistics::Finish()
{
	// We need also to add zero values for the features
	const CArray<double>& totalWeights = totalStatistics.Weights();
	const CArray<int>& totalCounts = totalStatistics.Counts();

	for( int i = 0; i < usedFeatures.Size(); i++ ) {
		const CArray<double>& weights = featureStatistics[i].Weights();
		const CArray<int>& counts = featureStatistics[i].Counts();

		for( int j = 0; j < classCount; j++ ) {
			if( totalCounts[j] - counts[j] > 0 ) {
				addValue( i, 0, totalCounts[j] - counts[j], j, totalWeights[j] - weights[j] );
			}
		}
		mergeIntervals( problem->GetDiscretizationValue( usedFeatures[i] ), discretizationIntervals[i] );
	}
}

size_t CClassificationStatistics::GetSize() const
{
	size_t result = usedFeatures.BufferSize() * sizeof(int) + usedFeatureNumber.BufferSize() * sizeof(int)
		+ totalStatistics.GetSize();

	for( int i = 0; i < discretizationIntervals.Size(); i++ ) {
		result += featureStatistics[i].GetSize();
		result += discretizationIntervals[i].BufferSize() * sizeof(CInterval);
	}

	result += featureStatistics.BufferSize() * sizeof(CVectorSetClassificationStatistic);
	result += discretizationIntervals.BufferSize() * sizeof(CIntervalArray);
	return result;
}

bool CClassificationStatistics::GetSplit( CDecisionTreeTrainingModel::CParams param,
	bool& isDiscrete, int& featureIndex, CArray<double>& values, double& criterionValue ) const
{
	// Choose the feature so that splitting by it will give the smallest criterion value
	// If that is smaller than the whole subset criterion value, splitting is successful
	criterionValue = totalStatistics.CalcCriterion( param.SplitCriterion );
	featureIndex = NotFound;

	CArray<double> splitValues;
	for( int i = 0; i < discretizationIntervals.Size(); i++ ) {
		double splitCriterionValue = 0;
		const bool isDiscreteFeature = problem->IsDiscreteFeature( usedFeatures[i] );
		if( isDiscreteFeature ) { 
			splitCriterionValue = calcDiscreteSplitCriterion( param, discretizationIntervals[i], totalStatistics, splitValues );
		} else {
			splitCriterionValue = calcContinuousSplitCriterion( param, discretizationIntervals[i], totalStatistics, splitValues );
		}

		if( criterionValue > splitCriterionValue ) { // the split with a better criterion value is found
			criterionValue = splitCriterionValue;
			featureIndex = usedFeatures[i];
			isDiscrete = isDiscreteFeature;
			splitValues.CopyTo( values );
		}
	}

	return ( featureIndex != NotFound );
}

double CClassificationStatistics::GetPredictions( CArray<double>& probabilities ) const
{
	NeoAssert( probabilities.IsEmpty() );

	const CArray<double>& weights = totalStatistics.Weights();
	probabilities.SetBufferSize( weights.Size() );
	double maxClassProbability = 0;
	for( int i = 0; i < weights.Size(); i++ ) {
		const double classProbability = weights[i] / totalStatistics.TotalWeight();
		probabilities.Add( classProbability );
		if( classProbability > maxClassProbability ) {
			maxClassProbability = classProbability;
		}
	}

	return maxClassProbability;
}

// Adds a new value as a separate interval
void CClassificationStatistics::addValue( int index, double value, int count, int classIndex, double weight )
{
	CInterval interval;
	interval.Class = classIndex;
	interval.Count = count;
	interval.Begin = value;
	interval.End = value;
	interval.Weight = weight;

	const int discretizationValue = problem->GetDiscretizationValue( usedFeatures[index] );
	const int maxIntervalCount = discretizationValue * classCount * 10;

	CIntervalArray& intervals = discretizationIntervals[index];

	if( intervals.Size() >= maxIntervalCount ) {
		mergeIntervals( discretizationValue, intervals );
	}
	NeoAssert( intervals.Size() < maxIntervalCount );

	if( intervals.Size() == intervals.BufferSize() ) {
		const int newBufferSize = min( (intervals.BufferSize() * 3) / 2, maxIntervalCount );
		intervals.SetBufferSize( newBufferSize );
	}

	intervals.Add( interval );
}

// Merges the intervals
void CClassificationStatistics::mergeIntervals( int discretizationValue, CIntervalArray& intervals )
{
	// Altogether we have discretizationValue * classCount * BigCoef intervals, create
	// discretizationValue * classCount * SmallCoef intervals
	NeoAssert( intervals.Size() <= discretizationValue * classCount * BigCoef );
	static_assert( SmallCoef < BigCoef, "SmallCoef >= BigCoef" );

	intervals.QuickSort< CompositeComparer<CInterval, AscendingByMember<CInterval, double, &CInterval::Begin>,
		AscendingByMember<CInterval, double, &CInterval::End> > >();

	// Merge the overlapping intervals
	mergeOverlappingIntervals( intervals );

	if( intervals.Size() <= discretizationValue * classCount * SmallCoef ) {
		return;
	}

	// Leaving a margin at the edges to avoid outliers
	static_assert( 2 < SmallCoef, "2 >= SmallCoef" );
	const int borderSize = discretizationValue * classCount;

	// The same intervals should either all be merged or all be left at the edges
	// Otherwise some will be duplicated
	int left = borderSize;
	while( left > 0 && isEqual( intervals[left], intervals[left - 1] ) ) {
		left--;
	}
	int right = intervals.Size() - borderSize - 1;
	while( right + 1 < intervals.Size() && isEqual( intervals[right], intervals[right + 1] ) ) {
		right++;
	}

	// Merge the central intervals by weight
	mergeIntervalsByWeight( left, right,
		SmallCoef * discretizationValue * classCount - left - ( intervals.Size() - right - 1), intervals );
}

// Merges overlapping intervals
void CClassificationStatistics::mergeOverlappingIntervals( CIntervalArray& intervals )
{
	CArray<int> curIntervals; // the current intervals for each class
	curIntervals.Add( NotFound, classCount );
	int newSize = 0; // the size after merging
	for( int i = 0; i < intervals.Size(); i++ ) {
		// Finding an interval that contains the current one
		bool found = false;
		for( int j = 0; j < curIntervals.Size(); j++ ) {
			if( curIntervals[j] != NotFound &&  intervals[i].End <= intervals[curIntervals[j]].End ) {
				// The intervals should either coincide or one should contain the whole of the other
				NeoAssert( intervals[curIntervals[j]].Begin <= intervals[i].Begin );
				// The intervals should be consistent so the new interval is expanded to the existing one
				intervals[i].Begin = intervals[curIntervals[j]].Begin;
				intervals[i].End = intervals[curIntervals[j]].End;
				found = true;
				break;
			}
		}

		const int curClass = intervals[i].Class;
		if( !found ) {
			// There is no containing interval, therefore the current intervals for all classes are invalid
			curIntervals.Empty();
			curIntervals.Add( NotFound, classCount );
		} else if( curIntervals[curClass] != NotFound ) {
			// There is a containing interval, as well as an open interval for the same class
			NeoAssert( isEqual( intervals[curIntervals[curClass]], intervals[i] ) );
			intervals[curIntervals[curClass]].Count += intervals[i].Count;
			intervals[curIntervals[curClass]].Weight += intervals[i].Weight;
			continue;
		}
		intervals[newSize] = intervals[i];
		curIntervals[curClass] = newSize;
		newSize++;
	}
	intervals.SetSize( newSize );
}

// Merges the intervals by weight
void CClassificationStatistics::mergeIntervalsByWeight( int left, int right, int resultIntervalCount, CIntervalArray& intervals )
{
	NeoAssert( left <= right );
	NeoAssert( resultIntervalCount > 0 );

	// Maximum interval weight
	const double maxSumWeight = classCount * sumWeight( intervals, left, right ) / resultIntervalCount;

	CArray<int> curIntervals; // the current intervals for each class
	curIntervals.Add( NotFound, classCount );
	// A new interval
	CInterval cur;
	cur.Begin = intervals[left].Begin;
	cur.End = intervals[left].End;
	cur.Weight = 0;

	int newSize = left; // the size after merge
	int i = left;
	while( i <= right ) {
		// The equal intervals should be processed simultaneously
		double currentWeight = 0;
		int j = i;
		while( j <= right && isEqual( intervals[i], intervals[j] ) ) {
			currentWeight += intervals[j].Weight;
			j++;
		}

		if( ( cur.Begin < 0 && 0 < intervals[i].Begin ) // the intervals should not pass through 0
			|| ( cur.Weight + currentWeight >= maxSumWeight ) ) // the required weight threshold crossed
		{
			closeIntervals( cur.Begin, cur.End, curIntervals, intervals );
			cur.Begin = intervals[i].Begin;
			cur.End = intervals[i].End;
			cur.Weight = 0;
		}

		// Adding to the current interval
		j = i;
		while( j <= right && isEqual( intervals[i], intervals[j] ) ) {
			if( curIntervals[intervals[j].Class] == NotFound ) {
				intervals[newSize] = intervals[j];
				curIntervals[intervals[j].Class] = newSize;
				newSize++;
			} else {
				intervals[curIntervals[intervals[j].Class]].Count += intervals[j].Count;
				intervals[curIntervals[intervals[j].Class]].Weight += intervals[j].Weight;
			}
			j++;
		}
		if( cur.End < intervals[i].End ) {
			cur.End = intervals[i].End;
		}
		cur.Weight += currentWeight;
		i = j;
	}

	// Skip the part at the end
	while( i < intervals.Size() ) {
		intervals[newSize] = intervals[i];
		newSize++;
		i++;
	}

	intervals.SetSize( newSize );
}

// Closes the intervals for all classes
void CClassificationStatistics::closeIntervals( double begin, double end, CArray<int>& link, CIntervalArray& intervals )
{
	for( int i = 0; i < link.Size(); i++ ) {
		if( link[i] != NotFound ) {
			intervals[link[i]].Begin = begin;
			intervals[link[i]].End = end;
		}
	}
	link.Empty();
	link.Add( NotFound, classCount );
}

// Calculates the criterion value for continuous feature split
double CClassificationStatistics::calcContinuousSplitCriterion( CDecisionTreeTrainingModel::CParams param,
	const CIntervalArray& intervals, const CVectorSetClassificationStatistic& total, CArray<double>& splitValues ) const
{
	CVectorSetClassificationStatistic first( total.Weights().Size() ); // empty
	CVectorSetClassificationStatistic second( total ); // all elements

	double resultValue = DBL_MAX;
	bool success = false;
	double threshold = 0;

	for( int i = 0; i < intervals.Size(); i++ ) {
		first.AddVectorSet( intervals[i].Count, intervals[i].Class, intervals[i].Weight );
		second.SubVectorSet( intervals[i].Count, intervals[i].Class, intervals[i].Weight );

		if( i + 1 < intervals.Size()
			&& intervals[i].Begin == intervals[i + 1].Begin
			&& intervals[i].End == intervals[i + 1].End )
		{
			continue;
		}
		if( first.TotalCount() < param.MinContinuousSubsetSize
			|| first.TotalWeight() < total.TotalWeight() * param.MinContinuousSubsetPart )
		{
			continue;
		}
		if( second.TotalCount() < param.MinContinuousSubsetSize
			|| second.TotalWeight() < total.TotalWeight() * param.MinContinuousSubsetPart )
		{
			break; // it can only decrease from here
		}

		const double value = ( first.CalcCriterion( param.SplitCriterion ) * first.TotalWeight()
			+ second.CalcCriterion( param.SplitCriterion ) * second.TotalWeight() ) / total.TotalWeight();

		if( resultValue > value ) {
			resultValue = value;
			success = true;
			if( i + 1 < intervals.Size() && fabs( intervals[i].End - intervals[i + 1].Begin ) > 1e-10 ) {
				threshold = ( intervals[i].End + intervals[i + 1].Begin ) / 2;
			} else {
				threshold = intervals[i].End;
			}
		}
	}

	if( success ) {
		splitValues.Empty();
		splitValues.Add( threshold, 2 );
	}
	return resultValue;
}

// Calculates the criterion value for discrete feature split
double CClassificationStatistics::calcDiscreteSplitCriterion( CDecisionTreeTrainingModel::CParams param,
	const CIntervalArray& intervals, const CVectorSetClassificationStatistic& total, CArray<double>& splitValues ) const
{
	splitValues.Empty();
	double result = 0;
	CVectorSetClassificationStatistic curSet( total.Weights().Size() ); // an empty set
	for( int i = 0; i < intervals.Size(); i++ ) {
		curSet.AddVectorSet( intervals[i].Count, intervals[i].Class, intervals[i].Weight );
		if( i + 1 < intervals.Size() && intervals[i].Begin == intervals[i+1].Begin ) {
			continue;
		}
		if( curSet.TotalCount() < param.MinDiscreteSubsetSize || curSet.TotalWeight() < total.TotalWeight() * param.MinDiscreteSubsetPart ) {
			return DBL_MAX;
		}
		result += curSet.CalcCriterion( param.SplitCriterion ) * curSet.TotalWeight();
		curSet.Erase();
		splitValues.Add( intervals[i].Begin );
	}

	return result / total.TotalWeight();
}

// Checks if the intervals are equal
bool CClassificationStatistics::isEqual( const CInterval& interval1, const CInterval& interval2 )
{
	return interval1.Begin == interval2.Begin && interval1.End == interval2.End;
}

// Calculates the total interval weight in [left, right] range
double CClassificationStatistics::sumWeight( const CIntervalArray& intervals, int left, int right )
{
	double weight = 0;
	for( int i = left; i <= right; i++ ) {
		weight += intervals[i].Weight;
	}
	return weight;
}

} // namespace NeoML
