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

#include <NeoML/TraditionalML/FeatureSelection.h>
#include <DecisionTreeNodeClassificationStatistic.h>

namespace NeoML {

struct CClusterStatistics {
	CArray<double> Sum;
	CArray<double> SumSquare;
	double SumWeight;

	explicit CClusterStatistics( int featuresCount );

	void AddVector( const CFloatVectorDesc& vector, double weight );
	void GetVariance( CArray<double>& variance );
};

inline CClusterStatistics::CClusterStatistics( int featuresCount ) :
	SumWeight( 0 )
{
	Sum.Add( 0.0, featuresCount );
	SumSquare.Add( 0.0, featuresCount );
}

inline void CClusterStatistics::AddVector( const CFloatVectorDesc& vector, double weight )
{
	SumWeight += weight;

	if( vector.Indexes == nullptr ) {
		for( int j = 0; j < vector.Size; j++ ) {
			Sum[j] += vector.Values[j] * weight;
			SumSquare[j] += vector.Values[j] * vector.Values[j] * weight;
		}
	} else {
		for( int j = 0; j < vector.Size; j++ ) {
			Sum[vector.Indexes[j]] += vector.Values[j] * weight;
			SumSquare[vector.Indexes[j]] += vector.Values[j] * vector.Values[j] * weight;
		}
	}
}

inline void CClusterStatistics::GetVariance( CArray<double>& variance )
{
	variance.Empty();
	variance.SetBufferSize( Sum.Size() );
	for( int i = 0; i < Sum.Size(); i++ ) {
		variance.Add( ( SumSquare[i] / SumWeight ) - ( Sum[i] * Sum[i] / SumWeight / SumWeight ) );
	}
}

void CalcFeaturesVariance( const IProblem& problem, CArray<double>& variance )
{
	const int featuresCount = problem.GetFeatureCount();
	const int vectorsCount = problem.GetVectorCount();

	CClusterStatistics statistic( featuresCount );

	CFloatMatrixDesc matrix = problem.GetMatrix();

	for( int i = 0; i < vectorsCount; i++ ) {
		CFloatVectorDesc desc;
		matrix.GetRow( i, desc );
		statistic.AddVector( desc, problem.GetVectorWeight( i ) );
	}

	statistic.GetVariance( variance );
}

void CalcFeaturesVarianceRatio( const IProblem& problem, CArray<double>& varianceRatio )
{
	const int vectorsCount = problem.GetVectorCount();
	const int featuresCount = problem.GetFeatureCount();

	CClusterStatistics total( featuresCount ); // total statistics for the entire data set
	CPointerArray<CClusterStatistics> statistics; // statistics for each cluster separately
	for( int i = 0; i < problem.GetClassCount(); i++ ) {
		statistics.Add( FINE_DEBUG_NEW CClusterStatistics( featuresCount ) );
	}

	CFloatMatrixDesc matrix = problem.GetMatrix();

	for( int i = 0; i < vectorsCount; i++ ) {
		CFloatVectorDesc desc;
		matrix.GetRow( i, desc );
		double weight = problem.GetVectorWeight( i );

		total.AddVector( desc, weight );
		statistics[problem.GetClass( i )]->AddVector( desc, weight );
	}

	CArray<double> averageVariance;
	averageVariance.Add( 0.0, featuresCount );
	CArray<double> curClassVariance;
	for( int i = 0; i < problem.GetClassCount(); i++ ) {
		statistics[i]->GetVariance( curClassVariance );
		for( int j = 0; j < featuresCount; j++ ) {
			averageVariance[j] += curClassVariance[j];
		}
	}

	total.GetVariance( varianceRatio );
	for( int i = 0; i < problem.GetClassCount(); i++ ) {
		varianceRatio[i] /= ( averageVariance[i] / problem.GetClassCount() );
	}
}

// Calculates the chi-square statistics for observations and expectations
static void calcChiSquare( const CArray< CArray<double> >& observed, const CArray< CArray<double> >& expected,
	CArray<double>& chi2 )
{
	NeoAssert( observed.Size() == expected.Size() );
	int len = observed.First().Size();
	NeoAssert( len > 0 );
	chi2.Empty();
	chi2.SetBufferSize( len );

	for( int i = 0; i < len; i++ ) {
		double value = 0;
		for( int j = 0; j < observed.Size(); j++ ) {
			const double diff = observed[j][i] - expected[j][i];
			value += diff * diff / expected[j][i];
		}
		chi2.Add( value );
	}
}

void CalcFeaturesChiSquare( const IProblem& problem, CArray<double>& chi2 )
{
	const int featureCount = problem.GetFeatureCount();
	const int vectorCount = problem.GetVectorCount();
	const int classCount = problem.GetClassCount();

	// Credits to sklearn
	CArray< CArray<double> > observed; // the observed distribution
	observed.SetSize( classCount );
	for( int i = 0; i < observed.Size(); i++ ) {
		observed[i].Add( 0.0, featureCount );
	}
	CArray<double> classWeight; // the total weight for the class
	classWeight.Add( 0.0, classCount );
	double totalWeight = 0; // the total weight of all vectors

	CFloatMatrixDesc matrix = problem.GetMatrix();

	// Calculate the observed distribution
	for( int i = 0; i < vectorCount; i++ ) {
		CFloatVectorDesc vector;
		matrix.GetRow( i, vector );
		const double weight = problem.GetVectorWeight( i );
		const int classIndex = problem.GetClass( i );

		totalWeight += weight;
		classWeight[classIndex] += weight;

		CArray<double>& oneObserved = observed[classIndex];
		if( vector.Indexes == nullptr ) {
			for( int j = 0; j < vector.Size; j++ ) {
				oneObserved[j] += weight * vector.Values[j];
			}
		} else {
			for( int j = 0; j < vector.Size; j++ ) {
				oneObserved[vector.Indexes[j]] += weight * vector.Values[j];
			}
		}
	}

	// Calculate the expected distribution
	for( int i = 0; i < classWeight.Size(); i++ ) {
		classWeight[i] /= totalWeight;
	}
	CArray<double> totalFeatureValue;
	totalFeatureValue.SetBufferSize( featureCount );
	for( int i = 0; i < featureCount; i++ ) {
		double value = 0;
		for( int j = 0; j < classCount; j++ ) {
			value += observed[j][i];
		}
		totalFeatureValue.Add( value );
	}

	CArray< CArray<double> > expected;
	expected.SetSize( classCount );
	for( int i = 0; i < expected.Size(); i++ ) {
		CArray<double>& oneExpected = expected[i];
		oneExpected.SetBufferSize( featureCount );
		for( int j = 0; j < featureCount; j++ ) {
			oneExpected.Add( totalFeatureValue[j] * classWeight[i] );
		}
	}

	// Calculate the chi-square for the two distributions
	calcChiSquare( observed, expected, chi2 );
}

double CalcTwoFeaturesCorrelation( const IProblem& problem, int index1, int index2 )
{
	NeoAssert( index1 >= 0 );
	NeoAssert( index1 < problem.GetFeatureCount() );
	NeoAssert( index2 >= 0 );
	NeoAssert( index2 < problem.GetFeatureCount() );

	const int vectorCount = problem.GetVectorCount();
	double mean1 = 0;
	double mean2 = 0;

	CFloatMatrixDesc matrix = problem.GetMatrix();
	CFloatVectorDesc vector;
	for( int i = 0; i < vectorCount; i++ ) {
		matrix.GetRow( i, vector );
		mean1 += GetValue( vector, index1 );
		mean2 += GetValue( vector, index2 );
	}

	mean1 /= vectorCount;
	mean2 /= vectorCount;
	double cov = 0;
	double variance1 = 0;
	double variance2 = 0;

	for( int i = 0; i < vectorCount; i++ ) {
		matrix.GetRow( i, vector );
		const double value1 = GetValue( vector, index1 ) - mean1;
		const double value2 = GetValue( vector, index2 ) - mean2;
		cov += value1 * value2;
		variance1 += value1 * value1;
		variance2 += value2 * value2;
	}

	return cov / sqrt( variance1 * variance2 );
}

double CalcFeatureAndClassCorrelation( const IProblem& problem, int featureIndex, int classIndex )
{
	NeoAssert( featureIndex >= 0 );
	NeoAssert( featureIndex < problem.GetFeatureCount() );
	NeoAssert( classIndex >= 0 );
	NeoAssert( classIndex < problem.GetClassCount() );

	const int vectorCount = problem.GetVectorCount();
	double mean1 = 0;
	double mean2 = 0;

	CFloatMatrixDesc matrix = problem.GetMatrix();
	CFloatVectorDesc vector;

	for( int i = 0; i < vectorCount; i++ ) {
		matrix.GetRow( i, vector );
		mean1 += GetValue( vector, featureIndex );
		mean2 += problem.GetClass( i ) == classIndex ? 1 : 0;
	}

	mean1 /= vectorCount;
	mean2 /= vectorCount;
	double cov = 0;
	double variance1 = 0;
	double variance2 = 0;

	for( int i = 0; i < vectorCount; i++ ) {
		matrix.GetRow( i, vector );
		const double value1 = GetValue( vector, featureIndex ) - mean1;
		const double value2 = ( problem.GetClass( i ) == classIndex ? 1 : 0 ) - mean2;
		cov += value1 * value2;
		variance1 += value1 * value1;
		variance2 += value2 * value2;
	}

	return cov / sqrt( variance1 * variance2 );
}

void CalcFeaturesInformationGain( const IProblem& problem, CArray<double>& informationGain )
{
	const CDecisionTree::TSplitCriterion criterion = CDecisionTree::SC_InformationGain;
	const int vectorCount = problem.GetVectorCount();
	const int classCount = problem.GetClassCount();
	const int featureCount = problem.GetFeatureCount();

	// Accumulate statistics to calculate information gain
	CPointerArray< CMap<double, CVectorSetClassificationStatistic*> > statistics; // statistics for each feature value
	statistics.SetBufferSize( featureCount );
	for( int i = 0; i < featureCount; i++ ) {
		if( problem.IsDiscreteFeature( i ) ) {
			statistics.Add( FINE_DEBUG_NEW CMap<double, CVectorSetClassificationStatistic*>() );
		} else {
			statistics.Add( 0 );
		}
	}

	CFloatMatrixDesc matrix = problem.GetMatrix();

	CVectorSetClassificationStatistic fullProblemStatistic( classCount );

	for( int i = 0; i < vectorCount; i++ ) {
		CFloatVectorDesc vector;
		matrix.GetRow( i, vector );
		const int classIndex = problem.GetClass( i );
		const double weight = problem.GetVectorWeight( i );

		for( int j = 0; j < vector.Size; j++ ) {
			if( vector.Values[j] != 0.0 ) {
				const int index = vector.Indexes == nullptr ? j : vector.Indexes[j];
				if( !problem.IsDiscreteFeature( index ) ) {
					continue;
				}
				CVectorSetClassificationStatistic*& oneValueStatistics = statistics[index]->GetOrCreateValue( vector.Values[j], 0 );
				if( oneValueStatistics == 0 ) {
					oneValueStatistics = FINE_DEBUG_NEW CVectorSetClassificationStatistic( classCount );
				}
				oneValueStatistics->AddVectorSet( 1, classIndex, weight );
			}
		}
		fullProblemStatistic.AddVectorSet( 1, classIndex, weight );
	}

	// Calculate the information gain
	informationGain.Empty();
	informationGain.SetBufferSize( featureCount );
	for( int i = 0; i < statistics.Size(); i++ ) {
		double value = 0;
		CMap<double, CVectorSetClassificationStatistic*>* oneStatistics = statistics[i];
		if( oneStatistics == 0 ) {
			informationGain.Add( 0 );
		} else {
			TMapPosition pos = oneStatistics->GetFirstPosition();
			while( pos != NotFound ) {
				CVectorSetClassificationStatistic* oneStatisticsValue = oneStatistics->GetValue( pos );
				value += oneStatisticsValue->CalcCriterion( criterion ) * oneStatisticsValue->TotalWeight();
				delete oneStatisticsValue; // delete directly
				pos = oneStatistics->GetNextPosition( pos );
			}
			informationGain.Add( fullProblemStatistic.CalcCriterion( criterion ) - value / fullProblemStatistic.TotalWeight() );
		}
	}
}

} // namespace NeoML
