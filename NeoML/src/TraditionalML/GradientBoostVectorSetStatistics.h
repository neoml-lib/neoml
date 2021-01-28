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

namespace NeoML {
	typedef CArrayIterator<CArray<CArray<double>>> GradientBoostStatType;

	// The statistics accumulated for a vector set while building a tree with gradient boosting
	class CGradientBoostVectorSetStatistics {
	public:

		CGradientBoostVectorSetStatistics() = default;
		CGradientBoostVectorSetStatistics( int classCount );
		explicit CGradientBoostVectorSetStatistics( const CGradientBoostVectorSetStatistics& other );
		CGradientBoostVectorSetStatistics& operator=( const CGradientBoostVectorSetStatistics& other );

		// Adds a vector
		void Add( const CArray<double>& gradient, const CArray<double>& hessian, float weight );
		void Add( const GradientBoostStatType& gradients, const GradientBoostStatType& hessians, const CArray<float>& weights, int vectorIndex );
		void Add( const CGradientBoostVectorSetStatistics& that );
		void Add( const CGradientBoostVectorSetStatistics& that, int classIndex );

		// Deletes a vector
		void Sub( const CArray<double>& gradient, const CArray<double>& hessian, float weight );
		void Sub( CGradientBoostVectorSetStatistics& that );
		void Sub( CGradientBoostVectorSetStatistics& that, int classIndex );

		// Clears all accumulated data
		void Erase();

		// Gets the total gradient
		const CArray<double>& TotalGradient() const { return totalGradient; }

		// Gets the total hessian
		const CArray<double>& TotalHessian() const { return totalHessian; }

		// Gets the total weight
		float TotalWeight() const { return totalWeight; }

		// Check if total statistics enough for split
		bool CGradientBoostVectorSetStatistics::StatisticsIsSmall( double minSubsetHessian, double minSubsetWeight, int classIndex ) const;

		// Calculates the criterion for class
		double CalcCriterion( float l1, float l2, int classIndex ) const;
		// Sum of criterions for all classes
		double CalcCriterion( float l1, float l2 ) const;

	private:
		CArray<double> totalGradient; // total gradient
		CArray<double> totalHessian; // total hessian
		float totalWeight; // total weight
	};

inline CGradientBoostVectorSetStatistics::CGradientBoostVectorSetStatistics( int classCount ) :
	totalWeight( 0.0 )
{
	totalGradient.Add( 0, classCount );
	totalHessian.Add( 0, classCount );
}

inline CGradientBoostVectorSetStatistics::CGradientBoostVectorSetStatistics( const CGradientBoostVectorSetStatistics& other ) :
	totalWeight( other.totalWeight )
{
	other.totalGradient.CopyTo( totalGradient );
	other.totalHessian.CopyTo( totalHessian );
}

inline CGradientBoostVectorSetStatistics& CGradientBoostVectorSetStatistics::operator=( const CGradientBoostVectorSetStatistics& other )
{
	if( &other != this ) {
		other.totalGradient.CopyTo( totalGradient );
		other.totalHessian.CopyTo( totalHessian );
		totalWeight = other.totalWeight;
	}
	return *this;
}

inline void CGradientBoostVectorSetStatistics::Add( const CArray<double>& gradient, const CArray<double>& hessian, float weight )
{
	for( int i = 0; i < gradient.Size(); i++ ){
		totalGradient[i] += gradient[i];
		totalHessian[i] += hessian[i];
	}
	totalWeight += weight;
}

inline void CGradientBoostVectorSetStatistics::Add( const CGradientBoostVectorSetStatistics& that )
{
	Add( that.TotalGradient(), that.TotalHessian(), that.TotalWeight() );
}

inline void CGradientBoostVectorSetStatistics::Add( const CGradientBoostVectorSetStatistics& that, int classIndex )
{
	totalGradient[classIndex] += that.TotalGradient()[classIndex];
	totalHessian[classIndex] += that.TotalHessian()[classIndex];
	totalWeight += that.TotalWeight();
}

inline void CGradientBoostVectorSetStatistics::Add( const GradientBoostStatType& gradients, const GradientBoostStatType& hessians,
	const CArray<float>& weights, int vectorIndex )
{
	for( int i = 0; i < totalGradient.Size(); i++ ){
		totalGradient[i] += gradients[i][vectorIndex];
		totalHessian[i] += hessians[i][vectorIndex];
	}
	totalWeight += weights[vectorIndex];
}

inline void CGradientBoostVectorSetStatistics::Sub( const CArray<double>& gradients, const CArray<double>& hessians, float weight )
{
	for( int i = 0; i < gradients.Size(); i++ ){
		totalGradient[i] -= gradients[i];
		totalHessian[i] -= hessians[i];
	}
	totalWeight -= weight;
}

inline void CGradientBoostVectorSetStatistics::Sub( CGradientBoostVectorSetStatistics& that )
{
	Sub( that.TotalGradient(), that.TotalHessian(), that.TotalWeight() );
}

inline void CGradientBoostVectorSetStatistics::Sub( CGradientBoostVectorSetStatistics& that, int classIndex )
{
	totalGradient[classIndex] -= that.TotalGradient()[classIndex];
	totalHessian[classIndex] -= that.TotalHessian()[classIndex];
	totalWeight -= that.TotalWeight();
}

inline void CGradientBoostVectorSetStatistics::Erase()
{
	for( int i = 0; i < totalGradient.Size(); i++ ){
		totalGradient[i] = 0;
		totalHessian[i] = 0;
	}
	totalWeight = 0;
}

inline double CGradientBoostVectorSetStatistics::CalcCriterion( float l1, float l2, int classIndex ) const
{
	double temp = totalGradient[classIndex];
	if( temp > l1 ) {
		temp -= l1;
	} else if( temp < -l1 ) {
		temp += l1;
	}
	return temp * temp / (totalHessian[classIndex] + l2);
}

inline double CGradientBoostVectorSetStatistics::CalcCriterion( float l1, float l2 ) const
{
	double criterion = 0;
	for( int i = 0; i < totalGradient.Size(); i++ ){
		criterion += CalcCriterion( l1, l2, i );
	}
	return criterion;
}

inline bool CGradientBoostVectorSetStatistics::StatisticsIsSmall( double minSubsetHessian, double minSubsetWeight, int classIndex ) const
{
	return totalHessian[classIndex] < minSubsetHessian || totalWeight < minSubsetWeight;
}

} // namespace NeoML
