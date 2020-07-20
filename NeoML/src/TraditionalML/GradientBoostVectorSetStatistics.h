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

// The statistics accumulated for a vector set while building a tree with gradient boosting
class CGradientBoostVectorSetStatistics {
public:
	CGradientBoostVectorSetStatistics();
	explicit CGradientBoostVectorSetStatistics( const CGradientBoostVectorSetStatistics& other );
	CGradientBoostVectorSetStatistics& operator=( const CGradientBoostVectorSetStatistics& other );

	// Adds a vector
	void Add( double gradient, double hessian, float weight );
	void Add( const CGradientBoostVectorSetStatistics& that );

	// Deletes a vector
	void Sub( double gradient, double hessian, float weight );
	void Sub( const CGradientBoostVectorSetStatistics& that );

	// Clears all accumulated data
	void Erase();

	// Gets the total gradient
	double TotalGradient() const { return totalGradient; }

	// Gets the total hessian
	double TotalHessian() const { return totalHessian; }

	// Gets the total weight
	float TotalWeight() const { return totalWeight; }

	// Calculates the criterion
	double CalcCriterion( float l1, float l2 ) const;

private:
	double totalGradient; // total gradient
	double totalHessian; // total hessian
	float totalWeight; // total weight
};

inline CGradientBoostVectorSetStatistics::CGradientBoostVectorSetStatistics() :
	totalGradient( 0.0 ),
	totalHessian( 0.0 ),
	totalWeight( 0.0 )
{
}

inline CGradientBoostVectorSetStatistics::CGradientBoostVectorSetStatistics( const CGradientBoostVectorSetStatistics& other ) :
	totalGradient( other.totalGradient ),
	totalHessian( other.totalHessian ),
	totalWeight( other.totalWeight )
{
}

inline CGradientBoostVectorSetStatistics& CGradientBoostVectorSetStatistics::operator=( const CGradientBoostVectorSetStatistics& other )
{
	if( &other != this ) {
		totalGradient = other.totalGradient;
		totalHessian = other.totalHessian;
		totalWeight = other.totalWeight;
	}
	return *this;
}

inline void CGradientBoostVectorSetStatistics::Add( double gradient, double hessian, float weight )
{
	totalGradient += gradient;
	totalHessian += hessian;
	totalWeight += weight;
}

inline void CGradientBoostVectorSetStatistics::Add( const CGradientBoostVectorSetStatistics& that )
{
	Add( that.TotalGradient(), that.TotalHessian(), that.TotalWeight() );
}

inline void CGradientBoostVectorSetStatistics::Sub( double gradient, double hessian, float weight )
{
	totalGradient -= gradient;
	totalHessian -= hessian;
	totalWeight -= weight;
}

inline void CGradientBoostVectorSetStatistics::Sub( const CGradientBoostVectorSetStatistics& that )
{
	Sub( that.TotalGradient(), that.TotalHessian(), that.TotalWeight() );
}

inline void CGradientBoostVectorSetStatistics::Erase()
{
	totalGradient = 0;
	totalHessian = 0;
	totalWeight = 0;
}

inline double CGradientBoostVectorSetStatistics::CalcCriterion( float l1, float l2 ) const
{
	double temp = 0;
	if( totalGradient > l1 ) {
		temp = totalGradient - l1;
	} else if( totalGradient < -l1 ) {
		temp = totalGradient + l1;
	} else {
		temp = totalGradient;
	}
	return temp * temp / ( totalHessian + l2 );
}

} // namespace NeoML
