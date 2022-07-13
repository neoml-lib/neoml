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
class CGradientBoostStatisticsSingle {
public:
	typedef double Type;

	CGradientBoostStatisticsSingle();
	explicit CGradientBoostStatisticsSingle( int valueSize );
	CGradientBoostStatisticsSingle( const CGradientBoostStatisticsSingle& other );
	explicit CGradientBoostStatisticsSingle( double gradient, double hessian, double weight );
	CGradientBoostStatisticsSingle& operator=( const CGradientBoostStatisticsSingle& other );

	// Adds a vector
	void Add( double gradient, double hessian, double weight );
	void Add( const CArray<double>& gradient, const CArray<double>& hessian, const CArray<double>& weight, int vectorIndex );
	void Add( const CGradientBoostStatisticsSingle& other );

	// Deletes a vector
	void Sub( double gradient, double hessian, double weight );
	void Sub( const CGradientBoostStatisticsSingle& other );

	// Clears all accumulated data
	void Erase();

	// Calculates the criterion
	double CalcCriterion( float l1, float l2 ) const;

	// Calculates the split criterion
	static bool CalcCriterion( double& criterion, CGradientBoostStatisticsSingle& leftResult, CGradientBoostStatisticsSingle& rightResult, const CGradientBoostStatisticsSingle& totalStatistics,
		float l1RegFactor, float l2RegFactor, double minSubsetHessian, double minSubsetWeight, float denseTreeBoostCoefficient );

	// Gets the total gradient
	double TotalGradient() const { return totalGradient; }

	// Gets the total hessian
	double TotalHessian() const { return totalHessian; }

	// Gets the total weight
	double TotalWeight() const { return totalWeight; }

	// Get leaf value
	void LeafValue( double& value ) const;

	// Check if statistics is not enough
	bool IsSmall( double minSubsetHessian, double minSubsetWeight );

	// Get value size
	int ValueSize() const { return 1; }

	// Set value size
	void SetSize( int valueSize ) { NeoAssert( valueSize == 1 ); }

	// Mark classes that not splitting further
	void NullifyLeafClasses( const CGradientBoostStatisticsSingle& ) {};

private:
	double totalGradient; // total gradient
	double totalHessian; // total hessian
	double totalWeight; // total weight
};

inline CGradientBoostStatisticsSingle::CGradientBoostStatisticsSingle()
{
	totalGradient = 0.0;
	totalHessian = 0.0;
	totalWeight = 0.0;
}

inline CGradientBoostStatisticsSingle::CGradientBoostStatisticsSingle( int valueSize )
{
	NeoAssert( valueSize == 1 );
	totalGradient = 0.0;
	totalHessian = 0.0;
	totalWeight = 0.0;
}

inline CGradientBoostStatisticsSingle::CGradientBoostStatisticsSingle( double gradient, double hessian, double weight )
{
	totalGradient = gradient;
	totalHessian = hessian;
	totalWeight = weight;
}

inline CGradientBoostStatisticsSingle::CGradientBoostStatisticsSingle( const CGradientBoostStatisticsSingle& other )
{
	totalGradient = other.totalGradient;
	totalHessian = other.totalHessian;
	totalWeight = other.totalWeight;
}

inline CGradientBoostStatisticsSingle& CGradientBoostStatisticsSingle::operator=( const CGradientBoostStatisticsSingle& other )
{
	if( &other != this ) {
		totalGradient = other.totalGradient;
		totalHessian = other.totalHessian;
		totalWeight = other.totalWeight;
	}
	return *this;
}

inline void CGradientBoostStatisticsSingle::Add( double gradient, double hessian, double weight )
{
	totalGradient += gradient;
	totalHessian += hessian;
	totalWeight += weight;
}

inline void CGradientBoostStatisticsSingle::Add( const CArray<double>& gradient, const CArray<double>& hessian, const CArray<double>& weight, int vectorIndex )
{
	totalGradient += gradient[vectorIndex];
	totalHessian += hessian[vectorIndex];
	totalWeight += weight[vectorIndex];
}

inline void CGradientBoostStatisticsSingle::Add( const CGradientBoostStatisticsSingle& other )
{
	totalGradient += other.totalGradient;
	totalHessian += other.totalHessian;
	totalWeight += other.totalWeight;
}

inline void CGradientBoostStatisticsSingle::Sub( double gradient, double hessian, double weight )
{
	totalGradient -= gradient;
	totalHessian -= hessian;
	totalWeight -= weight;
}

inline void CGradientBoostStatisticsSingle::Sub( const CGradientBoostStatisticsSingle& other )
{
	totalGradient -= other.totalGradient;
	totalHessian -= other.totalHessian;
	totalWeight -= other.totalWeight;
}

inline void CGradientBoostStatisticsSingle::Erase()
{
	totalGradient = 0.0;
	totalHessian = 0.0;
	totalWeight = 0.0;
}

inline double CGradientBoostStatisticsSingle::CalcCriterion( float l1, float l2 ) const
{
	double temp = 0;
	if( totalGradient > l1 ) {
		temp = totalGradient - l1;
	} else if( totalGradient < -l1 ) {
		temp = totalGradient + l1;
	}
	return temp * temp / ( totalHessian + l2 );
}


inline bool CGradientBoostStatisticsSingle::IsSmall( double minSubsetHessian, double minSubsetWeight )
{
	return totalHessian < minSubsetHessian || totalWeight < minSubsetWeight;
}

inline void CGradientBoostStatisticsSingle::LeafValue( double& value ) const
{
	if( totalHessian == 0 ) {
		value = totalGradient;
	} else {
		value = -totalGradient / totalHessian;
	}
}

inline bool CGradientBoostStatisticsSingle::CalcCriterion( double& criterion,
	CGradientBoostStatisticsSingle& leftResult, CGradientBoostStatisticsSingle& rightResult, const CGradientBoostStatisticsSingle&,
	float l1RegFactor, float l2RegFactor, double minSubsetHessian, double minSubsetWeight, float )
{
	if( leftResult.IsSmall( minSubsetHessian, minSubsetWeight ) ||
		rightResult.IsSmall( minSubsetHessian, minSubsetWeight ) )
	{
		return false;
	}

	criterion = leftResult.CalcCriterion( l1RegFactor, l2RegFactor ) +
		rightResult.CalcCriterion( l1RegFactor, l2RegFactor );
	return true;
}

} // namespace NeoML