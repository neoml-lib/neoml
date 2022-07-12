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
class CGradientBoostStatisticsMulti {
public:
	typedef CArray<double> Type;
	CGradientBoostStatisticsMulti() = default;
	explicit CGradientBoostStatisticsMulti( int valueSize );
	CGradientBoostStatisticsMulti( const CGradientBoostStatisticsMulti& other );
	explicit CGradientBoostStatisticsMulti( const CArray<double>& gradient, const CArray<double>& hessian, double weight );
	CGradientBoostStatisticsMulti& operator=( const CGradientBoostStatisticsMulti& other );

	// Adds a vector
	void Add( const CArray<double>& gradient, const CArray<double>& hessian, double weight );
	void Add( const CArray<CArray<double>>& gradient, const CArray<CArray<double>>& hessian, const CArray<double>& weight, int vectorIndex );
	void Add( const CGradientBoostStatisticsMulti& other );

	// Deletes a vector
	void Sub( const CArray<double>& gradient, const CArray<double>& hessian, double weight );
	void Sub( const CGradientBoostStatisticsMulti& other );

	// Clears all accumulated data
	void Erase();

	// Calculates the criterion for single class
	double CalcCriterion( float l1, float l2, int classIndex ) const;

	// Calculates the criterion as sum of criterions for non-leaf classes
	double CalcCriterion( float l1, float l2 ) const;

	// Calculates the split criterion for multiple classes
	static bool CalcCriterion( double& criterion,
		CGradientBoostStatisticsMulti& leftResult, CGradientBoostStatisticsMulti& rightResult,
		const CGradientBoostStatisticsMulti& totalStatistics, float l1RegFactor, float l2RegFactor,
		double minSubsetHessian, double minSubsetWeight, float denseTreeBoostCoefficient );

	// Gets the total gradient
	const CArray<double>& TotalGradient() const { return totalGradient; }

	// Gets the total hessian
	const CArray<double>& TotalHessian() const { return totalHessian; }

	// Gets the total weight
	double TotalWeight() const { return totalWeight; }

	// Check if statistics is not enough
	bool IsSmall( double minSubsetHessian, double minSubsetWeight, int classIndex ) const;

	// Get leaf value
	void LeafValue( CArray<double>& value ) const;

	// Get value size
	int ValueSize() const { return totalGradient.Size(); }

	// Set value size
	void SetSize( int valueSize );

	// Mark classes that not splitting further
	void NullifyLeafClasses( const CGradientBoostStatisticsMulti& parent );

private:
	CArray<double> totalGradient; // total gradient
	CArray<double> totalHessian; // total hessian
	double totalWeight; // total weight
};

inline CGradientBoostStatisticsMulti::CGradientBoostStatisticsMulti( int valueSize )
{
	totalGradient.Add( 0.0, valueSize );
	totalHessian.Add( 0.0, valueSize );
	totalWeight = 0.0;
}

inline CGradientBoostStatisticsMulti::CGradientBoostStatisticsMulti( const CArray<double>& gradient, const CArray<double>& hessian, double weight )
{
	totalGradient.SetSize( gradient.Size() );
	totalHessian.SetSize( hessian.Size() );
	for( int i = 0; i < totalGradient.Size(); i++ ) {
		totalGradient[i] = gradient[i];
		totalHessian[i] = hessian[i];
	}
	totalWeight = weight;
}

inline CGradientBoostStatisticsMulti::CGradientBoostStatisticsMulti( const CGradientBoostStatisticsMulti& other )
{
	other.totalGradient.CopyTo( totalGradient );
	other.totalHessian.CopyTo( totalHessian );
	totalWeight = other.totalWeight;
}

inline CGradientBoostStatisticsMulti& CGradientBoostStatisticsMulti::operator=( const CGradientBoostStatisticsMulti& other )
{
	if( &other != this ) {
		other.totalGradient.CopyTo( totalGradient );
		other.totalHessian.CopyTo( totalHessian );
		totalWeight = other.totalWeight;
	}
	return *this;
}

inline void CGradientBoostStatisticsMulti::Add( const CArray<double>& gradient, const CArray<double>& hessian, double weight )
{
	for( int i = 0; i < totalGradient.Size(); i++ ) {
		totalGradient[i] += gradient[i];
		totalHessian[i] += hessian[i];
	}
	totalWeight += weight;
}

inline void CGradientBoostStatisticsMulti::Add( const CArray<CArray<double>>& gradient, const CArray<CArray<double>>& hessian, const CArray<double>& weight, int vectorIndex )
{
	for( int i = 0; i < gradient.Size(); i++ ) {
		totalGradient[i] += gradient[i][vectorIndex];
		totalHessian[i] += hessian[i][vectorIndex];
	}
	totalWeight += weight[vectorIndex];
}

inline void CGradientBoostStatisticsMulti::Add( const CGradientBoostStatisticsMulti& other )
{
	for( int i = 0; i < totalGradient.Size(); i++ ) {
		totalGradient[i] += other.totalGradient[i];
		totalHessian[i] += other.totalHessian[i];
	}
	totalWeight += other.totalWeight;
}

inline void CGradientBoostStatisticsMulti::Sub( const CArray<double>& gradient, const CArray<double>& hessian, double weight )
{
	for( int i = 0; i < totalGradient.Size(); i++ ) {
		totalGradient[i] -= gradient[i];
		totalHessian[i] -= hessian[i];
	}
	totalWeight -= weight;
}

inline void CGradientBoostStatisticsMulti::Sub( const CGradientBoostStatisticsMulti& other )
{
	for( int i = 0; i < totalGradient.Size(); i++ ) {
		totalGradient[i] -= other.totalGradient[i];
		totalHessian[i] -= other.totalHessian[i];
	}
	totalWeight -= other.totalWeight;
}

inline void CGradientBoostStatisticsMulti::Erase()
{
	for( int i = 0; i < totalGradient.Size(); i++ ) {
		totalGradient[i] = 0.0;
		totalHessian[i] = 0.0;
	}
	totalWeight = 0.0;
}

inline double CGradientBoostStatisticsMulti::CalcCriterion( float l1, float l2, int classIndex ) const
{
	double temp = 0;
	if( totalGradient[classIndex] > l1 ) {
		temp = totalGradient[classIndex] - l1;
	} else if( totalGradient[classIndex] < -l1 ) {
		temp = totalGradient[classIndex] + l1;
	}
	return temp * temp / ( totalHessian[classIndex] + l2 );
}

inline double CGradientBoostStatisticsMulti::CalcCriterion( float l1, float l2 ) const
{
	double res = 0;
	for( int i = 0; i < totalGradient.Size(); i++ ) {
		if( totalHessian[i] != 0 ) {
			res += CalcCriterion( l1, l2, i );
		}
	}
	return res;
}

inline bool CGradientBoostStatisticsMulti::IsSmall( double minSubsetHessian, double minSubsetWeight, int classIndex ) const
{
	return totalHessian[classIndex] < minSubsetHessian || totalWeight < minSubsetWeight;
}

inline void CGradientBoostStatisticsMulti::LeafValue( CArray<double>& value ) const
{
	value.SetSize( totalGradient.Size() );
	for( int i = 0; i < totalGradient.Size(); i++ ) {
		if( totalHessian[i] == 0 ) {
			value[i] = totalGradient[i];
		} else {
			value[i] = -totalGradient[i] / totalHessian[i];
		}
	}
}

inline bool CGradientBoostStatisticsMulti::CalcCriterion( double& criterion,
	CGradientBoostStatisticsMulti& leftResult, CGradientBoostStatisticsMulti& rightResult, const CGradientBoostStatisticsMulti& totalStatistics,
	float l1RegFactor, float l2RegFactor, double minSubsetHessian, double minSubsetWeight, float denseTreeBoostCoefficient )
{
	double result = 0;
	int leafClassesCount = 0;

	for( int i = 0; i < totalStatistics.ValueSize(); i++ ) {
		// hessian equals 0 for leaf classes
		bool isAlreadyLeafClass = totalStatistics.IsSmall( minSubsetHessian, minSubsetWeight, i );
		bool isNewLeafClass = false;
		double valueCriterion = 0;
		if( !isAlreadyLeafClass ) {
			if( leftResult.IsSmall( minSubsetHessian, minSubsetWeight, i ) ||
				rightResult.IsSmall( minSubsetHessian, minSubsetWeight, i ) )
			{
				isNewLeafClass |= true;
			}

			valueCriterion = totalStatistics.CalcCriterion( l1RegFactor, l2RegFactor, i );
			if( !isNewLeafClass ) {
				double splitCriterion = leftResult.CalcCriterion( l1RegFactor, l2RegFactor, i ) +
					rightResult.CalcCriterion( l1RegFactor, l2RegFactor, i );

				if( splitCriterion < valueCriterion ) {
					isNewLeafClass = true;
				} else {
					valueCriterion = splitCriterion;
				}
			}
		}

		if( isNewLeafClass || isAlreadyLeafClass ) {
			if( isNewLeafClass ) {
				leftResult.totalGradient[i] = -totalStatistics.totalGradient[i] / totalStatistics.totalHessian[i];
				rightResult.totalGradient[i] = -totalStatistics.totalGradient[i] / totalStatistics.totalHessian[i];
			} else {
				leftResult.totalGradient[i] = totalStatistics.totalGradient[i];
				rightResult.totalGradient[i] = totalStatistics.totalGradient[i];
			}
			// Set totalHessian to 0 for classes that not splitting further
			leftResult.totalHessian[i] = 0;
			rightResult.totalHessian[i] = 0;
			leafClassesCount++;
		}
		result += valueCriterion;
	}

	if( leafClassesCount == totalStatistics.ValueSize() ) {
		return false;
	}
	criterion = result * ( 1 + denseTreeBoostCoefficient / ( leafClassesCount + 1 ) );
	return true;
}

inline void CGradientBoostStatisticsMulti::SetSize( int valueSize )
{
	totalGradient.SetSize( valueSize );
	totalHessian.SetSize( valueSize );
}

inline void CGradientBoostStatisticsMulti::NullifyLeafClasses( const CGradientBoostStatisticsMulti& parent )
{
	for( int i = 0; i < parent.totalGradient.Size(); i++ ) {
		if( parent.totalHessian[i] == 0 ) {
			totalGradient[i] = parent.totalGradient[i];
			totalHessian[i] = 0;
		}
	}
}

} // namespace NeoML

