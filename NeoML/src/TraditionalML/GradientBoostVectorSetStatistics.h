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
template<class T>
class CGradientBoostVectorSetStatistics {
public:
	CGradientBoostVectorSetStatistics() = default;
	explicit CGradientBoostVectorSetStatistics( int valueSize );
	explicit CGradientBoostVectorSetStatistics( const CGradientBoostVectorSetStatistics& other );
	CGradientBoostVectorSetStatistics& operator=( const CGradientBoostVectorSetStatistics& other );

	// Adds a vector
	void Add( const T& gradient, const T& hessian, float weight );
	void Add( const CArray<T>& gradients, const CArray<T>& hessians, const CArray<float>& weights, int vectorIndex );
	void Add( const CGradientBoostVectorSetStatistics<T>& other );

	// Deletes a vector
	void Sub( const T& gradient, const T& hessian, float weight );
	void Sub( const CArray<T>& gradients, const CArray<T>& hessians, const CArray<float>& weights, int vectorIndex );
	void Sub( const CGradientBoostVectorSetStatistics<T>& other );

	// Clears all accumulated data
	void Erase();

	// Calculates the criterion
	double CalcCriterion( float l1, float l2, int classIndex ) const;
	double CalcCriterion( float l1, float l2 ) const;

	// Statistics is not enough
	bool StatisticsIsSmall( double minSubsetHessian, double minSubsetWeight, int classIndex ) const;

	T TotalGradient; // total gradient
	T TotalHessian; // total hessian
	float TotalWeight; // total weight
};

template<>
inline CGradientBoostVectorSetStatistics<double>::CGradientBoostVectorSetStatistics( int valueSize )
{
	NeoAssert( valueSize == 1 );
	TotalGradient = 0.0;
	TotalHessian = 0.0;
	TotalWeight = 0.0;
}

template<>
inline CGradientBoostVectorSetStatistics<CArray<double>>::CGradientBoostVectorSetStatistics( int valueSize )
{
	TotalGradient.Add( 0.0, valueSize );
	TotalHessian.Add( 0.0, valueSize );
	TotalWeight = 0.0;
}

template<>
inline CGradientBoostVectorSetStatistics<double>::CGradientBoostVectorSetStatistics( const CGradientBoostVectorSetStatistics& other )
{
	TotalGradient = other.TotalGradient;
	TotalHessian = other.TotalHessian;
	TotalWeight = other.TotalWeight;
}

template<>
inline CGradientBoostVectorSetStatistics<CArray<double>>::CGradientBoostVectorSetStatistics( const CGradientBoostVectorSetStatistics& other )
{
	other.TotalGradient.CopyTo( TotalGradient );
	other.TotalHessian.CopyTo( TotalHessian );
	TotalWeight = other.TotalWeight;
}

template<>
inline CGradientBoostVectorSetStatistics<double>& CGradientBoostVectorSetStatistics<double>::operator=( const CGradientBoostVectorSetStatistics& other )
{
	if( &other != this ) {
		TotalGradient = other.TotalGradient;
		TotalHessian = other.TotalHessian;
		TotalWeight = other.TotalWeight;
	}
	return *this;
}

template<>
inline CGradientBoostVectorSetStatistics<CArray<double>>& CGradientBoostVectorSetStatistics<CArray<double>>::operator=( const CGradientBoostVectorSetStatistics& other )
{
	if( &other != this ) {
		other.TotalGradient.CopyTo( TotalGradient );
		other.TotalHessian.CopyTo( TotalHessian );
		TotalWeight = other.TotalWeight;
	}
	return *this;
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Add( const double& gradient, const double& hessian, float weight )
{
	TotalGradient += gradient;
	TotalHessian += hessian;
	TotalWeight += weight;
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Add( const CArray<double>& gradient, const CArray<double>& hessian, float weight )
{
	for( int i = 0; i < TotalGradient.Size(); i++ ){
		TotalGradient[i] += gradient[i];
		TotalHessian[i] += hessian[i];
	}
	TotalWeight += weight;
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Add( const CArray<double>& gradients, const CArray<double>& hessians,
	const CArray<float>& weights, int vectorIndex )
{
	TotalGradient += gradients[vectorIndex];
	TotalHessian += hessians[vectorIndex];
	TotalWeight += weights[vectorIndex];
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Add( const CArray<CArray<double>>& gradients, const CArray<CArray<double>>& hessians,
	const CArray<float>& weights, int vectorIndex )
{
	for( int i = 0; i < gradients.Size(); i++ ){
		TotalGradient[i] += gradients[i][vectorIndex];
		TotalHessian[i] += hessians[i][vectorIndex];
	}
	TotalWeight += weights[vectorIndex];
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Add( const CGradientBoostVectorSetStatistics& other )
{
	TotalGradient += other.TotalGradient;
	TotalHessian += other.TotalHessian;
	TotalWeight += other.TotalWeight;
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Add( const CGradientBoostVectorSetStatistics& other )
{
	for( int i = 0; i < TotalGradient.Size(); i++ ){
		TotalGradient[i] += other.TotalGradient[i];
		TotalHessian[i] += other.TotalHessian[i];
	}
	TotalWeight += other.TotalWeight;
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Sub( const double& gradient, const double& hessian, float weight )
{
	TotalGradient -= gradient;
	TotalHessian -= hessian;
	TotalWeight -= weight;
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Sub( const CArray<double>& gradient, const CArray<double>& hessian, float weight )
{
	for( int i = 0; i < TotalGradient.Size(); i++ ){
		TotalGradient[i] -= gradient[i];
		TotalHessian[i] -= hessian[i];
	}
	TotalWeight -= weight;
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Sub( const CArray<double>& gradients, const CArray<double>& hessians,
	const CArray<float>& weights, int vectorIndex )
{
	TotalGradient -= gradients[vectorIndex];
	TotalHessian -= hessians[vectorIndex];
	TotalWeight -= weights[vectorIndex];
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Sub( const CArray<CArray<double>>& gradients, const CArray<CArray<double>>& hessians,
	const CArray<float>& weights, int vectorIndex )
{
	for( int i = 0; i < gradients.Size(); i++ ){
		TotalGradient[i] -= gradients[i][vectorIndex];
		TotalHessian[i] -= hessians[i][vectorIndex];
	}
	TotalWeight -= weights[vectorIndex];
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Sub( const CGradientBoostVectorSetStatistics& other )
{
	TotalGradient -= other.TotalGradient;
	TotalHessian -= other.TotalHessian;
	TotalWeight -= other.TotalWeight;
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Sub( const CGradientBoostVectorSetStatistics& other )
{
	for( int i = 0; i < TotalGradient.Size(); i++ ){
		TotalGradient[i] -= other.TotalGradient[i];
		TotalHessian[i] -= other.TotalHessian[i];
	}
	TotalWeight -= other.TotalWeight;
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Erase()
{
	TotalGradient = 0.0;
	TotalHessian = 0.0;
	TotalWeight = 0.0;
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Erase()
{
	for( int i = 0; i < TotalGradient.Size(); i++ ){
		TotalGradient[i] = 0.0;
		TotalHessian[i] = 0.0;
	}
	TotalWeight = 0.0;
}

template<>
inline double CGradientBoostVectorSetStatistics<double>::CalcCriterion( float l1, float l2, int classIndex ) const
{
	NeoAssert( classIndex == 0 );
	return CalcCriterion( l1, l2 );
}

template<>
inline double CGradientBoostVectorSetStatistics<CArray<double>>::CalcCriterion( float l1, float l2, int classIndex ) const
{
	double temp = TotalGradient[classIndex];
	if( temp > l1 ) {
		temp -= l1;
	} else if( temp < -l1 ) {
		temp += l1;
	}
	return temp * temp / (TotalHessian[classIndex] + l2);
}

template<>
inline double CGradientBoostVectorSetStatistics<double>::CalcCriterion( float l1, float l2 ) const
{
	double temp = TotalGradient;
	if( temp > l1 ) {
		temp -= l1;
	}
	else if( temp < -l1 ) {
		temp += l1;
	}
	return temp * temp / (TotalHessian + l2);
}

template<>
inline double CGradientBoostVectorSetStatistics<CArray<double>>::CalcCriterion( float l1, float l2 ) const
{
	double res = 0;
	for( int i = 0; i < TotalGradient.Size(); i++ ){
		res += CalcCriterion( l1, l2, i);
	}
	return res;
}

template<>
inline bool CGradientBoostVectorSetStatistics<double>::StatisticsIsSmall( double minSubsetHessian, double minSubsetWeight, int classIndex ) const
{
	NeoAssert( classIndex == 0 );
	return TotalHessian < minSubsetHessian || TotalWeight < minSubsetWeight;
}

template<>
inline bool CGradientBoostVectorSetStatistics<CArray<double>>::StatisticsIsSmall( double minSubsetHessian, double minSubsetWeight, int classIndex ) const
{
	return TotalHessian[classIndex] < minSubsetHessian || TotalWeight < minSubsetWeight;
}

} // namespace NeoML
