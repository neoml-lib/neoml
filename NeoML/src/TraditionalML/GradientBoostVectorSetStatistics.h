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
	void Add( const CArray<T>& gradient, const CArray<T>& hessian, const CArray<float>& weight, int vectorIndex );
	void Add( const CGradientBoostVectorSetStatistics<T>& other );

	// Deletes a vector
	void Sub( const T& gradient, const T& hessian, float weight );
	void Sub( const CGradientBoostVectorSetStatistics<T>& other );

	// Clears all accumulated data
	void Erase();

	// Calculates the criterion
	double CalcCriterion( float l1, float l2, int classIndex ) const;
	double CalcCriterion( float l1, float l2 ) const;

	// Gets the total gradient
	const T& TotalGradient() const { return totalGradient; }

	// Gets the total hessian
	const T& TotalHessian() const { return totalHessian; }

	// Set total gradient and hessian for class
	void SetGradientAndHessian( const CGradientBoostVectorSetStatistics<T>& other, int classIndex );

	// Gets the total weight
	float TotalWeight() const { return totalWeight; }

	// Set the total weight
	void SetWeight( float weight ) { totalWeight = weight; }


private:
	T totalGradient; // total gradient
	T totalHessian; // total hessian
	float totalWeight; // total weight
};

template<>
inline CGradientBoostVectorSetStatistics<double>::CGradientBoostVectorSetStatistics( int valueSize )
{
	NeoAssert( valueSize == 1 );
	totalGradient = 0.0;
	totalHessian = 0.0;
	totalWeight = 0.0;
}

template<>
inline CGradientBoostVectorSetStatistics<CArray<double>>::CGradientBoostVectorSetStatistics( int valueSize )
{
	totalGradient.Add( 0.0, valueSize );
	totalHessian.Add( 0.0, valueSize );
	totalWeight = 0.0;
}

template<>
inline CGradientBoostVectorSetStatistics<double>::CGradientBoostVectorSetStatistics( const CGradientBoostVectorSetStatistics& other )
{
	totalGradient = other.totalGradient;
	totalHessian = other.totalHessian;
	totalWeight = other.totalWeight;
}

template<>
inline CGradientBoostVectorSetStatistics<CArray<double>>::CGradientBoostVectorSetStatistics( const CGradientBoostVectorSetStatistics& other )
{
	other.totalGradient.CopyTo( totalGradient );
	other.totalHessian.CopyTo( totalHessian );
	totalWeight = other.totalWeight;
}

template<>
inline CGradientBoostVectorSetStatistics<double>& CGradientBoostVectorSetStatistics<double>::operator=( const CGradientBoostVectorSetStatistics& other )
{
	if( &other != this ) {
		totalGradient = other.totalGradient;
		totalHessian = other.totalHessian;
		totalWeight = other.totalWeight;
	}
	return *this;
}

template<>
inline CGradientBoostVectorSetStatistics<CArray<double>>& CGradientBoostVectorSetStatistics<CArray<double>>::operator=( const CGradientBoostVectorSetStatistics& other )
{
	if( &other != this ) {
		other.totalGradient.CopyTo( totalGradient );
		other.totalHessian.CopyTo( totalHessian );
		totalWeight = other.totalWeight;
	}
	return *this;
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Add( const double& gradient, const double& hessian, float weight )
{
	totalGradient += gradient;
	totalHessian += hessian;
	totalWeight += weight;
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Add( const CArray<double>& gradient, const CArray<double>& hessian, float weight )
{
	for( int i = 0; i < totalGradient.Size(); i++ ){
		totalGradient[i] += gradient[i];
		totalHessian[i] += hessian[i];
	}
	totalWeight += weight;
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Add( const CArray<double>& gradient, const CArray<double>& hessian, const CArray<float>& weight, int vectorIndex )
{
	totalGradient += gradient[vectorIndex];
	totalHessian += hessian[vectorIndex];
	totalWeight += weight[vectorIndex];
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Add( const CArray<CArray<double>>& gradient, const CArray<CArray<double>>& hessian, const CArray<float>& weight, int vectorIndex )
{
	for( int i = 0; i < gradient.Size(); i++ ) {
		totalGradient[i] += gradient[i][vectorIndex];
		totalHessian[i] += hessian[i][vectorIndex];
	}
	totalWeight += weight[vectorIndex];
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Add( const CGradientBoostVectorSetStatistics& other )
{
	totalGradient += other.totalGradient;
	totalHessian += other.totalHessian;
	totalWeight += other.totalWeight;
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Add( const CGradientBoostVectorSetStatistics& other )
{
	for( int i = 0; i < totalGradient.Size(); i++ ){
		totalGradient[i] += other.totalGradient[i];
		totalHessian[i] += other.totalHessian[i];
	}
	totalWeight += other.totalWeight;
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Sub( const double& gradient, const double& hessian, float weight )
{
	totalGradient -= gradient;
	totalHessian -= hessian;
	totalWeight -= weight;
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Sub( const CArray<double>& gradient, const CArray<double>& hessian, float weight )
{
	for( int i = 0; i < totalGradient.Size(); i++ ){
		totalGradient[i] -= gradient[i];
		totalHessian[i] -= hessian[i];
	}
	totalWeight -= weight;
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Sub( const CGradientBoostVectorSetStatistics& other )
{
	totalGradient -= other.totalGradient;
	totalHessian -= other.totalHessian;
	totalWeight -= other.totalWeight;
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Sub( const CGradientBoostVectorSetStatistics& other )
{
	for( int i = 0; i < totalGradient.Size(); i++ ){
		totalGradient[i] -= other.totalGradient[i];
		totalHessian[i] -= other.totalHessian[i];
	}
	totalWeight -= other.totalWeight;
}

template<>
inline void CGradientBoostVectorSetStatistics<double>::Erase()
{
	totalGradient = 0.0;
	totalHessian = 0.0;
	totalWeight = 0.0;
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::Erase()
{
	for( int i = 0; i < totalGradient.Size(); i++ ){
		totalGradient[i] = 0.0;
		totalHessian[i] = 0.0;
	}
	totalWeight = 0.0;
}

template<>
inline double CGradientBoostVectorSetStatistics<CArray<double>>::CalcCriterion( float l1, float l2, int classIndex ) const
{
	double temp = totalGradient[classIndex];
	if( temp > l1 ) {
		temp -= l1;
	} else if( temp < -l1 ) {
		temp += l1;
	}
	return temp * temp / ( totalHessian[classIndex] + l2 );
}

template<>
inline double CGradientBoostVectorSetStatistics<double>::CalcCriterion( float l1, float l2 ) const
{
	double temp = totalGradient;
	if( temp > l1 ) {
		temp -= l1;
	}
	else if( temp < -l1 ) {
		temp += l1;
	}
	return temp * temp / ( totalHessian + l2 );
}

template<>
inline double CGradientBoostVectorSetStatistics<CArray<double>>::CalcCriterion( float l1, float l2 ) const
{
	double res = 0;
	for( int i = 0; i < totalGradient.Size(); i++ ){
		res += CalcCriterion( l1, l2, i );
	}
	return res;
}

template<>
inline void CGradientBoostVectorSetStatistics<CArray<double>>::SetGradientAndHessian( const CGradientBoostVectorSetStatistics<CArray<double>>& other,
	int classIndex )
{
	totalGradient[classIndex] = other.totalGradient[classIndex];
	totalHessian[classIndex] = other.totalHessian[classIndex];
}

} // namespace NeoML
