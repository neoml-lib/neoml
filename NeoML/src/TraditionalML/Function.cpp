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

#include <NeoML/TraditionalML/Function.h>
#include <NeoMathEngine/OpenMP.h>

namespace NeoML {

// Calculates the L1 regularization factor
static void calcL1Regularization( const CFloatVector& w, float l1Coeff, double& value, CFloatVector& gradient )
{
	value = 0;
	for( int i = 0; i < w.Size(); i++ ) {
		float z = w[i];
		if( abs(z) < l1Coeff ) {
			value += z * z / 2;
			gradient.SetAt( i, z );
		} else {
			value += l1Coeff * ( abs(z) - l1Coeff / 2 );
			gradient.SetAt( i, l1Coeff * z / abs(z) );
		}
	}
}

//------------------------------------------------------------------------------------------------------------

// Multiplies hessian by vector
static CFloatVector calcHessianProduct( int threadCount, const CSparseFloatMatrixDesc& matrix, const CFloatVector& arg,
	float errorWeight, const CArray<double>& hessian )
{
	const int vectorCount = matrix.Height;

	CFloatVector result = arg / errorWeight;
	result.SetAt( result.Size() - 1, 0 );

	const int curThreadCount = IsOmpRelevant( vectorCount ) ? threadCount : 1;

	CArray<CFloatVector> resultReduction;
	resultReduction.Add( CFloatVector( result.Size() ), curThreadCount );

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		const int threadNumber = OmpGetThreadNum();
		CFloatVector& privateResult = resultReduction[threadNumber];
		privateResult.Nullify();

		int index = 0;
		int count = 0;
		if( OmpGetTaskIndexAndCount( vectorCount, index, count ) ) {
			for( int i = 0; i < count; i++ ) {
				if( hessian[index] != 0 ) {
					CSparseFloatVectorDesc desc;
					matrix.GetRow( index, desc );

					double temp = LinearFunction( arg, desc );
					temp *= hessian[index];

					privateResult.MultiplyAndAddExt( desc, temp );
				}

				index++;
			}
		}
	}

	for( int i = 0; i < resultReduction.Size(); i++ ) {
		result += resultReduction[i];
	}

	return result;
}

//------------------------------------------------------------------------------------------------------------

CSquaredHinge::CSquaredHinge( const IProblem& data, double _errorWeight, float _l1Coeff, int _threadCount ) :
	matrix( data.GetMatrix() ),
	errorWeight( static_cast<float>( _errorWeight ) ),
	l1Coeff( _l1Coeff ),
	threadCount( _threadCount ),
	value( 0.f ),
	answers( data.GetVectorCount() ),
	weights( data.GetVectorCount() )
{
	float* answersPtr = answers.CopyOnWrite();
	float* weightsPtr = weights.CopyOnWrite();
	for( int i = 0; i < matrix.Height; i++ ) {
		answersPtr[i] = static_cast<float>( data.GetBinaryClass( i ) );
		weightsPtr[i] = static_cast<float>( data.GetVectorWeight( i ) );
	}
}

void CSquaredHinge::SetArgument( const CFloatVector& w )
{
	NeoAssert( w.Size() == NumberOfDimensions() );

	value = 0;
	gradient = w;
	gradient.SetAt( gradient.Size() - 1, 0 );

	if( l1Coeff > 0 ) {
		calcL1Regularization( gradient, l1Coeff, value, gradient );
	} else {
		value = DotProduct( gradient, gradient ) / 2;
	}

	gradient /= errorWeight;
	value /= errorWeight;

	CFloatVector arg = w;

	const int vectorCount = matrix.Height;
	const int curThreadCount = IsOmpRelevant( matrix.Height ) ? threadCount : 1;

	CArray<CFloatVector> gradientReduction;
	gradientReduction.Add( CFloatVector( gradient.Size() ), curThreadCount );
	CArray<double> valueReduction;
	valueReduction.Add( 0., curThreadCount );

	const float* answersPtr = answers.GetPtr();
	const float* weightsPtr = weights.GetPtr();
	hessian.SetSize( vectorCount );

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		const int threadNumber = OmpGetThreadNum();
		double* valuePtr = valueReduction.GetPtr() + threadNumber;
		gradientReduction[threadNumber].Nullify();
		CFloatVector& gradientPrivate = gradientReduction[threadNumber];

		int index = 0;
		int count = 0;
		if( OmpGetTaskIndexAndCount( vectorCount, index, count ) ) {
			for( int i = 0; i < count; i++ ) {
				double answer = answersPtr[index];
				double weight = weightsPtr[index];
				CSparseFloatVectorDesc desc;
				matrix.GetRow( index, desc );

				double x = answer * LinearFunction( arg, desc );
				double d = 1 - x;

				if( x < 1 ) {
					*valuePtr += weight * d * d;
					gradientPrivate.MultiplyAndAddExt( desc, -weight * answer * d * 2 );

					hessian[index] = weight * 2;
				} else {
					hessian[index] = 0;
				}

				index++;
			}
		}
	}

	for( int i = 0; i < curThreadCount; i++ ) {
		value += valueReduction[i];
		gradient += gradientReduction[i];
	}
}

CFloatVector CSquaredHinge::HessianProduct( const CFloatVector& arg )
{
	return calcHessianProduct( threadCount, matrix, arg, errorWeight, hessian );
}

//-----------------------------------------------------------------------------------------------------------------------

CL2Regression::CL2Regression( const IRegressionProblem& data, double errorWeight, double _p, float _l1Coeff, int _threadCount ) :
	matrix( data.GetMatrix() ),
	errorWeight( static_cast<float>( errorWeight ) ),
	p( static_cast<float>( _p ) ),
	l1Coeff(_l1Coeff ),
	threadCount( _threadCount ),
	value( 0.f ),
	answers( data.GetVectorCount() ),
	weights( data.GetVectorCount() )
{
	float* answersPtr = answers.CopyOnWrite();
	float* weightsPtr = weights.CopyOnWrite();
	for( int i = 0; i < matrix.Height; i++ ) {
		answersPtr[i] = static_cast<float>( data.GetValue( i ) );
		weightsPtr[i] = static_cast<float>( data.GetVectorWeight( i ) );
	}
}

void CL2Regression::SetArgument( const CFloatVector& w )
{
	NeoAssert( w.Size() == NumberOfDimensions() );

	gradient = w;
	gradient.SetAt( gradient.Size() - 1, 0 ); // don't take the regularization bias into account

	if( l1Coeff > 0 ) {
		calcL1Regularization( gradient, l1Coeff, value, gradient );
	} else {
		value = DotProduct( gradient, gradient ) / 2;
	}
	value = value / errorWeight;
	gradient = gradient / errorWeight;

	CFloatVector arg = w;

	const int vectorCount = matrix.Height;
	const int curThreadCount = IsOmpRelevant( matrix.Height ) ? threadCount : 1;

	CArray<CFloatVector> gradientReduction;
	gradientReduction.Add( CFloatVector( gradient.Size() ), curThreadCount );
	CArray<double> valueReduction;
	valueReduction.Add( 0.f, curThreadCount );

	const float* answersPtr = answers.GetPtr();
	const float* weightsPtr = weights.GetPtr();
	hessian.SetSize( vectorCount );

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		const int threadNumber = OmpGetThreadNum();
		double* valuePtr = valueReduction.GetPtr() + threadNumber;
		gradientReduction[threadNumber].Nullify();
		CFloatVector& gradientPrivate = gradientReduction[threadNumber];

		int index = 0;
		int count = 0;
		if( OmpGetTaskIndexAndCount( vectorCount, index, count ) ) {
			for( int i = 0; i < count; i++ ) {
				float weight = weightsPtr[index];
				CSparseFloatVectorDesc desc;
				matrix.GetRow( index, desc );

				double d = LinearFunction( arg, desc ) - answersPtr[index];

				if( d < -p ) {
					*valuePtr += weight * ( d + p ) * ( d + p );
					hessian[index] = weight * 2;
					gradientPrivate.MultiplyAndAddExt( desc, weight * ( d + p ) * 2 );
				} else {
					*valuePtr += weight * ( d - p ) * ( d - p );

					if( d > p ) {
						hessian[index] = weight * 2;
						gradientPrivate.MultiplyAndAddExt( desc, weight * ( d - p ) * 2 );
					} else {
						hessian[index] = 0.f;
					}
				}

				index++;
			}
		}
	}

	for( int i = 0; i < curThreadCount; i++ ) {
		gradient += gradientReduction[i];
		value += valueReduction[i];
	}
}

CFloatVector CL2Regression::HessianProduct( const CFloatVector& arg )
{
	return calcHessianProduct( threadCount, matrix, arg, errorWeight, hessian );
}

//-----------------------------------------------------------------------------------------------------------------------

CLogRegression::CLogRegression( const IProblem& data, double _errorWeight, float _l1Coeff, int _threadCount ) :
	matrix( data.GetMatrix() ),
	errorWeight( static_cast<float>( _errorWeight ) ),
	l1Coeff( _l1Coeff ),
	threadCount( _threadCount ),
	value( 0.f ),
	answers( data.GetVectorCount() ),
	weights( data.GetVectorCount() )
{
	float* answersPtr = answers.CopyOnWrite();
	float* weightsPtr = weights.CopyOnWrite();
	for( int i = 0; i < matrix.Height; i++ ) {
		answersPtr[i] = static_cast<float>( data.GetBinaryClass( i ) );
		weightsPtr[i] = static_cast<float>( data.GetVectorWeight( i ) );
	}
}

void CLogRegression::SetArgument( const CFloatVector& w )
{
	NeoAssert( w.Size() == NumberOfDimensions() );

	const float logNormalizer = 1.f / logf( 2.f );

	gradient = w;
	gradient.SetAt( gradient.Size() - 1, 0 ); // do not take bias into account for regularization

	value = 0;
	double rValue = 0;
	if( l1Coeff > 0 ) {
		calcL1Regularization( gradient, l1Coeff, rValue, gradient );
	} else {
		rValue = DotProduct( gradient, gradient ) / 2;
	}
	rValue = rValue / errorWeight;
	gradient = gradient / errorWeight;

	CFloatVector arg = w;

	const int vectorCount = matrix.Height;
	const int curThreadCount = IsOmpRelevant( matrix.Height ) ? threadCount : 1;

	CArray<CFloatVector> gradientReduction;
	gradientReduction.Add( CFloatVector( gradient.Size() ), curThreadCount );
	CArray<double> valueReduction;
	valueReduction.Add( 0.f, curThreadCount );

	const float* answersPtr = answers.GetPtr();
	const float* weightsPtr = weights.GetPtr();
	hessian.SetSize( vectorCount );

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		const int threadNumber = OmpGetThreadNum();
		double* valuePtr = valueReduction.GetPtr() + threadNumber;
		gradientReduction[threadNumber].Nullify();
		CFloatVector& gradientPrivate = gradientReduction[threadNumber];

		int index = 0;
		int count = 0;
		if( OmpGetTaskIndexAndCount( vectorCount, index, count ) ) {
			for( int i = 0; i < count; i++ ) {
				double answer = answersPtr[index];
				double weight = weightsPtr[index];
				CSparseFloatVectorDesc desc;
				matrix.GetRow( index, desc );

				double dot = LinearFunction( arg, desc );
				double expCoeff = exp( -answer * dot );

				*valuePtr += weight * log( 1.f + expCoeff );

				gradientPrivate.MultiplyAndAddExt( desc, -weight * logNormalizer * answer * expCoeff / ( 1.f + expCoeff ) );
				hessian[index] = weight * logNormalizer * expCoeff / ( 1.f + expCoeff ) / ( 1.f + expCoeff );

				index++;
			}
		}
	}

	for( int i = 0; i < curThreadCount; i++ ) {
		gradient += gradientReduction[i];
		value += valueReduction[i];
	}

	value *= logNormalizer;
	value += rValue;
}

CFloatVector CLogRegression::HessianProduct( const CFloatVector& arg )
{
	return calcHessianProduct( threadCount, matrix, arg, errorWeight, hessian );
}

//-----------------------------------------------------------------------------------------------------------------------

CSmoothedHinge::CSmoothedHinge( const IProblem& data, double _errorWeight, float _l1Coeff, int _threadCount ) :
	matrix( data.GetMatrix() ),
	errorWeight( static_cast<float>( _errorWeight ) ),
	l1Coeff( _l1Coeff ),
	threadCount( _threadCount ),
	value( 0.f ),
	answers( data.GetVectorCount() ),
	weights( data.GetVectorCount() )
{
	float* answersPtr = answers.CopyOnWrite();
	float* weightsPtr = weights.CopyOnWrite();
	for( int i = 0; i < matrix.Height; i++ ) {
		answersPtr[i] = static_cast<float>( data.GetBinaryClass( i ) );
		weightsPtr[i] = static_cast<float>( data.GetVectorWeight( i ) );
	}
}

void CSmoothedHinge::SetArgument( const CFloatVector& w )
{
	NeoAssert( w.Size() == NumberOfDimensions() );

	gradient = w;
	gradient.SetAt( gradient.Size() - 1, 0 ); // don't take the regularization bias into account

	if( l1Coeff > 0 ) {
		calcL1Regularization( gradient, l1Coeff, value, gradient );
	} else {
		value = DotProduct( gradient, gradient ) / 2;
	}
	value = value / errorWeight;
	gradient = gradient / errorWeight;

	CFloatVector arg = w;

	const int vectorCount = matrix.Height;
	const int curThreadCount = IsOmpRelevant( matrix.Height ) ? threadCount : 1;

	CArray<CFloatVector> gradientReduction;
	gradientReduction.Add( CFloatVector( gradient.Size() ), curThreadCount );
	CArray<double> valueReduction;
	valueReduction.Add( 0.f, curThreadCount );

	const float* answersPtr = answers.GetPtr();
	const float* weightsPtr = weights.GetPtr();
	hessian.SetSize( vectorCount );

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		const int threadNumber = OmpGetThreadNum();
		double* valuePtr = valueReduction.GetPtr() + threadNumber;
		gradientReduction[threadNumber].Nullify();
		CFloatVector& gradientPrivate = gradientReduction[threadNumber];

		int index = 0;
		int count = 0;
		if( OmpGetTaskIndexAndCount( vectorCount, index, count ) ) {
			for( int i = 0; i < count; i++ ) {
				float answer = answersPtr[index];
				float weight = weightsPtr[index];
				CSparseFloatVectorDesc desc;
				matrix.GetRow( index, desc );

				double d = answer * LinearFunction( arg, desc ) - 1;

				if( d < 0 ) {
					const float sqrtValue = static_cast<float>( sqrt( d * d + 1 ) );
					*valuePtr += weight * ( sqrtValue - 1 );

					gradientPrivate.MultiplyAndAddExt( desc, weight * answer * d / sqrtValue );

					hessian[index] = weight / ( ( d * d + 1 ) * sqrtValue );
				} else {
					hessian[index] = 0;
				}

				index++;
			}
		}
	}

	for( int i = 0; i < curThreadCount; i++ ) {
		gradient += gradientReduction[i];
		value += valueReduction[i];
	}
}

CFloatVector CSmoothedHinge::HessianProduct( const CFloatVector& arg )
{
	return calcHessianProduct( threadCount, matrix, arg, errorWeight, hessian );
}

} // namespace NeoML
