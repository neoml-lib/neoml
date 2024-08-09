/* Copyright © 2017-2023 ABBYY

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
#include <NeoMathEngine/ThreadPool.h>

namespace NeoML {

// Exponent function with limitations to avoid NaN
static inline double exponentFunc( double f )
{
	constexpr double DBL_LOG_MAX = 709.;
	constexpr double DBL_LOG_MIN = -709.;

	if( f < DBL_LOG_MIN ) {
		return 0;
	} else if( f > DBL_LOG_MAX ) {
		return DBL_MAX;
	} else {
		return exp( f );
	}
}

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
static CFloatVector calcHessianProduct( IThreadPool& threadPool, const CFloatMatrixDesc& matrix, const CFloatVector& arg,
	float errorWeight, const CArray<double>& hessian )
{
	CFloatVector result = arg / errorWeight;
	result.SetAt( result.Size() - 1, 0 );

	struct CFunctionParams {
		const CFloatMatrixDesc& Matrix;
		const CFloatVector& Arg;
		const CArray<double>& Hessian;
		CArray<CFloatVector> ResultReduction;

		CFunctionParams( int threadCount, const CFloatMatrixDesc& matrix, const CFloatVector& arg, const CArray<double>& hessian ) :
			Matrix( matrix ),
			Arg( arg ),
			Hessian( hessian )
		{
			ResultReduction.Add( CFloatVector( arg.Size() ), threadCount );
		}
	} params( threadPool.Size(), matrix, arg, hessian);

	IThreadPool::TFunction f = [] ( int threadIndex, void* paramPtr )
	{
		CFunctionParams& params = *( CFunctionParams* )paramPtr;
		const CFloatMatrixDesc& matrix = params.Matrix;
		const CFloatVector& arg = params.Arg;
		const CArray<double>& hessian = params.Hessian;

		const int threadCount = params.ResultReduction.Size();
		CFloatVector& privateResult = params.ResultReduction[threadIndex];
		privateResult.Nullify();

		int index = 0;
		int count = 0;
		if( GetTaskIndexAndCount( threadCount, threadIndex, matrix.Height, index, count ) ) {
			for( int i = 0; i < count; i++ ) {
				if( hessian[index] != 0 ) {
					CFloatVectorDesc desc;
					matrix.GetRow( index, desc );

					double temp = LinearFunction( arg, desc );
					temp *= hessian[index];

					privateResult.MultiplyAndAddExt( desc, temp );
				}

				index++;
			}
		}
	};

	NEOML_NUM_THREADS( threadPool, &params, f );

	for( const CFloatVector& resultPerThread : params.ResultReduction ) {
		result += resultPerThread;
	}

	return result;
}

//------------------------------------------------------------------------------------------------------------

// Struct which is used to pass parameters of CFunction::SetArgument to multithreading (IThreadPool)
struct CSetArgumentParams {
	const float* Answers;
	const float* Weights;
	const CFloatMatrixDesc& Matrix;
	const CFloatVector& Arg;
	CArray<double>& Hessian;
	CArray<CFloatVector> GradientReduction;
	CArray<double> ValueReduction;
	const float P;

	CSetArgumentParams( int threadCount, const float* answersPtr, const float* weightsPtr,
		const CFloatMatrixDesc& matrix, const CFloatVector& arg, CArray<double>& hessian, float p = 0.f ) :
		Answers( answersPtr ),
		Weights( weightsPtr ),
		Matrix( matrix ),
		Arg( arg ),
		Hessian( hessian ),
		P( p )
	{
		GradientReduction.Add( CFloatVector( arg.Size() ), threadCount );
		ValueReduction.Add( 0., threadCount );
	}
};

//------------------------------------------------------------------------------------------------------------

CSquaredHinge::CSquaredHinge( const IProblem& data, double _errorWeight, float _l1Coeff, int threadCount ) :
	matrix( data.GetMatrix() ),
	errorWeight( static_cast<float>( _errorWeight ) ),
	l1Coeff( _l1Coeff ),
	threadPool( CreateThreadPool( threadCount ) ),
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

CSquaredHinge::~CSquaredHinge()
{
	delete threadPool;
}

void CSquaredHinge::SetArgument( const CFloatVector& arg )
{
	NeoAssert( arg.Size() == NumberOfDimensions() );

	value = 0;
	gradient = arg;
	gradient.SetAt( gradient.Size() - 1, 0 );

	if( l1Coeff > 0 ) {
		calcL1Regularization( gradient, l1Coeff, value, gradient );
	} else {
		value = DotProduct( gradient, gradient ) / 2;
	}

	gradient /= errorWeight;
	value /= errorWeight;

	hessian.SetSize( matrix.Height );

	CSetArgumentParams params( threadPool->Size(), answers.GetPtr(), weights.GetPtr(), matrix, arg, hessian );

	IThreadPool::TFunction f = []( int threadIndex, void* paramPtr )
	{
		CSetArgumentParams& params = *( CSetArgumentParams* )paramPtr;
		const CFloatMatrixDesc& matrix = params.Matrix;
		const CFloatVector& arg = params.Arg;
		CArray<double>& hessian = params.Hessian;
		const int threadCount = params.ValueReduction.Size();

		double& valuePrivate = params.ValueReduction[threadIndex];
		CFloatVector& gradientPrivate = params.GradientReduction[threadIndex];
		gradientPrivate.Nullify();

		int index = 0;
		int count = 0;
		if( GetTaskIndexAndCount( threadCount, threadIndex, matrix.Height, index, count ) ) {
			for( int i = 0; i < count; i++ ) {
				double answer = params.Answers[index];
				double weight = params.Weights[index];
				CFloatVectorDesc desc;
				matrix.GetRow( index, desc );

				double x = answer * LinearFunction( arg, desc );
				double d = 1 - x;

				if( x < 1 ) {
					valuePrivate += weight * d * d;
					gradientPrivate.MultiplyAndAddExt( desc, -weight * answer * d * 2 );

					hessian[index] = weight * 2;
				} else {
					hessian[index] = 0;
				}

				index++;
			}
		}
	};

	NEOML_NUM_THREADS( *threadPool, &params, f );

	for( int i = 0; i < params.ValueReduction.Size(); i++ ) {
		value += params.ValueReduction[i];
		gradient += params.GradientReduction[i];
	}
}

CFloatVector CSquaredHinge::HessianProduct( const CFloatVector& arg )
{
	return calcHessianProduct( *threadPool, matrix, arg, errorWeight, hessian );
}

//-----------------------------------------------------------------------------------------------------------------------

CL2Regression::CL2Regression( const IRegressionProblem& data, double errorWeight, double _p, float _l1Coeff, int threadCount ) :
	matrix( data.GetMatrix() ),
	errorWeight( static_cast<float>( errorWeight ) ),
	p( static_cast<float>( _p ) ),
	l1Coeff(_l1Coeff ),
	threadPool( CreateThreadPool( threadCount ) ),
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

CL2Regression::~CL2Regression()
{
	delete threadPool;
}

void CL2Regression::SetArgument( const CFloatVector& arg )
{
	NeoAssert( arg.Size() == NumberOfDimensions() );

	gradient = arg;
	gradient.SetAt( gradient.Size() - 1, 0 ); // don't take the regularization bias into account

	if( l1Coeff > 0 ) {
		calcL1Regularization( gradient, l1Coeff, value, gradient );
	} else {
		value = DotProduct( gradient, gradient ) / 2;
	}
	value = value / errorWeight;
	gradient = gradient / errorWeight;

	hessian.SetSize( matrix.Height );

	CSetArgumentParams params( threadPool->Size(), answers.GetPtr(), weights.GetPtr(), matrix, arg, hessian, p );

	IThreadPool::TFunction f = []( int threadIndex, void* paramPtr )
	{
		CSetArgumentParams& params = *( CSetArgumentParams* )paramPtr;
		const CFloatMatrixDesc& matrix = params.Matrix;
		const CFloatVector& arg = params.Arg;
		CArray<double>& hessian = params.Hessian;
		const int threadCount = params.ValueReduction.Size();
		float p = params.P;

		double& valuePrivate = params.ValueReduction[threadIndex];
		CFloatVector& gradientPrivate = params.GradientReduction[threadIndex];
		gradientPrivate.Nullify();

		int index = 0;
		int count = 0;
		if( GetTaskIndexAndCount( threadCount, threadIndex, matrix.Height, index, count ) ) {
			for( int i = 0; i < count; i++ ) {
				float weight = params.Weights[index];
				CFloatVectorDesc desc;
				matrix.GetRow( index, desc );

				double d = LinearFunction( arg, desc ) - params.Answers[index];

				if( d < -p ) {
					valuePrivate += weight * ( d + p ) * ( d + p );
					hessian[index] = weight * 2;
					gradientPrivate.MultiplyAndAddExt( desc, weight * ( d + p ) * 2 );
				} else {
					valuePrivate += weight * ( d - p ) * ( d - p );

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
	};

	NEOML_NUM_THREADS( *threadPool, &params, f );

	for( int i = 0; i < params.ValueReduction.Size(); i++ ) {
		gradient += params.GradientReduction[i];
		value += params.ValueReduction[i];
	}
}

CFloatVector CL2Regression::HessianProduct( const CFloatVector& arg )
{
	return calcHessianProduct( *threadPool, matrix, arg, errorWeight, hessian );
}

//-----------------------------------------------------------------------------------------------------------------------

CLogRegression::CLogRegression( const IProblem& data, double _errorWeight, float _l1Coeff, int threadCount ) :
	matrix( data.GetMatrix() ),
	errorWeight( static_cast<float>( _errorWeight ) ),
	l1Coeff( _l1Coeff ),
	threadPool( CreateThreadPool( threadCount ) ),
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

CLogRegression::~CLogRegression()
{
	delete threadPool;
}

void CLogRegression::SetArgument( const CFloatVector& arg )
{
	NeoAssert( arg.Size() == NumberOfDimensions() );

	gradient = arg;
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

	hessian.SetSize( matrix.Height );

	CSetArgumentParams params( threadPool->Size(), answers.GetPtr(), weights.GetPtr(), matrix, arg, hessian );

	IThreadPool::TFunction f = []( int threadIndex, void* paramPtr )
	{
		CSetArgumentParams& params = *( CSetArgumentParams* )paramPtr;
		const CFloatMatrixDesc& matrix = params.Matrix;
		const CFloatVector& arg = params.Arg;
		CArray<double>& hessian = params.Hessian;
		const int threadCount = params.ValueReduction.Size();

		double& valuePrivate = params.ValueReduction[threadIndex];
		CFloatVector& gradientPrivate = params.GradientReduction[threadIndex];
		gradientPrivate.Nullify();

		int index = 0;
		int count = 0;
		const float logNormalizer = 1.f / logf( 2.f );
		if( GetTaskIndexAndCount( threadCount, threadIndex, matrix.Height, index, count ) ) {
			for( int i = 0; i < count; i++ ) {
				double answer = params.Answers[index];
				double weight = params.Weights[index];
				CFloatVectorDesc desc;
				matrix.GetRow( index, desc );

				double dot = LinearFunction( arg, desc );
				double expCoeff = exponentFunc( -answer * dot );

				valuePrivate += weight * log1p( expCoeff );

				bool isNaN = false;
				for( int k = 0; k < desc.Size; ++k ) {
					if( desc.Values[k] != desc.Values[k] ) {
						isNaN = true;
						break;
					}
				}

				static bool isPrinted = false;
				if( !isPrinted && ( isNaN || expCoeff != expCoeff || valuePrivate != valuePrivate || dot != dot ) ) {
					printf( " dot = %lf, exp = %lf, v = %lf, %lf \n", dot, expCoeff, valuePrivate, ( -weight * logNormalizer * answer * expCoeff / ( 1.f + expCoeff ) ) );
					printf( " index = %d, desc = { ", index );
					for( int k = 0; k < desc.Size; ++k ) {
						printf( "(%d %f) ", desc.Indexes[k], desc.Values[k] );
					}
					printf( "}\n" );
					isPrinted = true;
				}

				gradientPrivate.MultiplyAndAddExt( desc, -weight * logNormalizer * answer * expCoeff / ( 1.f + expCoeff ) );
				hessian[index] = weight * logNormalizer * expCoeff / ( 1.f + expCoeff ) / ( 1.f + expCoeff );

				index++;
			}
		}
	};

	NEOML_NUM_THREADS( *threadPool, &params, f );

	for( int i = 0; i < params.ValueReduction.Size(); i++ ) {
		gradient += params.GradientReduction[i];
		value += params.ValueReduction[i];
	}

	value /= logf( 2.f );
	value += rValue;

	auto fContainsNaN = []( const CFloatVector& v ) -> bool 
	{
		for( float f : v ) {
			if( f != f )
				return true;
		}
		return false;
	};

	auto dContainsNaN = []( const CArray<double> & v ) -> bool
	{
		for( double f : v ) {
			if( f != f )
				return true;
		}
		return false;
	};

	auto printV = []( const CFloatVector& v, const char* name )
	{
		printf( " %s = { ", name );
		for( float f : v ) {
			printf( "%f ", f );
		}
		printf( "}\n" );
	};

	static bool isPrinted = false;
	if( isPrinted ) {
		return;
	}
	if( fContainsNaN( arg )
		|| fContainsNaN( answers )
		|| fContainsNaN( weights )
		|| fContainsNaN( gradient )
		|| dContainsNaN( hessian )
		)
	{
		isPrinted = true;
		printf( "CLogRegression: value = %lf \n", value );

		printV( arg, "arg" );
		printV( answers, "answers" );
		printV( weights, "weights" );
		printV( gradient, "gradient" );

		printf( " hessian = { " );
		for( double f : hessian ) {
			printf( "%lf ", f );
		}
		printf( "}\n" );
	}
}

CFloatVector CLogRegression::HessianProduct( const CFloatVector& arg )
{
	return calcHessianProduct( *threadPool, matrix, arg, errorWeight, hessian );
}

//-----------------------------------------------------------------------------------------------------------------------

CSmoothedHinge::CSmoothedHinge( const IProblem& data, double _errorWeight, float _l1Coeff, int threadCount ) :
	matrix( data.GetMatrix() ),
	errorWeight( static_cast<float>( _errorWeight ) ),
	l1Coeff( _l1Coeff ),
	threadPool( CreateThreadPool( threadCount ) ),
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

CSmoothedHinge::~CSmoothedHinge()
{
	delete threadPool;
}

void CSmoothedHinge::SetArgument( const CFloatVector& arg )
{
	NeoAssert( arg.Size() == NumberOfDimensions() );

	gradient = arg;
	gradient.SetAt( gradient.Size() - 1, 0 ); // don't take the regularization bias into account

	if( l1Coeff > 0 ) {
		calcL1Regularization( gradient, l1Coeff, value, gradient );
	} else {
		value = DotProduct( gradient, gradient ) / 2;
	}
	value = value / errorWeight;
	gradient = gradient / errorWeight;

	hessian.SetSize( matrix.Height );

	CSetArgumentParams params( threadPool->Size(), answers.GetPtr(), weights.GetPtr(), matrix, arg, hessian );

	IThreadPool::TFunction f = []( int threadIndex, void* paramPtr )
	{
		CSetArgumentParams& params = *( CSetArgumentParams* )paramPtr;
		const CFloatMatrixDesc& matrix = params.Matrix;
		const CFloatVector& arg = params.Arg;
		CArray<double>& hessian = params.Hessian;
		const int threadCount = params.ValueReduction.Size();

		double& valuePrivate = params.ValueReduction[threadIndex];
		CFloatVector& gradientPrivate = params.GradientReduction[threadIndex];
		gradientPrivate.Nullify();

		int index = 0;
		int count = 0;
		if( GetTaskIndexAndCount( threadCount, threadIndex, matrix.Height, index, count ) ) {
			for( int i = 0; i < count; i++ ) {
				float answer = params.Answers[index];
				float weight = params.Weights[index];
				CFloatVectorDesc desc;
				matrix.GetRow( index, desc );

				double d = answer * LinearFunction( arg, desc ) - 1;

				if( d < 0 ) {
					const float sqrtValue = static_cast<float>( sqrt( d * d + 1 ) );
					valuePrivate += weight * ( sqrtValue - 1 );

					gradientPrivate.MultiplyAndAddExt( desc, weight * answer * d / sqrtValue );

					hessian[index] = weight / ( ( d * d + 1 ) * sqrtValue );
				} else {
					hessian[index] = 0;
				}

				index++;
			}
		}
	};

	NEOML_NUM_THREADS( *threadPool, &params, f );

	for( int i = 0; i < params.ValueReduction.Size(); i++ ) {
		gradient += params.GradientReduction[i];
		value += params.ValueReduction[i];
	}
}

CFloatVector CSmoothedHinge::HessianProduct( const CFloatVector& arg )
{
	return calcHessianProduct( *threadPool, matrix, arg, errorWeight, hessian );
}

} // namespace NeoML
