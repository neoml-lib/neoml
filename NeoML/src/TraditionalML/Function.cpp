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

// Calculates the L1 regularization factor
static void calcL1Regularization( const CFloatVector& w, float l1Coeff, double& value, CFloatVector& gradient )
{
	value = 0;
	for( int i = 0; i < w.Size(); ++i ) {
		float z = w[i];
		if( abs( z ) < l1Coeff ) {
			value += z * z / 2;
			gradient.SetAt( i, z );
		} else {
			value += l1Coeff * ( abs( z ) - l1Coeff / 2 );
			gradient.SetAt( i, l1Coeff * z / abs( z ) );
		}
	}
}

//------------------------------------------------------------------------------------------------------------

struct CThreadFunctionWithHessianImpl::IArgs {
	explicit IArgs( CThreadFunctionWithHessianImpl& f ) :
		F( f ),
		ThreadsCount( F.ThreadPool->Size() )
	{}
	virtual ~IArgs() {}

	virtual int Size() const { return F.Matrix.Height; }
	virtual void Run( int threadIndex, int index, int count ) = 0;
	virtual void SubRun( int index, CFloatVectorDesc,
		double* value, CFloatVector& grad, float answer, float weight ) = 0;
	virtual void Reduction() = 0;

	CThreadFunctionWithHessianImpl& F;
	const int ThreadsCount;
};

//------------------------------------------------------------------------------------------------------------

struct CThreadTask final {
	CThreadTask( CThreadFunctionWithHessianImpl::IArgs& args ) : args( args ) {}

	void CallRun( int threadIndex );

private:
	CThreadFunctionWithHessianImpl::IArgs& args;
};

void CThreadTask::CallRun( int threadIndex )
{
	int index = 0;
	int count = 0;
	if( GetTaskIndexAndCount( args.ThreadsCount, threadIndex, args.Size(), index, count ) ) {
		args.Run( threadIndex, index, count );
	}
}

//------------------------------------------------------------------------------------------------------------

struct CThreadFunctionWithHessianImpl::CProductArgs final : public CThreadFunctionWithHessianImpl::IArgs {
	CProductArgs( CThreadFunctionWithHessianImpl& f, const CFloatVector& argument );

	void Run( int threadIndex, int index, int count ) override;
	void SubRun( int /*index*/, CFloatVectorDesc, double* /*value*/,
		CFloatVector& /*grad*/, float /*answer*/, float /*weight*/ ) override {}
	void Reduction() override;

	const CFloatVector& Argument;
	CArray<CFloatVector> ResultReduction{};
	CFloatVector Result{};
};

CThreadFunctionWithHessianImpl::CProductArgs::CProductArgs(
		CThreadFunctionWithHessianImpl& f, const CFloatVector& argument ) :
	IArgs( f ),
	Argument( argument ),
	Result( argument / F.ErrorWeight )
{
	Result.SetAt( Result.Size() - 1, 0 );
	ResultReduction.Add( CFloatVector( Result.Size() ), ThreadsCount );
}

void CThreadFunctionWithHessianImpl::CProductArgs::Run( int threadIndex, int index, int count )
{
	CFloatVector& result = ResultReduction[threadIndex];
	result.Nullify();

	for( int i = 0; i < count; ++i, ++index ) {
		if( F.Hessian[index] != 0 ) {
			CFloatVectorDesc desc;
			F.Matrix.GetRow( index, desc );

			const double temp = LinearFunction( Argument, desc ) * F.Hessian[index];
			result.MultiplyAndAddExt( desc, temp );
		}
	}
}

void CThreadFunctionWithHessianImpl::CProductArgs::Reduction()
{
	for( int i = 0; i < ResultReduction.Size(); ++i ) {
		Result += ResultReduction[i];
	}
}

//------------------------------------------------------------------------------------------------------------

struct CThreadFunctionWithHessianImpl::CSetArgs : public CThreadFunctionWithHessianImpl::IArgs {
	CSetArgs( CThreadFunctionWithHessianImpl&, const CFloatVector& argument );

	void Run( int threadIndex, int index, int count ) override;
	void Reduction() override;

	CFloatVector Argument{};
	CArray<CFloatVector> GradientReduction{};
	CArray<double> ValueReduction{};
};

CThreadFunctionWithHessianImpl::CSetArgs::CSetArgs( CThreadFunctionWithHessianImpl& f, const CFloatVector& argument ) :
	IArgs( f ),
	Argument( argument )
{
	NeoAssert( Argument.Size() == F.NumberOfDimensions() );

	F.Grad = Argument;
	F.Grad.SetAt( F.Grad.Size() - 1, 0 ); // don't take the regularization bias into account

	if( F.L1Coeff > 0 ) {
		calcL1Regularization( F.Grad, F.L1Coeff, F.Val, F.Grad );
	} else {
		F.Val = DotProduct( F.Grad, F.Grad ) / 2;
	}
	F.Val = F.Val / F.ErrorWeight;
	F.Grad = F.Grad / F.ErrorWeight;

	GradientReduction.Add( CFloatVector( F.Grad.Size() ), ThreadsCount );
	ValueReduction.Add( 0., ThreadsCount );
	f.Hessian.SetSize( Size() );
}

void CThreadFunctionWithHessianImpl::CSetArgs::Run( int threadIndex, int index, int count )
{
	double* value = ValueReduction.GetPtr() + threadIndex;
	GradientReduction[threadIndex].Nullify();
	CFloatVector& grad = GradientReduction[threadIndex];

	const float* AnswersPtr = F.Answers.GetPtr();
	const float* WeightsPtr = F.Weights.GetPtr();

	for( int i = 0; i < count; ++i, ++index ) {
		CFloatVectorDesc desc;
		F.Matrix.GetRow( index, desc );

		const float answer = AnswersPtr[index];
		const float weight = WeightsPtr[index];

		SubRun( index, desc, value, grad, answer, weight );
	}
}

void CThreadFunctionWithHessianImpl::CSetArgs::Reduction()
{
	for( int t = 0; t < ThreadsCount; ++t ) {
		F.Val += ValueReduction[t];
		F.Grad += GradientReduction[t];
	}
}

//------------------------------------------------------------------------------------------------------------

CThreadFunctionWithHessianImpl::CThreadFunctionWithHessianImpl(
		CFloatMatrixDesc matrix, int vectorCount, double errorWeight, float l1Coeff, int threadCount ) :
	ThreadPool( CreateThreadPool( threadCount ) ),
	ErrorWeight( static_cast<float>( errorWeight ) ),
	L1Coeff( l1Coeff ),
	Matrix( matrix ),
	Answers( vectorCount ),
	Weights( vectorCount )
{}

CThreadFunctionWithHessianImpl::CThreadFunctionWithHessianImpl(
		const IProblem& data, double errorWeight, float l1Coeff, int threadCount ) :
	CThreadFunctionWithHessianImpl( data.GetMatrix(), data.GetVectorCount(),
		errorWeight, l1Coeff, threadCount )
{
	float* answersPtr = Answers.CopyOnWrite();
	float* weightsPtr = Weights.CopyOnWrite();
	for( int i = 0; i < Matrix.Height; ++i ) {
		answersPtr[i] = static_cast<float>( data.GetBinaryClass( i ) );
		weightsPtr[i] = static_cast<float>( data.GetVectorWeight( i ) );
	}
}

CThreadFunctionWithHessianImpl::CThreadFunctionWithHessianImpl(
		const IRegressionProblem& data, double errorWeight, float l1Coeff, int threadCount ) :
	CThreadFunctionWithHessianImpl(	data.GetMatrix(), data.GetVectorCount(),
		errorWeight, l1Coeff, threadCount )
{
	float* answersPtr = Answers.CopyOnWrite();
	float* weightsPtr = Weights.CopyOnWrite();
	for( int i = 0; i < Matrix.Height; ++i ) {
		answersPtr[i] = static_cast<float>( data.GetValue( i ) );
		weightsPtr[i] = static_cast<float>( data.GetVectorWeight( i ) );
	}
}

CThreadFunctionWithHessianImpl::~CThreadFunctionWithHessianImpl()
{
	delete ThreadPool;
}

// Multiplies hessian by vector
CFloatVector CThreadFunctionWithHessianImpl::HessianProduct( const CFloatVector& argument )
{
	CThreadFunctionWithHessianImpl::CProductArgs args( *this, argument );
	CThreadTask task( args );

	NEOML_NUM_THREADS( *ThreadPool, &task, []( int threadIndex, void* ptr ) {
		( ( CThreadTask* )ptr )->CallRun( threadIndex );
	} );

	args.Reduction();
	return args.Result;
}

void CThreadFunctionWithHessianImpl::SetArguments( CSetArgs& args )
{
	CThreadTask task( args );

	NEOML_NUM_THREADS( *ThreadPool, &task, []( int threadIndex, void* ptr ) {
		( ( CThreadTask* )ptr )->CallRun( threadIndex );
	} );

	args.Reduction();
}

//------------------------------------------------------------------------------------------------------------

struct CSquaredHinge::CArgs : public CThreadFunctionWithHessianImpl::CSetArgs {
	CArgs( CThreadFunctionWithHessianImpl& f, const CFloatVector& argument ) :
		CSetArgs( f, argument )
	{}

	void SubRun( int index, CFloatVectorDesc, double* value,
		CFloatVector& grad, float answer, float weight ) override;

	CSquaredHinge& Function() { return dynamic_cast<CSquaredHinge&>( F ); }
};

void CSquaredHinge::CArgs::SubRun( int index, CFloatVectorDesc desc,
	double* value, CFloatVector& grad, float answer, float weight )
{
	CSquaredHinge& f = Function();

	const double x = answer * LinearFunction( Argument, desc );
	const double d = 1 - x;

	if( x < 1 ) {
		*value += weight * d * d;
		grad.MultiplyAndAddExt( desc, -weight * answer * d * 2 );

		f.Hessian[index] = weight * 2;
	} else {
		f.Hessian[index] = 0;
	}
}

CSquaredHinge::CSquaredHinge(
		const IProblem& data, double errorWeight, float l1Coeff, int threadCount ) :
	CThreadFunctionWithHessianImpl( data, errorWeight, l1Coeff, threadCount )
{}

void CSquaredHinge::SetArgument( const CFloatVector& argument )
{
	CArgs args( *this, argument );
	SetArguments( args );
}

//-----------------------------------------------------------------------------------------------------------------------

struct CL2Regression::CArgs : public CThreadFunctionWithHessianImpl::CSetArgs {
	CArgs( CL2Regression& f, const CFloatVector& argument, double p ) :
		CSetArgs( f, argument ),
		P( p )
	{}

	void SubRun( int index, CFloatVectorDesc, double* value,
		CFloatVector& grad, float answer, float weight ) override;

	CL2Regression& Function() { return dynamic_cast<CL2Regression&>( F ); }

	const double P;
};

void CL2Regression::CArgs::SubRun( int index, CFloatVectorDesc desc,
	double* value, CFloatVector& grad, float answer, float weight )
{
	CL2Regression& f = Function();
	const double d = LinearFunction( Argument, desc ) - answer;

	if( d < -P ) {
		*value += weight * ( d + P ) * ( d + P );
		f.Hessian[index] = weight * 2;
		grad.MultiplyAndAddExt( desc, weight * ( d + P ) * 2 );
	} else {
		*value += weight * ( d - P ) * ( d - P );

		if( d > P ) {
			f.Hessian[index] = weight * 2;
			grad.MultiplyAndAddExt( desc, weight * ( d - P ) * 2 );
		} else {
			f.Hessian[index] = 0.f;
		}
	}
}

CL2Regression::CL2Regression(
		const IRegressionProblem& data, double errorWeight, double p, float l1Coeff, int threadCount ) :
	CThreadFunctionWithHessianImpl( data, errorWeight, l1Coeff, threadCount ),
	P( static_cast<float>( p ) )
{}

void CL2Regression::SetArgument( const CFloatVector& argument )
{
	CArgs args( *this, argument, P );
	SetArguments( args );
}

//-----------------------------------------------------------------------------------------------------------------------

struct CLogRegression::CArgs : public CThreadFunctionWithHessianImpl::CSetArgs {
	CArgs( CLogRegression&, const CFloatVector& argument );

	void SubRun( int index, CFloatVectorDesc, double* value,
		CFloatVector& grad, float answer, float weight ) override;

	CLogRegression& Function() { return dynamic_cast<CLogRegression&>( F ); }

	double rValue = 0;
	const float LogNormalizer = 1.f / logf( 2.f );
};

CLogRegression::CArgs::CArgs( CLogRegression& f, const CFloatVector& argument ) :
	CSetArgs( f, argument )
{
	f.Val = 0; // should repeat all of this, because Val should == 0, and rValue used instead
	rValue = 0;

	f.Grad = Argument;  // calculate Grad the same again
	f.Grad.SetAt( f.Grad.Size() - 1, 0 ); // don't take the regularization bias into account

	if( f.L1Coeff > 0 ) {
		calcL1Regularization( f.Grad, f.L1Coeff, rValue, f.Grad );  // !!!
	} else {
		rValue = DotProduct( f.Grad, f.Grad ) / 2; // !!!
	}
	rValue = rValue / f.ErrorWeight; // !!!
	f.Grad = f.Grad / f.ErrorWeight;
}

void CLogRegression::CArgs::SubRun( int index, CFloatVectorDesc desc,
	double* value, CFloatVector& grad, float answer, float weight )
{
	CLogRegression& f = Function();

	const double dot = LinearFunction( Argument, desc );
	const double expCoeff = exp( -answer * dot );

	*value += weight * log1p( expCoeff );

	grad.MultiplyAndAddExt( desc, -weight * LogNormalizer * answer * expCoeff / ( 1.f + expCoeff ) );
	f.Hessian[index] = weight * LogNormalizer * expCoeff / ( 1.f + expCoeff ) / ( 1.f + expCoeff );
}

CLogRegression::CLogRegression(
		const IProblem& data, double errorWeight, float l1Coeff, int threadCount ) :
	CThreadFunctionWithHessianImpl( data, errorWeight, l1Coeff, threadCount )
{}

void CLogRegression::SetArgument( const CFloatVector& argument )
{
	CArgs args( *this, argument );
	SetArguments( args );

	Val *= args.LogNormalizer;
	Val += args.rValue;
}

//-----------------------------------------------------------------------------------------------------------------------

struct CSmoothedHinge::CArgs : public CThreadFunctionWithHessianImpl::CSetArgs {
	CArgs( CSmoothedHinge& f, const CFloatVector& argument ) :
		CSetArgs( f, argument )
	{}

	void SubRun( int index, CFloatVectorDesc, double* value,
		CFloatVector& grad, float answer, float weight ) override;

	CSmoothedHinge& Function() { return dynamic_cast<CSmoothedHinge&>( F ); }
};

void CSmoothedHinge::CArgs::SubRun( int index, CFloatVectorDesc desc,
	double* value, CFloatVector& grad, float answer, float weight )
{
	CSmoothedHinge& f = Function();

	const double d = answer * LinearFunction( Argument, desc ) - 1;

	if( d < 0 ) {
		const float sqrtValue = static_cast<float>( sqrt( d * d + 1 ) );
		*value += weight * ( sqrtValue - 1 );

		grad.MultiplyAndAddExt( desc, weight * answer * d / sqrtValue );
		f.Hessian[index] = weight / ( ( d * d + 1 ) * sqrtValue );
	} else {
		f.Hessian[index] = 0;
	}
}

CSmoothedHinge::CSmoothedHinge(
		const IProblem& data, double errorWeight, float l1Coeff, int threadCount ) :
	CThreadFunctionWithHessianImpl( data, errorWeight, l1Coeff, threadCount )
{}

void CSmoothedHinge::SetArgument( const CFloatVector& argument )
{
	CArgs args( *this, argument );
	SetArguments( args );
}

} // namespace NeoML
