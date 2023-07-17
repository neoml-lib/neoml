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
	value = 0.;
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

struct CThreadFunctionWithHessianImpl::IThreadTask {
	virtual ~IThreadTask() {}
	void ParallelRun();

protected:
	static constexpr int MultiThreadMinTasksCount = 2;

	IThreadTask( CThreadFunctionWithHessianImpl& f, const CFloatVector& argument ) :
		F( f ),
		Argument( argument )
	{}
	void RunSplittedByThreads( int threadIndex );

	virtual int ParallelizeSize() const { return F.Matrix.Height; }
	virtual int ThreadsCount() const { return F.ThreadPool->Size(); }

	virtual void Run( int threadIndex, int index, int count ) = 0;
	virtual void Reduction() = 0;

	CThreadFunctionWithHessianImpl& F;
	const CFloatVector& Argument;
};

void CThreadFunctionWithHessianImpl::IThreadTask::ParallelRun()
{
	if( ParallelizeSize() < MultiThreadMinTasksCount ) {
		Run( /*threadIndex*/0, /*index*/0, ParallelizeSize() );
	} else {
		NEOML_NUM_THREADS( *F.ThreadPool, this, []( int threadIndex, void* ptr ) {
			 ( ( IThreadTask* )ptr )->RunSplittedByThreads( threadIndex );
		} );
	}
	Reduction();
}

void CThreadFunctionWithHessianImpl::IThreadTask::RunSplittedByThreads( int threadIndex )
{
	int index = 0;
	int count = 0;
	// 1 dimensional split
	if( GetTaskIndexAndCount( ThreadsCount(), threadIndex, ParallelizeSize(), index, count ) ) {
		Run( threadIndex, index, count );
	}
}

//------------------------------------------------------------------------------------------------------------

struct CThreadFunctionWithHessianImpl::CProductThreadTask final : public CThreadFunctionWithHessianImpl::IThreadTask {
	CProductThreadTask( CThreadFunctionWithHessianImpl& f, const CFloatVector& argument );

	void Reduction() override;

	CFloatVector Result{};
protected:
	void Run( int threadIndex, int index, int count ) override;

	CArray<CFloatVector> ThreadsResult{};
};

CThreadFunctionWithHessianImpl::CProductThreadTask::CProductThreadTask(
		CThreadFunctionWithHessianImpl& f, const CFloatVector& argument ) :
	IThreadTask( f, argument ),
	Result( argument / F.ErrorWeight )
{
	Result.SetAt( Result.Size() - 1, 0 );
	ThreadsResult.Add( CFloatVector( Result.Size() ), ThreadsCount() );
}

void CThreadFunctionWithHessianImpl::CProductThreadTask::Run( int threadIndex, int index, int count )
{
	CFloatVector& result = ThreadsResult[threadIndex];
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

void CThreadFunctionWithHessianImpl::CProductThreadTask::Reduction()
{
	for( int t = 0; t < ThreadsCount(); ++t ) {
		Result += ThreadsResult[t];
	}
}

//------------------------------------------------------------------------------------------------------------

struct CThreadFunctionWithHessianImpl::CSetArgThreadTask : public CThreadFunctionWithHessianImpl::IThreadTask {
	CSetArgThreadTask( CThreadFunctionWithHessianImpl&, const CFloatVector& argument );

	void Run( int threadIndex, int index, int count ) override final;
	void Reduction() override;

protected:
	virtual void SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
		double& hessian, double& value, CFloatVector& gradient, float answer, float weight ) = 0;

	CArray<CFloatVector> ThreadsGradient{};
	CArray<double> ThreadsValue{};
};

CThreadFunctionWithHessianImpl::CSetArgThreadTask::CSetArgThreadTask(
		CThreadFunctionWithHessianImpl& f, const CFloatVector& argument ) :
	IThreadTask( f, argument )
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

	f.Hessian.SetSize( ParallelizeSize() );

	ThreadsGradient.Add( CFloatVector( F.Grad.Size() ), ThreadsCount() );
	ThreadsValue.Add( 0., ThreadsCount() );
}

void CThreadFunctionWithHessianImpl::CSetArgThreadTask::Run( int threadIndex, int index, int count )
{
	double* value = ThreadsValue.GetPtr() + threadIndex;
	CFloatVector& gradient = ThreadsGradient[threadIndex];
	gradient.Nullify();

	for( int i = 0; i < count; ++i, ++index ) {
		CFloatVectorDesc desc;
		F.Matrix.GetRow( index, desc );

		const float answer = F.Answers[index];
		const float weight = F.Weights[index];
		double& hessian = F.Hessian[index];

		SetArgument( Argument, desc, hessian, *value, gradient, answer, weight );
	}
}

void CThreadFunctionWithHessianImpl::CSetArgThreadTask::Reduction()
{
	for( int t = 0; t < ThreadsCount(); ++t ) {
		F.Val += ThreadsValue[t];
		F.Grad += ThreadsGradient[t];
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
	CThreadFunctionWithHessianImpl::CProductThreadTask task( *this, argument );
	task.ParallelRun();
	return task.Result;
}

void CThreadFunctionWithHessianImpl::SetArguments( CSetArgThreadTask& task )
{
	task.ParallelRun();
}

//------------------------------------------------------------------------------------------------------------

struct CSquaredHinge::CSetArgThreadTask : public CThreadFunctionWithHessianImpl::CSetArgThreadTask {
	CSetArgThreadTask( CThreadFunctionWithHessianImpl& f, const CFloatVector& argument ) :
		CThreadFunctionWithHessianImpl::CSetArgThreadTask( f, argument )
	{}
protected:
	void SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
		double& hessian, double& value, CFloatVector& gradient, float answer, float weight ) override;
};

void CSquaredHinge::CSetArgThreadTask::SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
	double& hessian, double& value, CFloatVector& gradient, float answer, float weight )
{
	const double x = answer * LinearFunction( argument, desc );
	const double d = 1 - x;

	if( x < 1 ) {
		value += weight * d * d;
		gradient.MultiplyAndAddExt( desc, -weight * answer * d * 2 );
		hessian = weight * 2;
	} else {
		hessian = 0;
	}
}

CSquaredHinge::CSquaredHinge(
		const IProblem& data, double errorWeight, float l1Coeff, int threadCount ) :
	CThreadFunctionWithHessianImpl( data, errorWeight, l1Coeff, threadCount )
{}

void CSquaredHinge::SetArgument( const CFloatVector& argument )
{
	CSetArgThreadTask task( *this, argument );
	SetArguments( task );
}

//-----------------------------------------------------------------------------------------------------------------------

struct CL2Regression::CSetArgThreadTask : public CThreadFunctionWithHessianImpl::CSetArgThreadTask {
	CSetArgThreadTask( CL2Regression& f, const CFloatVector& argument, double p ) :
		CThreadFunctionWithHessianImpl::CSetArgThreadTask( f, argument ),
		P( p )
	{}
protected:
	void SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
		double& hessian, double& value, CFloatVector& gradient, float answer, float weight ) override;

	const double P;
};

void CL2Regression::CSetArgThreadTask::SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
	double& hessian, double& value, CFloatVector& gradient, float answer, float weight )
{
	const double d = LinearFunction( argument, desc ) - answer;

	if( d < -P ) {
		value += weight * ( d + P ) * ( d + P );
		hessian = weight * 2;
		gradient.MultiplyAndAddExt( desc, weight * ( d + P ) * 2 );
	} else {
		value += weight * ( d - P ) * ( d - P );
		if( d > P ) {
			hessian = weight * 2;
			gradient.MultiplyAndAddExt( desc, weight * ( d - P ) * 2 );
		} else {
			hessian = 0;
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
	CSetArgThreadTask task( *this, argument, P );
	SetArguments( task );
}

//-----------------------------------------------------------------------------------------------------------------------

struct CLogRegression::CSetArgThreadTask : public CThreadFunctionWithHessianImpl::CSetArgThreadTask {
	CSetArgThreadTask( CLogRegression&, const CFloatVector& argument );

	double rValue = 0.;
	const float LogNormalizer = 1.f / logf( 2.f );
protected:
	void SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
		double& hessian, double& value, CFloatVector& gradient, float answer, float weight ) override;
};

CLogRegression::CSetArgThreadTask::CSetArgThreadTask( CLogRegression& f, const CFloatVector& argument ) :
	CThreadFunctionWithHessianImpl::CSetArgThreadTask( f, argument )
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

void CLogRegression::CSetArgThreadTask::SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
	double& hessian, double& value, CFloatVector& gradient, float answer, float weight )
{
	const double dot = LinearFunction( argument, desc );
	const double expCoeff = exp( -answer * dot );

	value += weight * log1p( expCoeff );

	gradient.MultiplyAndAddExt( desc, -weight * LogNormalizer * answer * expCoeff / ( 1.f + expCoeff ) );
	hessian = weight * LogNormalizer * expCoeff / ( 1.f + expCoeff ) / ( 1.f + expCoeff );
}

CLogRegression::CLogRegression(
		const IProblem& data, double errorWeight, float l1Coeff, int threadCount ) :
	CThreadFunctionWithHessianImpl( data, errorWeight, l1Coeff, threadCount )
{}

void CLogRegression::SetArgument( const CFloatVector& argument )
{
	CSetArgThreadTask task( *this, argument );
	SetArguments( task );

	Val *= task.LogNormalizer;
	Val += task.rValue;
}

//-----------------------------------------------------------------------------------------------------------------------

struct CSmoothedHinge::CSetArgThreadTask : public CThreadFunctionWithHessianImpl::CSetArgThreadTask {
	CSetArgThreadTask( CSmoothedHinge& f, const CFloatVector& argument ) :
		CThreadFunctionWithHessianImpl::CSetArgThreadTask( f, argument )
	{}
protected:
	void SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
		double& hessian, double& value, CFloatVector& gradient, float answer, float weight ) override;
};

void CSmoothedHinge::CSetArgThreadTask::SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
	double& hessian, double& value, CFloatVector& gradient, float answer, float weight )
{
	const double d = answer * LinearFunction( argument, desc ) - 1;

	if( d < 0 ) {
		const float sqrtValue = static_cast<float>( sqrt( d * d + 1 ) );
		value += weight * ( sqrtValue - 1 );
		gradient.MultiplyAndAddExt( desc, weight * answer * d / sqrtValue );
		hessian = weight / ( ( d * d + 1 ) * sqrtValue );
	} else {
		hessian = 0;
	}
}

CSmoothedHinge::CSmoothedHinge(
		const IProblem& data, double errorWeight, float l1Coeff, int threadCount ) :
	CThreadFunctionWithHessianImpl( data, errorWeight, l1Coeff, threadCount )
{}

void CSmoothedHinge::SetArgument( const CFloatVector& argument )
{
	CSetArgThreadTask task( *this, argument );
	SetArguments( task );
}

} // namespace NeoML
