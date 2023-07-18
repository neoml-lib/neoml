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
			value += z * z / 2.f;
			gradient.SetAt( i, z );
		} else {
			value += l1Coeff * ( abs( z ) - l1Coeff / 2.f );
			gradient.SetAt( i, l1Coeff * z / abs( z ) );
		}
	}
}

//------------------------------------------------------------------------------------------------------------

struct CFunctionWithHessianState {
	// Children creation
	static CFunctionWithHessianState* Create( CFloatMatrixDesc, int vectorCount,
		double errorWeight, double p, float l1Coeff, int threadCount, THessianFType type );

	// Ctor & dtor
	CFunctionWithHessianState( CFloatMatrixDesc, int vectorCount,
		double errorWeight, float l1Coeff, int threadCount, THessianFType type );
	virtual ~CFunctionWithHessianState();
	// Forbid coping
	CFunctionWithHessianState( const CFunctionWithHessianState& ) = delete;
	CFunctionWithHessianState& operator=( const CFunctionWithHessianState& ) = delete;

	// Basic function operations
	int NumberOfDimensions() const { return Matrix.Width + 1; }
	// Called in Step 1. Run of IThreadTask
	void HessianProduct( CFloatVector& result, const CFloatVector& argument, int index );

	// Call in Step 0. Constructor of IThreadTask
	virtual void PrepareSetArgument( const CFloatVector& argument, double& value );
	// Call in Step 1. Run of IThreadTask
	virtual void SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
		double& hessian, double& value, CFloatVector& gradient, float answer, float weight ) = 0;
	// Call in Step 2. Reduction of IThreadTask
	virtual void PostReductionSetArgument() {} // special for LogRegression

	// Executors
	IThreadPool* const ThreadPool;
	// Internal state
	const THessianFType Type;
	const float ErrorWeight;
	const float L1Coeff;
	double Value = 0.;
	CFloatVector Gradient{};
	CArray<double> Hessian{};
	// Problem description
	const CFloatMatrixDesc Matrix;
	CFloatVector Answers{};
	CFloatVector Weights{};
};

CFunctionWithHessianState::CFunctionWithHessianState( CFloatMatrixDesc matrix, int vectorCount,
		double errorWeight, float l1Coeff, int threadCount, THessianFType type ) :
	ThreadPool( CreateThreadPool( threadCount ) ),
	Type( type ),
	ErrorWeight( static_cast<float>( errorWeight ) ),
	L1Coeff( l1Coeff ),
	Matrix( matrix ),
	Answers( vectorCount ),
	Weights( vectorCount )
{
	NeoAssert( ThreadPool != nullptr );
}

CFunctionWithHessianState::~CFunctionWithHessianState()
{
	delete ThreadPool;
}

void CFunctionWithHessianState::HessianProduct( CFloatVector& result, const CFloatVector& argument, int index )
{
	if( Hessian[index] != 0. ) {
		CFloatVectorDesc desc;
		Matrix.GetRow( index, desc );
		const double temp = LinearFunction( argument, desc ) * Hessian[index];
		result.MultiplyAndAddExt( desc, temp );
	}
}

void CFunctionWithHessianState::PrepareSetArgument( const CFloatVector& argument, double& value )
{
	value = 0.;
	Gradient = argument;
	Gradient.SetAt( Gradient.Size() - 1, 0.f ); // don't take the regularization bias into account

	if( L1Coeff > 0.f ) {
		calcL1Regularization( Gradient, L1Coeff, Value, Gradient );
	} else {
		value = DotProduct( Gradient, Gradient ) / 2.;
	}
	value = value / ErrorWeight;
	Gradient = Gradient / ErrorWeight;

	Hessian.SetSize( Matrix.Height );
}

//------------------------------------------------------------------------------------------------------------

namespace {

// Task which processes in multiple threads
struct IThreadTask {
	virtual ~IThreadTask() {}
	// Run in a single thread or in parallel, corresponding to `ParallelizeSize()`
	// 1. Run       -- Split into sub-tasks by threads and start parallel execution.
	// 2. Reduction -- Combine results of each thread in 1 answer.
	void ParallelRun();

protected:
	static constexpr int MultiThreadMinTasksCount = 2;
	// Create a task
	IThreadTask( CFunctionWithHessianState& fs, const CFloatVector& argument ) :
		FS( fs ),
		Argument( argument )
	{}
	// Get way of split the task into sub-tasks
	void RunSplittedByThreads( int threadIndex );

	// The size of parallelization, max number of elements to perform
	virtual int ParallelizeSize() const { return FS.Matrix.Height; }
	// The number of separate executors
	virtual int ThreadsCount() const { return FS.ThreadPool->Size(); }

	// Step 1. Run the process in a separate thread
	virtual void Run( int threadIndex, int index, int count ) = 0;
	// Step 2. Combine the answer
	virtual void Reduction() = 0;

	CFunctionWithHessianState& FS;
	const CFloatVector& Argument;
};

void IThreadTask::ParallelRun()
{
	// Step 1. Run in a separate thread
	if( ParallelizeSize() < MultiThreadMinTasksCount ) {
		Run( /*threadIndex*/0, /*index*/0, ParallelizeSize() );
	} else {
		NEOML_NUM_THREADS( *FS.ThreadPool, this, []( int threadIndex, void* ptr ) {
			 ( ( IThreadTask* )ptr )->RunSplittedByThreads( threadIndex );
		} );
	}
	// Step 2. Combine the answer
	Reduction();
}

void IThreadTask::RunSplittedByThreads( int threadIndex )
{
	int index = 0;
	int count = 0;
	// 1 dimensional split
	if( GetTaskIndexAndCount( ThreadsCount(), threadIndex, ParallelizeSize(), index, count ) ) {
		Run( threadIndex, index, count );
	}
}

//------------------------------------------------------------------------------------------------------------

struct CHessianProductTask : public IThreadTask {
	CHessianProductTask( CFunctionWithHessianState&, const CFloatVector& argument );

	CFloatVector Result{};
protected:
	void Run( int threadIndex, int index, int count ) override;
	void Reduction() override;

	CArray<CFloatVector> ThreadsResult{};
};

CHessianProductTask::CHessianProductTask( CFunctionWithHessianState& fs, const CFloatVector& argument ) :
	IThreadTask( fs, argument ),
	Result( argument / FS.ErrorWeight )
{
	Result.SetAt( Result.Size() - 1, 0 );
	ThreadsResult.Add( CFloatVector( Result.Size() ), ThreadsCount() );
}

void CHessianProductTask::Run( int threadIndex, int index, int count )
{
	CFloatVector& result = ThreadsResult[threadIndex];
	result.Nullify();
	for( int i = 0; i < count; ++i, ++index ) {
		FS.HessianProduct( result, Argument, index );
	}
}

void CHessianProductTask::Reduction()
{
	for( int t = 0; t < ThreadsCount(); ++t ) {
		Result += ThreadsResult[t];
	}
}

//------------------------------------------------------------------------------------------------------------

struct CSetArgumentTask : public IThreadTask {
	CSetArgumentTask( CFunctionWithHessianState&, const CFloatVector& argument );
protected:
	void Run( int threadIndex, int index, int count ) override;
	void Reduction() override;

	CArray<CFloatVector> ThreadsGradient{};
	CArray<double> ThreadsValue{};
};

CSetArgumentTask::CSetArgumentTask( CFunctionWithHessianState& fs, const CFloatVector& argument ) :
	IThreadTask( fs, argument )
{
	// Important preparations
	FS.PrepareSetArgument( Argument, FS.Value );

	ThreadsGradient.Add( CFloatVector( FS.Gradient.Size() ), ThreadsCount() );
	ThreadsValue.Add( 0., ThreadsCount() );
}

void CSetArgumentTask::Run( int threadIndex, int index, int count )
{
	double* value = ThreadsValue.GetPtr() + threadIndex;
	CFloatVector& gradient = ThreadsGradient[threadIndex];
	gradient.Nullify();

	for( int i = 0; i < count; ++i, ++index ) {
		CFloatVectorDesc desc;
		FS.Matrix.GetRow( index, desc );

		const float answer = FS.Answers[index];
		const float weight = FS.Weights[index];
		double& hessian = FS.Hessian[index];
		// Main function call
		FS.SetArgument( Argument, desc, hessian, *value, gradient, answer, weight );
	}
}

void CSetArgumentTask::Reduction()
{
	for( int t = 0; t < ThreadsCount(); ++t ) {
		FS.Value += ThreadsValue[t];
		FS.Gradient += ThreadsGradient[t];
	}
	// Important for answer
	FS.PostReductionSetArgument();
}

} // namespace

//------------------------------------------------------------------------------------------------------------

IMultiThreadFunctionWithHessianImpl::IMultiThreadFunctionWithHessianImpl(
		const IProblem& problem, double errorWeight, float l1Coeff, int threadCount, THessianFType type ) :
	FS( CFunctionWithHessianState::Create( problem.GetMatrix(), problem.GetVectorCount(), errorWeight, 0., l1Coeff, threadCount, type ) )
{
	NeoAssert( FS != nullptr );
	float* answersPtr = FS->Answers.CopyOnWrite();
	float* weightsPtr = FS->Weights.CopyOnWrite();
	for( int i = 0; i < FS->Matrix.Height; ++i ) {
		answersPtr[i] = static_cast<float>( problem.GetBinaryClass( i ) );
		weightsPtr[i] = static_cast<float>( problem.GetVectorWeight( i ) );
	}
}

IMultiThreadFunctionWithHessianImpl::IMultiThreadFunctionWithHessianImpl(
		const IRegressionProblem& problem, double errorWeight, double p, float l1Coeff, int threadCount, THessianFType type ) :
	FS( CFunctionWithHessianState::Create( problem.GetMatrix(), problem.GetVectorCount(), errorWeight, p, l1Coeff, threadCount, type ) )
{
	NeoAssert( FS != nullptr );
	float* answersPtr = FS->Answers.CopyOnWrite();
	float* weightsPtr = FS->Weights.CopyOnWrite();
	for( int i = 0; i < FS->Matrix.Height; ++i ) {
		answersPtr[i] = static_cast<float>( problem.GetValue( i ) );
		weightsPtr[i] = static_cast<float>( problem.GetVectorWeight( i ) );
	}
}

IMultiThreadFunctionWithHessianImpl::~IMultiThreadFunctionWithHessianImpl()
{
	delete FS;
}

CFloatVector IMultiThreadFunctionWithHessianImpl::HessianProduct( const CFloatVector& argument )
{
	CHessianProductTask task( *FS, argument );
	task.ParallelRun();
	return task.Result;
}

int IMultiThreadFunctionWithHessianImpl::NumberOfDimensions() const
{
	return FS->NumberOfDimensions();
}

double IMultiThreadFunctionWithHessianImpl::Value() const
{
	return FS->Value;
}

CFloatVector IMultiThreadFunctionWithHessianImpl::Gradient() const
{
	return FS->Gradient;
}

void IMultiThreadFunctionWithHessianImpl::SetArgument( const CFloatVector& argument )
{
	NeoAssert( argument.Size() == NumberOfDimensions() );
	CSetArgumentTask task( *FS, argument );
	task.ParallelRun();
}

//------------------------------------------------------------------------------------------------------------

namespace {

struct CSquaredHingeState : public CFunctionWithHessianState {
	using CFunctionWithHessianState::CFunctionWithHessianState; // inherit constructor of the base class
	void SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
		double& hessian, double& value, CFloatVector& gradient, float answer, float weight ) override;
};

void CSquaredHingeState::SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
	double& hessian, double& value, CFloatVector& gradient, float answer, float weight )
{
	const double x = answer * LinearFunction( argument, desc );
	const double d = 1. - x;

	if( x < 1. ) {
		value += weight * d * d;
		gradient.MultiplyAndAddExt( desc, -weight * answer * d * 2. );
		hessian = weight * 2.;
	} else {
		hessian = 0.;
	}
}

//-----------------------------------------------------------------------------------------------------------------------

struct CL2RegressionState : public CFunctionWithHessianState {
	CL2RegressionState( CFloatMatrixDesc matrix, int vectorCount,
			double errorWeight, double p, float l1Coeff, int threadCount, THessianFType type ) :
		CFunctionWithHessianState( matrix, vectorCount, errorWeight, l1Coeff, threadCount, type ),
		P( p )
	{}
protected:
	void SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
		double& hessian, double& value, CFloatVector& gradient, float answer, float weight ) override;

	const double P;
};

void CL2RegressionState::SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
	double& hessian, double& value, CFloatVector& gradient, float answer, float weight )
{
	const double d = LinearFunction( argument, desc ) - answer;

	if( d < -P ) {
		value += weight * ( d + P ) * ( d + P );
		hessian = weight * 2.;
		gradient.MultiplyAndAddExt( desc, weight * ( d + P ) * 2. );
	} else {
		value += weight * ( d - P ) * ( d - P );
		if( d > P ) {
			hessian = weight * 2.;
			gradient.MultiplyAndAddExt( desc, weight * ( d - P ) * 2. );
		} else {
			hessian = 0.;
		}
	}
}

//-----------------------------------------------------------------------------------------------------------------------

struct CLogRegressionState : public CFunctionWithHessianState {
	using CFunctionWithHessianState::CFunctionWithHessianState; // inherit constructor of the base class

	void PrepareSetArgument( const CFloatVector& argument, double& value ) override;
	void SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
		double& hessian, double& value, CFloatVector& gradient, float answer, float weight ) override;
	void PostReductionSetArgument() override;

	double RValue = 0.;
	const float LogNormalizer = 1.f / logf( 2.f );
};

void CLogRegressionState::PrepareSetArgument( const CFloatVector& argument, double& value )
{
	value = 0.;
	// Should repeat all of this, because Value should == 0, and RValue used instead
	CFunctionWithHessianState::PrepareSetArgument( argument, RValue );
}

void CLogRegressionState::SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
	double& hessian, double& value, CFloatVector& gradient, float answer, float weight )
{
	const double dot = LinearFunction( argument, desc );
	const double expCoeff = exp( -answer * dot );

	value += weight * log1p( expCoeff );

	gradient.MultiplyAndAddExt( desc, -weight * LogNormalizer * answer * expCoeff / ( 1. + expCoeff ) );
	hessian = weight * LogNormalizer * expCoeff / ( 1. + expCoeff ) / ( 1. + expCoeff );
}

void CLogRegressionState::PostReductionSetArgument()
{
	Value *= LogNormalizer;
	Value += RValue;
}

//-----------------------------------------------------------------------------------------------------------------------

struct CSmoothedHingeState : public CFunctionWithHessianState {
	using CFunctionWithHessianState::CFunctionWithHessianState; // inherit constructor of the base class
	void SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
		double& hessian, double& value, CFloatVector& gradient, float answer, float weight ) override;
};

void CSmoothedHingeState::SetArgument( const CFloatVector& argument, const CFloatVectorDesc& desc,
	double& hessian, double& value, CFloatVector& gradient, float answer, float weight )
{
	const double d = answer * LinearFunction( argument, desc ) - 1.;

	if( d < 0. ) {
		const double sqrtValue = sqrt( d * d + 1. );
		value += weight * ( sqrtValue - 1. );
		gradient.MultiplyAndAddExt( desc, weight * answer * d / sqrtValue );
		hessian = weight / ( ( d * d + 1. ) * sqrtValue );
	} else {
		hessian = 0.;
	}
}

} // namespace

//-----------------------------------------------------------------------------------------------------------------------

CFunctionWithHessianState* CFunctionWithHessianState::Create( CFloatMatrixDesc matrix, int vectorCount,
	double errorWeight, double p, float l1Coeff, int threadCount, THessianFType type )
{
	switch( type ) {
		case THessianFType::SquaredHinge:
			return new CSquaredHingeState( matrix, vectorCount, errorWeight, l1Coeff, threadCount, type );
		case THessianFType::L2Regression:
			return new CL2RegressionState( matrix, vectorCount, errorWeight, p, l1Coeff, threadCount, type );
		case THessianFType::LogRegression:
			return new CLogRegressionState( matrix, vectorCount, errorWeight, l1Coeff, threadCount, type );
		case THessianFType::SmoothedHinge:
			return new CSmoothedHingeState( matrix, vectorCount, errorWeight, l1Coeff, threadCount, type );
		default:
			NeoAssert( false );
	}
	return nullptr;
}

} // namespace NeoML
