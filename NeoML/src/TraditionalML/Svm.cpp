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

#include <NeoML/TraditionalML/Svm.h>
#include <NeoML/TraditionalML/OneVersusAll.h>
#include <NeoML/TraditionalML/OneVersusOne.h>
#include <NeoML/TraditionalML/PlattScalling.h>
#include <SvmBinaryModel.h>
#include <LinearBinaryModel.h>
#include <NeoMathEngine/ThreadPool.h>
#include <SMOptimizer.h>

namespace NeoML {

namespace {

// Task which processes in multiple threads
struct ISvmThreadTask {
	virtual ~ISvmThreadTask() {}
	// Run in a single thread or in parallel, corresponding of task's `ParallelizeSize()`
	void ParallelRun();
protected:
	// Create a task
	ISvmThreadTask( IThreadPool&, const IProblem& );

	// The size of parallelization, max number of sub-tasks to perform
	int ParallelizeSize() const { return Problem.GetVectorCount(); }
	// The number of separate executors
	int ThreadCount() const { return ThreadPool.Size(); }
	// Get way of split the task into sub-tasks
	void RunSplittedByThreads( int threadIndex );
	// Run the problem vector in a separate thread, cycle on sub-tasks
	void Run( int threadIndex, int startIndex, int count );
	// Run on each element of the problem vector separately
	virtual void RunOnElement( int threadIndex, int index, const CFloatVectorDesc& ) = 0;

	static constexpr int MultiThreadMinTasksCount = 2;
	IThreadPool& ThreadPool; //parallel executors
	const IProblem& Problem; //performing problem
	const CFloatMatrixDesc Matrix; //performing problem's sizes
};

ISvmThreadTask::ISvmThreadTask( IThreadPool& threadPool, const IProblem& problem ) :
	ThreadPool( threadPool ),
	Problem( problem ),
	Matrix( Problem.GetMatrix() )
{
	NeoAssert( Matrix.Height == Problem.GetVectorCount() );
	NeoAssert( Matrix.Width == Problem.GetFeatureCount() );
}

void ISvmThreadTask::ParallelRun()
{
	if( ParallelizeSize() < MultiThreadMinTasksCount ) {
		// Run in a single thread
		Run( /*threadIndex*/0, /*index*/0, ParallelizeSize() );
		return;
	}
	// Run in parallel
	NEOML_NUM_THREADS( ThreadPool, this, []( int threadIndex, void* ptr ) {
		( ( ISvmThreadTask* )ptr )->RunSplittedByThreads( threadIndex );
	} );
}

void ISvmThreadTask::RunSplittedByThreads( int threadIndex )
{
	int index = 0;
	int count = 0;
	// 1 dimensional split
	if( GetTaskIndexAndCount( ThreadCount(), threadIndex, ParallelizeSize(), index, count ) ) {
		Run( threadIndex, index, count );
	}
}

void ISvmThreadTask::Run( int threadIndex, int startIndex, int count )
{
	const int endIndex = startIndex + count;
	for( int index = startIndex; index < endIndex; ++index ) {
		CFloatVectorDesc desc;
		Matrix.GetRow( index, desc );
		// main function call
		RunOnElement( threadIndex, index, desc );
	}
}

//------------------------------------------------------------------------------------------------------------

struct CSvmFindPlanesThreadTask : public ISvmThreadTask {
	// Create a task
	CSvmFindPlanesThreadTask( IThreadPool& threadPool, const IProblem& problem, CArray<double>& alpha ) :
		ISvmThreadTask( threadPool, problem ),
		Alpha( alpha )
	{ PlaneReduction.Add( CFloatVector( problem.GetFeatureCount() + 1, 0.f ), ThreadCount() ); }
	// Get the final result
	CFloatVector Reduction( float freeTerm );
protected:
	void RunOnElement( int threadIndex, int index, const CFloatVectorDesc& ) override;

	CArray<double>& Alpha;
	CArray<CFloatVector> PlaneReduction{};
};

void CSvmFindPlanesThreadTask::RunOnElement( int threadIndex, int index, const CFloatVectorDesc& desc )
{
	const float alpha = static_cast<float>( Alpha[index] * Problem.GetBinaryClass( index ) );
	PlaneReduction[threadIndex].MultiplyAndAdd( desc, alpha );
}

CFloatVector CSvmFindPlanesThreadTask::Reduction( float freeTerm )
{
	CFloatVector plane = PlaneReduction[0];
	for( int t = 1; t < ThreadCount(); ++t ) {
		plane += PlaneReduction[t];
	}
	plane.SetAt( Problem.GetFeatureCount(), freeTerm );
	return plane;
}

//-------------------------------------------------------------------------------------------------------------

struct CSvmCalcDistancesThreadTask : public ISvmThreadTask {
	// Create a task
	CSvmCalcDistancesThreadTask( IThreadPool& threadPool, const IProblem& problem, const CFloatVector& plane ) :
		ISvmThreadTask( threadPool, problem ),
		Plane( plane )
	{ Dist.Add( 0.0, ParallelizeSize() ); }
	// Get the final result
	const CArray<double>& Distances() const { return Dist; }
protected:
	void RunOnElement( int /*threadIndex*/, int index, const CFloatVectorDesc& desc ) override
	{ Dist[index] = LinearFunction( Plane, desc ); }

	const CFloatVector& Plane;
	CArray<double> Dist{};
};

} // namespace

//-------------------------------------------------------------------------------------------------------------

CSvm::CSvm( const CParams& _params ) :
	threadPool( CreateThreadPool( _params.ThreadCount ) ),
	params( _params, threadPool->Size() )
{
	NeoAssert( threadPool != nullptr );
}

CSvm::~CSvm()
{
	delete threadPool;
}

CPtr<IModel> CSvm::Train( const IProblem& problem )
{
	if( problem.GetClassCount() > 2 ) {
		static_assert( MM_Count == 3, "MM_Count != 3" );
		switch( params.MulticlassMode ) {
			case MM_OneVsAll:
				return COneVersusAll( *this ).Train( problem );
			case MM_OneVsOne:
				return COneVersusOne( *this ).Train( problem );
			case MM_SingleClassifier:
			default:
				NeoAssert( false );
		}
		return nullptr;
	}

	const CSvmKernel kernel( params.KernelType, params.Degree, params.Gamma, params.Coeff0 );

	CSMOptimizer optimizer( kernel, problem, params.MaxIterations,
		params.ErrorWeight, params.Tolerance, params.DoShrinking );
	if( log != nullptr ) {
		optimizer.SetLog( log );
	}

	CArray<double> alpha{};
	float freeTerm{};
	optimizer.Optimize( alpha, freeTerm );

	if( kernel.KernelType() == CSvmKernel::KT_Linear ) {
		CSvmFindPlanesThreadTask planeTask( *threadPool, problem, alpha );
		planeTask.ParallelRun();
		CFloatVector plane = planeTask.Reduction( freeTerm );

		CSvmCalcDistancesThreadTask distTask( *threadPool, problem, plane );
		distTask.ParallelRun();

		CSigmoid coefficients{};
		CalcSigmoidCoefficients( problem, distTask.Distances(), coefficients );
		return FINE_DEBUG_NEW CLinearBinaryModel( plane, coefficients );
	} else {
		return FINE_DEBUG_NEW CSvmBinaryModel( kernel, problem, alpha, freeTerm );
	}
}

} // namespace NeoML
