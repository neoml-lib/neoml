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

struct CSvm::IThreadTask {
	virtual ~IThreadTask() {}

	void CallRun( int threadIndex );
	int Size() const { return Problem.GetVectorCount(); }

protected:
	IThreadTask( int threadCount, const IProblem& );

	virtual void Run( int threadIndex, int index, const CFloatVectorDesc& ) = 0;

	const int ThreadCount;
	const IProblem& Problem;
	const CFloatMatrixDesc Matrix;
};

CSvm::IThreadTask::IThreadTask( int threadCount, const IProblem& problem ) :
	ThreadCount( threadCount ),
	Problem( problem ),
	Matrix( Problem.GetMatrix() )
{
	NeoAssert( Matrix.Height == Problem.GetVectorCount() );
	NeoAssert( Matrix.Width == Problem.GetFeatureCount() );
}

void CSvm::IThreadTask::CallRun( int threadIndex )
{
	int index = 0;
	int count = 0;
	if( GetTaskIndexAndCount( ThreadCount, threadIndex, Size(), index, count ) ) {
		for( int i = 0; i < count; ++i, ++index ) {
			CFloatVectorDesc desc;
			Matrix.GetRow( index, desc );

			Run( threadIndex, index, desc );
		}
	}
}

//------------------------------------------------------------------------------------------------------------

struct CSvm::CFindPlanesThreadTask : public CSvm::IThreadTask {
	CFindPlanesThreadTask( int threadCount, const IProblem& problem, CArray<double>& alpha ) :
		CSvm::IThreadTask( threadCount, problem ),
		Alpha( alpha )
	{ PlaneReduction.Add( CFloatVector( problem.GetFeatureCount() + 1, 0.f ), ThreadCount ); }

	CFloatVector Reduction( float freeTerm );
protected:
	void Run( int threadIndex, int index, const CFloatVectorDesc& ) override;

	CArray<double>& Alpha;
	CArray<CFloatVector> PlaneReduction{};
};

void CSvm::CFindPlanesThreadTask::Run( int threadIndex, int index, const CFloatVectorDesc& desc )
{
	const float alpha = static_cast<float>( Alpha[index] * Problem.GetBinaryClass( index ) );
	PlaneReduction[threadIndex].MultiplyAndAdd( desc, alpha );
}

CFloatVector CSvm::CFindPlanesThreadTask::Reduction( float freeTerm )
{
	CFloatVector plane = PlaneReduction[0];
	for( int t = 1; t < ThreadCount; ++t ) {
		plane += PlaneReduction[t];
	}
	plane.SetAt( Problem.GetFeatureCount(), freeTerm );
	return plane;
}

//-------------------------------------------------------------------------------------------------------------

struct CSvm::CCalcDistancesThreadTask : public CSvm::IThreadTask {
	CCalcDistancesThreadTask( int threadCount, const IProblem& problem, const CFloatVector& plane ) :
		CSvm::IThreadTask( threadCount, problem ),
		Plane( plane )
	{ Dist.Add( 0.0, Size() ); }

	const CArray<double>& Distances() const { return Dist; }
protected:
	void Run( int /*threadIndex*/, int index, const CFloatVectorDesc& desc ) override
	{ Dist[index] = LinearFunction( Plane, desc ); }

	const CFloatVector& Plane;
	CArray<double> Dist{};
};

//-------------------------------------------------------------------------------------------------------------

CSvm::CSvm( const CParams& params )
	: params( params ),
	threadPool( CreateThreadPool( params.ThreadCount ) )
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
		CFindPlanesThreadTask planeTask( params.ThreadCount, problem, alpha );
		NEOML_NUM_THREADS( *threadPool, &planeTask, []( int threadIndex, void* ptr ) {
			( ( IThreadTask* )ptr )->CallRun( threadIndex );
		} );
		CFloatVector plane = planeTask.Reduction( freeTerm );

		CCalcDistancesThreadTask distTask( params.ThreadCount, problem, plane );
		NEOML_NUM_THREADS( *threadPool, &distTask, []( int threadIndex, void* ptr ) {
			( ( IThreadTask* )ptr )->CallRun( threadIndex );
		} );

		CSigmoid coefficients;
		CalcSigmoidCoefficients( problem, distTask.Distances(), coefficients );
		return FINE_DEBUG_NEW CLinearBinaryModel( plane, coefficients );
	} else {
		return FINE_DEBUG_NEW CSvmBinaryModel( kernel, problem, alpha, freeTerm );
	}
}

} // namespace NeoML
