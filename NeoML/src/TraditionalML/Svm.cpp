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

#include <NeoML/TraditionalML/Svm.h>
#include <NeoML/TraditionalML/OneVersusAll.h>
#include <NeoML/TraditionalML/PlattScalling.h>
#include <SvmBinaryModel.h>
#include <LinearBinaryModel.h>
#include <NeoMathEngine/OpenMP.h>
#include <SMOptimizer.h>

namespace NeoML {

CSvm::CSvm( const CParams& _params ) :
	params( _params ),
	log( nullptr )
{
}

CPtr<IModel> CSvm::Train( const IProblem& problem )
{
	if( problem.GetClassCount() > 2 ) {
		return COneVersusAll( *this ).Train( problem );
	}

	CSvmKernel kernel( params.KernelType, params.Degree, params.Gamma, params.Coeff0 );

	CSMOptimizer optimizer( kernel, problem, params.MaxIterations, params.ErrorWeight, params.Tolerance,
		params.DoShrinking );
	if( log != nullptr ) {
		optimizer.SetLog( log );
	}

	CArray<double> alpha;
	float freeTerm;
	optimizer.Optimize( alpha, freeTerm );

	if( kernel.KernelType() == CSvmKernel::KT_Linear ) {
		const int vectorCount = problem.GetVectorCount();
		const int curThreadCount = IsOmpRelevant( vectorCount ) ? params.ThreadCount : 1;
		const CFloatMatrixDesc matrix = problem.GetMatrix();
		NeoAssert( matrix.Height == problem.GetVectorCount() );
		NeoAssert( matrix.Width == problem.GetFeatureCount() );

		CArray<CFloatVector> planeReduction;
		planeReduction.Add( CFloatVector( problem.GetFeatureCount() + 1, 0.f ), curThreadCount );

		NEOML_OMP_NUM_THREADS( curThreadCount )
		{
			const int threadNumber = OmpGetThreadNum();
			CFloatVector& plane = planeReduction[threadNumber];

			int index = 0;
			int count = 0;
			if( OmpGetTaskIndexAndCount( vectorCount, index, count ) ) {
				for( int i = 0; i < count; i++ ) {
					float alphaValue = static_cast<float>( alpha[index] * problem.GetBinaryClass(index) );
					CFloatVectorDesc desc;
					matrix.GetRow( index, desc );
					plane.MultiplyAndAdd( desc, alphaValue );
					index++;
				}
			}
		}

		CFloatVector plane = planeReduction[0];
		for( int i = 1; i < planeReduction.Size(); i++ ) {
			plane += planeReduction[i];
		}
		plane.SetAt( problem.GetFeatureCount(), freeTerm );

		CArray<double> distances;
		distances.Add( 0.0, vectorCount );

		NEOML_OMP_NUM_THREADS( curThreadCount )
		{
			int index = 0;
			int count = 0;
			if( OmpGetTaskIndexAndCount( vectorCount, index, count ) ) {
				for( int i = 0; i < count; i++ ) {
					CFloatVectorDesc desc;
					matrix.GetRow( index, desc );
					distances[index] = LinearFunction( plane, desc );
					index++;
				}
			}
		}

		CSigmoid coefficients;
		CalcSigmoidCoefficients( problem, distances, coefficients );
		return FINE_DEBUG_NEW CLinearBinaryModel( plane, coefficients );
	} else {
		return FINE_DEBUG_NEW CSvmBinaryModel( kernel, problem, alpha, freeTerm );
	}
}

} // namespace NeoML
