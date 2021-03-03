/* Copyright © 2021 ABBYY Production LLC

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

#include <RandomProblem.h>

namespace NeoMLTest {

CClassificationRandomProblem::CClassificationRandomProblem( int height, int width, float* values, const int* _classes, const float* _weights ) :
	matrix( width ),
	classCount( 0 ),
	classes( _classes ),
	weights( _weights )
{
	CSparseFloatMatrixDesc* desc = matrix.CopyOnWrite();
	NeoAssert( desc != nullptr );
	desc->Height = height;
	desc->Width = width;
	desc->Values = values;
	desc->Columns = nullptr;
	desc->PointerB = desc->PointerE = nullptr;

	for( int i = 0; i < height; i++ ) {
		if( classCount < classes[i] ) {
			classCount = classes[i];
		}
	}
	classCount++;
}

CPtr<CClassificationRandomProblem> CClassificationRandomProblem::Random( CRandom& rand, int samples, int features, int classes )
{
	CPtr<CClassificationRandomProblem> res = new CClassificationRandomProblem();

	res->valuesArr.SetBufferSize( samples * features );
	res->classesArr.SetBufferSize( samples );
	for( int i = 0; i < samples; ++i ) {
		for( int j = 0; j < features; ++j ) {
			if( rand.UniformInt( 0, 3 ) != 0 ) { // 1/4 probability of null element
				res->valuesArr.Add( rand.Uniform( -10, 10 ) );
			} else {
				res->valuesArr.Add( 0.0 );
			}
		}
		res->classesArr.Add( rand.UniformInt( 0, classes - 1 ) );
	}

	// set weights to 1
	res->weightsArr.Add( 1., samples );
	res->classCount = classes;
	res->classes = res->classesArr.GetPtr();
	res->weights = res->weightsArr.GetPtr();

	res->matrix = CSparseFloatMatrix( features );
	CSparseFloatMatrixDesc* desc = res->matrix.CopyOnWrite();
	desc->Height = samples;
	desc->Width = features;
	desc->Values = res->valuesArr.GetPtr();
	desc->Columns = nullptr;
	desc->PointerB = desc->PointerE = nullptr;

	return res;
}

CPtr<CClassificationRandomProblem> CClassificationRandomProblem::CreateSparse() const
{
	CSparseFloatMatrix sparse( GetMatrix() ); // convert here dense into sparse
	CPtr<CClassificationRandomProblem> res = new CClassificationRandomProblem();
	res->matrix = sparse;
	weightsArr.CopyTo( res->weightsArr );
	res->weights = res->weightsArr.GetPtr();
	classesArr.CopyTo( res->classesArr );
	res->classes = res->classesArr.GetPtr();
	res->classCount = classCount;
	return res;
}

} // namespace NeoMLTest
