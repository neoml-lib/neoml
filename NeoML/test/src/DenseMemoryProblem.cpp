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

#include <DenseMemoryProblem.h>

namespace NeoMLTest {

CDenseMemoryProblem::CDenseMemoryProblem( int height, int width, float* values, const int* _classes, const float* _weights ) :
	classCount( 0 ),
	classes( _classes ),
	weights( _weights )
{
	desc.Height = height;
	desc.Width = width;
	desc.Values = values;

	for( int i = 0; i < height; i++ ) {
		if( classCount < classes[i] ) {
			classCount = classes[i];
		}
	}
	classCount++;
}

CPtr<CDenseMemoryProblem> CDenseMemoryProblem::Random( int samples, int features, int classes )
{
	CPtr<CDenseMemoryProblem> res = new CDenseMemoryProblem();

	CRandom rand( 0 );
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
	res->desc.Height = samples;
	res->desc.Width = features;
	res->desc.Values = res->valuesArr.GetPtr();

	return res;
}

CPtr<CMemoryProblem> CDenseMemoryProblem::CreateSparse() const
{
	CPtr<CMemoryProblem> res = new CMemoryProblem( desc.Width, classCount );
	for( int i = 0; i < desc.Height; ++i ) {
		CSparseFloatVectorDesc row = desc.GetRow( i );
		res->Add( row, weights[i], classes[i] );
	}
	return res;
}

} // namespace NeoMLTest
