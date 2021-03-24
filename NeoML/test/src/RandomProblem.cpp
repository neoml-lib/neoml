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

template<typename TLabel>
CPtr< CRandomProblemImpl<TLabel> > CRandomProblemImpl<TLabel>::Random( CRandom& rand, int samples, int features, int labelsCount )
{
	CPtr< CRandomProblemImpl<TLabel> > res = new CRandomProblemImpl();

	res->Matrix = CSparseFloatMatrix( features, samples, samples * features );
	CFloatMatrixDesc* desc = res->Matrix.CopyOnWrite();
	desc->Height = samples;
	desc->Width = features;
	desc->Columns = nullptr;

	res->LabelsArr.SetBufferSize( samples );
	uint32_t pos = 0;
	for( int i = 0; i < samples; ++i ) {
		desc->PointerB[i] = pos;
		for( int j = 0; j < features; ++j, ++pos ) {
			if( rand.UniformInt( 0, 3 ) != 0 ) { // 1/4 probability of null element
				desc->Values[pos] = static_cast<float>( rand.Uniform( -10, 10 ) );
			} else {
				desc->Values[pos] = 0.0;
			}
		}
		desc->PointerE[i] = pos;
		res->LabelsArr.Add( static_cast<TLabel>( rand.UniformInt( 0, labelsCount - 1 ) ) );
	}

	// set Weights to 1
	res->WeightsArr.Add( 1., samples );
	res->Labels = res->LabelsArr.GetPtr();
	res->Weights = res->WeightsArr.GetPtr();

	return res;
}

template<typename TLabel>
CPtr< CRandomProblemImpl<TLabel> > CRandomProblemImpl<TLabel>::CreateSparse() const
{
	CSparseFloatMatrix sparse( Matrix.GetDesc() ); // convert here dense into sparse
	CPtr< CRandomProblemImpl<TLabel> > res = new CRandomProblemImpl();
	res->Matrix = sparse;
	WeightsArr.CopyTo( res->WeightsArr );
	res->Weights = res->WeightsArr.GetPtr();
	LabelsArr.CopyTo( res->LabelsArr );
	res->Labels = res->LabelsArr.GetPtr();
	return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CPtr<CClassificationRandomProblem> CClassificationRandomProblem::Random( CRandom& rand, int samples, int features, int classCount )
{
	CPtr<CClassificationRandomProblem> res = new CClassificationRandomProblem();
	res->impl = CRandomProblemImpl<int>::Random( rand, samples, features, classCount );
	res->classCount = classCount;
	return res;
}

CPtr<CClassificationRandomProblem> CClassificationRandomProblem::CreateSparse() const
{
	CPtr<CClassificationRandomProblem> res = new CClassificationRandomProblem();
	res->impl = impl->CreateSparse();
	res->classCount = classCount;
	return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CPtr<CRegressionRandomProblem> CRegressionRandomProblem::Random( CRandom& rand, int samples, int features, int labelsCount )
{
	CPtr<CRegressionRandomProblem> res = new CRegressionRandomProblem();
	res->impl = CRandomProblemImpl<double>::Random( rand, samples, features, labelsCount );
	return res;
}

CPtr<CRegressionRandomProblem> CRegressionRandomProblem::CreateSparse() const
{
	CPtr<CRegressionRandomProblem> res = new CRegressionRandomProblem();
	res->impl = impl->CreateSparse();
	return res;
}

} // namespace NeoMLTest
