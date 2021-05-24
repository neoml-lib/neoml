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

#include <NeoML/Dnn/DnnBlob.h>
#include <NeoML/TraditionalML/PCA.h>

namespace NeoML {

CPCA::CPCA( IMathEngine& _mathEngine ):
	mathEngine( _mathEngine ){};

static CPtr<CDnnBlob> createDataBlob( IMathEngine& mathEngine, const CFloatMatrixDesc& data )
{
	NeoAssert( data.Columns == nullptr );
	const int vectorCount = data.Height;
	const int featureCount = data.Width;
	CPtr<CDnnBlob> result = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, vectorCount, featureCount );
	CFloatHandle currData = result->GetData();
	for( int row = 0; row < data.Height; ++row ) {
		mathEngine.DataExchangeTyped( currData, data.Values + data.PointerB[row], featureCount );
		currData += featureCount;
	}
	return result;
}

CPtr<IModel> CPCA::Train( const IPCAData& data )
{
	int n = data.GetFeaturesCount(), m = data.GetVectorCount();
	CPtr<CDnnBlob> a = createDataBlob( mathEngine, data.GetMatrix() );
	CPtr<CDnnBlob> u = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, m, m );
	CPtr<CDnnBlob> s = CDnnBlob::CreateVector( mathEngine, CT_Float, min( n, m ) );
	CPtr<CDnnBlob> vt = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, n, n );
	CPtr<CDnnBlob> superb = CDnnBlob::CreateVector( mathEngine, CT_Float, min( n, m ) - 1 );

	mathEngine.SingularValueDecomposition( a->GetData(), n, m, u->GetData(), s->GetData(), vt->GetData(), superb->GetData() );

	{
		CArray<float> res;
		res.SetSize( s->GetDataSize() );
		s->CopyTo( res.GetPtr() );
		for( int i = 0; i < res.Size(); i++ ) {
			printf( "%f ", res[i] );
		}
		printf("\n");
	}

	{
		CArray<float> res;
		res.SetSize( a->GetDataSize() );
		a->CopyTo( res.GetPtr() );
		for( int i = 0; i < res.Size(); i++ ) {
			printf( "%f ", res[i] );
		}
		printf("\n");
	}

	{
		CArray<float> res;
		res.SetSize( vt->GetDataSize() );
		vt->CopyTo( res.GetPtr() );
		for( int i = 0; i < res.Size(); i++ ) {
			printf( "%f ", res[i] );
		}
		printf("\n");
	}

	{
		CArray<float> res;
		res.SetSize( superb->GetDataSize() );
		superb->CopyTo( res.GetPtr() );
		for( int i = 0; i < res.Size(); i++ ) {
			printf( "%f ", res[i] );
		}
		printf("\n");
	}
	return nullptr;
}

} // namespace NeoML