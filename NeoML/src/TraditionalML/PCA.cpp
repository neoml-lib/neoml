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
#include <memory>

using namespace NeoML;

CPCA::CPCA( const CParams& _params ): params( _params )
{
	NeoAssert( ( params.ComponentsType == PCAC_None ) ||
		( ( params.ComponentsType == PCAC_Int ) && ( params.Components > 0 ) ) ||
		( ( params.ComponentsType == PCAC_Float ) && ( 0 < params.Components ) && ( params.Components < 1 ) ) );
}

static CPtr<CDnnBlob> convertToBlob( IMathEngine& mathEngine, const CFloatMatrixDesc& data )
{
	const int vectorCount = data.Height;
	const int featureCount = data.Width;
	CPtr<CDnnBlob> result = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, vectorCount, featureCount );
	CFloatHandle currData = result->GetData();
	if( data.Columns == nullptr ) {
		mathEngine.DataExchangeTyped( currData, data.Values, vectorCount * featureCount );
	} else {
		CArray<float> row;
		for( int r = 0; r < vectorCount; ++r ) {
			row.Empty();
			row.Add( 0, featureCount );
			for( int i = data.PointerB[r]; i < data.PointerE[r]; i++ ) {
				row[data.Columns[i]] = data.Values[i];
			}
			mathEngine.DataExchangeTyped( currData, row.GetPtr(), featureCount );
			currData += featureCount;
		}
	}
	return result;
}

static void subtractMean( IMathEngine& mathEngine, const CFloatHandle& data, int matrixHeight, int matrixWidth )
{
	CPtr<CDnnBlob> meanVector = CDnnBlob::CreateVector( mathEngine, CT_Float, matrixWidth );
	CPtr<CDnnBlob> height = CDnnBlob::CreateVector( mathEngine, CT_Float, 1 );
	const CFloatHandle& heightHandle = height->GetData();
	heightHandle.SetValue( -1.f / static_cast<float>( matrixHeight ) );
	mathEngine.SumMatrixRows( 1, meanVector->GetData(), data, matrixHeight, matrixWidth );
	mathEngine.VectorMultiply( meanVector->GetData(), meanVector->GetData(), matrixWidth, heightHandle );
	mathEngine.AddVectorToMatrixRows( 1, data, data, matrixHeight, matrixWidth, meanVector->GetData() );
}

static void flipSVD( IMathEngine& mathEngine, const CPtr<CDnnBlob>& u, const CFloatHandle& vt, int m, int k, int n )
{
	CPtr<CDnnBlob> maxValues = CDnnBlob::CreateVector( mathEngine, CT_Float, k );
	maxValues->Fill( 0 );
	float* uPtr = u->GetBuffer<float>( 0, m * k, false );
	float* maxValuesPtr = maxValues->GetBuffer<float>( 0, k, false );

	for( int row = 0; row < m; row++ ) {
		for( int col = 0; col < k; col++ ) {
			if( abs( uPtr[row * k + col] ) > abs( maxValuesPtr[col] ) ) {
				maxValuesPtr[col] = uPtr[row * k + col];
			}
		}
	}
	for( int col = 0; col < k; col++ ) {
		maxValuesPtr[col] = ( ( maxValuesPtr[col] >= 0 ) ? 1.f : -1.f );
	}
	maxValues->ReleaseBuffer( maxValuesPtr, false );
	mathEngine.MultiplyMatrixByDiagMatrix( u->GetData(), m, k, maxValues->GetData<float>(), u->GetData(), m * k );
	mathEngine.MultiplyDiagMatrixByMatrix( maxValues->GetData<float>(), k, vt, n, vt, n * k );
}

static void blobToMatrix( const CPtr<CDnnBlob>& blob, CSparseFloatMatrix& matrix, int matrixHeight, int matrixWidth, int blobWidth )
{
	CFloatVectorDesc row;
	row.Size = matrixWidth;

	int pos = 0;
	for( int i = 0; i < matrixHeight; i++, pos += blobWidth ) {
		row.Values = blob->GetBuffer<float>( pos, matrixWidth, false );
		matrix.AddRow( row );
	}
}

void CPCA::getComponentsNum( const CArray<float>& explainedVarianceRatio, int k )
{
	if( params.ComponentsType == PCAC_None ) {
		components = k;
	} else if( params.ComponentsType == PCAC_Int ) {
		components = static_cast<int>( params.Components );
		NeoAssert( components <= k );
	} else if( params.ComponentsType == PCAC_Float ) {
		float currentSum = 0;
		components = explainedVarianceRatio.Size();
		for( int i = 0; i < explainedVarianceRatio.Size(); i++ ) {
			currentSum += explainedVarianceRatio[i];
			if( currentSum > params.Components ) {
				components = i + 1;
				break;
			}
		}
	} else {
		NeoAssert( false );
	}
}

void CPCA::calculateVariance( IMathEngine& mathEngine, const CFloatHandle& s, int m, int k )
{
	CPtr<CDnnBlob> var = CDnnBlob::CreateVector( mathEngine, CT_Float, k );
	CPtr<CDnnBlob> temp = CDnnBlob::CreateVector( mathEngine, CT_Float, 1 );
	const CFloatHandle& tempHandle = temp->GetData();

	// calculate explained_variance
	mathEngine.VectorEltwiseMultiply( s, s, var->GetData(), k );
	tempHandle.SetValue( 1.f / static_cast<float>( m - 1 ) );
	mathEngine.VectorMultiply( var->GetData(), var->GetData(), k, tempHandle );
	explainedVariance.SetSize( k );
	var->CopyTo( explainedVariance.GetPtr(), k );

	// calculate explained_variance_ratio
	mathEngine.VectorSum( var->GetData(), k, tempHandle );
	tempHandle.SetValue( 1.f / tempHandle.GetValue() );
	mathEngine.VectorMultiply( var->GetData(), var->GetData(), k, tempHandle );
	explainedVarianceRatio.SetSize( k );
	var->CopyTo( explainedVarianceRatio.GetPtr(), k );

	getComponentsNum( explainedVarianceRatio, k );

	// calculate noise_variance
	noiseVariance = 0;
	for( int i = components; i < k; i++ ) {
		noiseVariance += explainedVariance[i];
	}
	noiseVariance /= max( 1, k - components );

	explainedVariance.SetSize( components );
	explainedVarianceRatio.SetSize( components );
}

void CPCA::train( const CFloatMatrixDesc& data, bool isTransform )
{
	int n = data.Width;
	int m = data.Height;
	int k = min( n, m );

	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( 1, 0 ) );
	CPtr<CDnnBlob> a = convertToBlob( *mathEngine, data );
	CPtr<CDnnBlob> u = CDnnBlob::CreateDataBlob( *mathEngine, CT_Float, 1, m, k );
	CPtr<CDnnBlob> s = CDnnBlob::CreateVector( *mathEngine, CT_Float, k );
	CPtr<CDnnBlob> vt = CDnnBlob::CreateDataBlob( *mathEngine, CT_Float, 1, k, n );
	CPtr<CDnnBlob> superb = CDnnBlob::CreateVector( *mathEngine, CT_Float, k - 1 );

	// data -= mean(data)
	subtractMean( *mathEngine, a->GetData(), m, n );

	mathEngine->SingularValueDecomposition( a->GetData(), n, m, u->GetData(), s->GetData(), vt->GetData(), superb->GetData() );

	// flip signs of u columns and vt rows to obtain deterministic result
	flipSVD( *mathEngine, u, vt->GetData(), m, k, n );

	calculateVariance( *mathEngine, s->GetData(), m, k );

	singularValues.SetSize( components );
	s->CopyTo( singularValues.GetPtr(), components );
	componentsMatrix = CSparseFloatMatrix( components, n, components * n );
	blobToMatrix( vt, componentsMatrix, components, n, n );

	if( isTransform ) {
		transformedMatrix = CSparseFloatMatrix( components, m, components * m );
		mathEngine->MultiplyMatrixByDiagMatrix( u->GetData(), m, k, s->GetData(), u->GetData(), m * k );
		blobToMatrix( u, transformedMatrix, m, components, k );
	}
}

void CPCA::Train( const CFloatMatrixDesc& data )
{
	train( data, false );
}

CSparseFloatMatrixDesc CPCA::Transform( const CFloatMatrixDesc& data )
{
	train( data, true );
	return transformedMatrix.GetDesc();
}
