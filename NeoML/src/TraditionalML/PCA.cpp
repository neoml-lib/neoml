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
#include <NeoML/TraditionalML/FeatureSelection.h>

using namespace NeoML;

class CFeatureVarianceProblem : public IProblem {
public:
	CFeatureVarianceProblem( const CFloatMatrixDesc& _desc ) : desc( _desc ) {};

	virtual int GetClassCount() const { return 1; }
	virtual bool IsDiscreteFeature( int ) const { return false; }
	virtual int GetFeatureCount() const { return desc.Width; }
	virtual int GetVectorCount() const { return desc.Height; }
	virtual int GetClass( int ) const { return 0; }
	virtual CFloatMatrixDesc GetMatrix() const { return desc; }
	virtual double GetVectorWeight( int ) const { return 1; }

private:
	CFloatMatrixDesc desc;
};

// Convert CFloatMatrix to CDnnBlob
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

void reverseArrays( CArray<float>& leftVectors, CArray<float>& singularVectors, CArray<float>& rightVectors,
	int m, int components, int n, bool hasLeft, bool hasRight )
{
	if( hasLeft ) {
		for( int row = 0; row < m; row++ ) {
			int leftIndex = row * components;
			int rightIndex = ( row + 1 ) * components - 1;
			for( int col = 0; col < components / 2; col++ ) {
				float temp = leftVectors[leftIndex];
				leftVectors[leftIndex] = leftVectors[rightIndex];
				leftVectors[rightIndex] = temp;
				leftIndex++;
				rightIndex--;
			}
		}
	}

	for( int i = 0; i < components / 2; i++ ) {
		float temp = singularVectors[i];
		singularVectors[i] = singularVectors[components - i - 1];
		singularVectors[components - i - 1] = temp;
	}

	if( hasRight ) {
		for( int row = 0; row < components / 2; row++ ) {
			int leftIndex = row * n;
			int rightIndex = ( components - row - 1 ) * n;
			for( int col = 0; col < n; col++ ) {
				float temp = rightVectors[leftIndex];
				rightVectors[leftIndex] = rightVectors[rightIndex];
				rightVectors[rightIndex] = temp;
				leftIndex++;
				rightIndex++;
			}
		}
	}
}

CSparseMatrixDesc getSparseMatrixDesc( IMathEngine& mathEngine, const CFloatMatrixDesc& data,
	CPtr<CDnnBlob>& columns, CPtr<CDnnBlob>& rows, CPtr<CDnnBlob>& values )
{
	int m = data.Height;
	int n = data.Width;
	CSparseMatrixDesc desc;
	desc.ElementCount = ( data.Columns == nullptr ) ? m * n : data.PointerE[m - 1];
	columns = CDnnBlob::CreateVector( mathEngine, CT_Int, desc.ElementCount );
	rows = CDnnBlob::CreateVector( mathEngine, CT_Int, m + 1 );
	values = CDnnBlob::CreateVector( mathEngine, CT_Float, desc.ElementCount );
	if( data.Columns != nullptr ) {
		columns->CopyFrom( data.Columns );
		int* rowBuffer = rows->GetBuffer<int>( 0, m + 1, false );
		for( int i = 0; i < m; i++ ) {
			rowBuffer[i] = data.PointerB[i];
		}
		rowBuffer[m] = data.PointerE[m - 1];
		rows->ReleaseBuffer( rowBuffer, true );
	} else {
		desc.ElementCount = m * n;
		int* colBuffer = columns->GetBuffer<int>( 0, m * n, false );
		int* rowBuffer = rows->GetBuffer<int>( 0, m + 1, false );
		for( int i = 0; i < m; i++ ) {
			rowBuffer[i] = i * n;
			for( int j = 0; j < n; j++ ) {
				colBuffer[i * n + j] = j;
			}
		}
		rowBuffer[m] = m * n;
		rows->ReleaseBuffer( rowBuffer, true );
		columns->ReleaseBuffer( colBuffer, true );
	}
	desc.Columns = columns->GetData<int>();
	values->CopyFrom( data.Values );
	desc.Values = values->GetData<float>();
	desc.Rows = rows->GetData<int>();
	return desc;
}

namespace NeoML {

void SingularValueDecomposition( const CFloatMatrixDesc& data, const TSvd& svdSolver,
	CArray<float>& leftVectors_, CArray<float>& singularValues_, CArray<float>& rightVectors_,
	bool returnLeftVectors, bool returnRightVectors, int components )
{
	if( svdSolver == SVD_Sparse && ( returnLeftVectors == returnRightVectors ) ) {
		// exactly one flag must be true
		NeoAssert( false );
	}
	int n = data.Width;
	int m = data.Height;

	if( svdSolver == SVD_Full || components == 0 ) {
		components = min( n, m );
	}
	
	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( 1, 0 ) );
	CPtr<CDnnBlob> leftVectors;
	if( returnLeftVectors ) {
		leftVectors = CDnnBlob::CreateMatrix( *mathEngine, CT_Float, components, m );
	} else {
		leftVectors = CDnnBlob::CreateVector( *mathEngine, CT_Float, 1 );
	}
	CPtr<CDnnBlob> rightVectors;
	if( returnRightVectors ) {
		rightVectors = CDnnBlob::CreateMatrix( *mathEngine, CT_Float, components, n );
	} else {
		rightVectors = CDnnBlob::CreateVector( *mathEngine, CT_Float, 1 );
	}
	CPtr<CDnnBlob> singularValues = CDnnBlob::CreateVector( *mathEngine, CT_Float, components );
	CPtr<CDnnBlob> superb = CDnnBlob::CreateVector( *mathEngine, CT_Float, components );
	if( svdSolver == SVD_Full ) {
		CPtr<CDnnBlob> a = convertToBlob( *mathEngine, data );
		mathEngine->SingularValueDecomposition( a->GetData(), n, m, leftVectors->GetData(),
			singularValues->GetData(), rightVectors->GetData(), superb->GetData(), returnLeftVectors, returnRightVectors );
	} else {
		CPtr<CDnnBlob> columns;
		CPtr<CDnnBlob> rows;
		CPtr<CDnnBlob> values;
		CSparseMatrixDesc desc = getSparseMatrixDesc( *mathEngine, data, columns, rows, values );
		mathEngine->SparseSingularValueDecomposition( desc, m, n, leftVectors->GetData(),
			singularValues->GetData(), rightVectors->GetData(), superb->GetData(),
			components, returnLeftVectors );
		leftVectors = leftVectors->GetTransposed( 0, 1 );
	}

	if( returnLeftVectors ) {
		leftVectors_.SetSize( m * components );
		leftVectors->CopyTo( leftVectors_.GetPtr() );
	}
	singularValues_.SetSize( components );             
	singularValues->CopyTo( singularValues_.GetPtr() );
	if( returnRightVectors ) {
		rightVectors_.SetSize( components * n );
		rightVectors->CopyTo( rightVectors_.GetPtr() );
	}
	if( svdSolver == SVD_Sparse ) {
		reverseArrays( leftVectors_, singularValues_, rightVectors_, m, components, n,
			returnLeftVectors, returnRightVectors );
	}
}

} // namespace NeoML

CPca::CPca( const CParams& _params ) :
	params( _params )
{
	NeoAssert( ( params.ComponentsType == PCAC_None ) ||
		( ( params.ComponentsType == PCAC_Int ) && ( params.Components > 0 ) ) ||
		( ( params.ComponentsType == PCAC_Float ) && ( 0 < params.Components ) && ( params.Components < 1 ) ) );
	NeoAssert( ( params.SvdSolver == SVD_Full ) || ( params.ComponentsType != PCAC_Float ) );
}

// returns data - mean(data)
CSparseFloatMatrix subtractMean( const CFloatMatrixDesc& data )
{
	CSparseFloatVector mean( data.GetRow( 0 ) );
	for( int i = 1; i < data.Height; i++ ) {
		CSparseFloatVector cur( data.GetRow( i ) );
		mean += cur;
	}
	mean /= data.Height;
	CSparseFloatMatrix centeredMatrix( data.Width );
	for( int i = 0; i < data.Height; i++ ) {
		CSparseFloatVector cur( data.GetRow( i ) );
		cur -= mean;
		centeredMatrix.AddRow( cur );
	}
	return centeredMatrix;
}

// Flip signs of u columns and vt rows to obtain deterministic result
static void flipSVD( CArray<float>& u, CArray<float>& vt, int m, int k, int n )
{
	CArray<float> maxValues;
	maxValues.Add( 0, k );

	for( int row = 0; row < m; row++ ) {
		for( int col = 0; col < k; col++ ) {
			if( abs( u[row * k + col] ) > abs( maxValues[col] ) ) {
				maxValues[col] = u[row * k + col];
			}
		}
	}
	for( int col = 0; col < k; col++ ) {
		maxValues[col] = ( ( maxValues[col] >= 0 ) ? 1.f : -1.f );
	}

	for( int row = 0; row < m; row++ ) {
		for( int col = 0; col < k; col++ ) {
			u[row * k + col] *= maxValues[col];
		}
	}

	for( int row = 0; row < k; row++ ) {
		for( int col = 0; col < n; col++ ) {
			vt[row * n + col] *= maxValues[row];
		}
	}
}

// Convert CDnnBlob to CSparseFloatMatrix
static void convertToMatrix( CArray<float>& input, CSparseFloatMatrix& matrix, int matrixHeight, int components, int matrixWidth )
{
	CFloatVectorDesc row;
	row.Size = components;

	for( int i = 0; i < matrixHeight; i++ ) {
		row.Values = input.GetPtr() + i * matrixWidth;
		matrix.AddRow( row );
	}
}

void CPca::getComponentsNum( const CArray<float>& explainedVarianceRatio, int k )
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

void CPca::calculateVariance( const CFloatMatrixDesc& data, const CArray<float>& s, int total_components )
{
	int m = data.Height;
	int n = data.Width;

	// calculate explained_variance
	explainedVariance.SetSize( total_components );
	for( int i = 0; i < total_components; i++ ) {
		explainedVariance[i] = s[i] * s[i] / ( m - 1 );
	}

	// calculate total variance
	float totalVariance = 0;
	if( params.SvdSolver == SVD_Full ) {
		for( int i = 0; i < total_components; i++ ) {
			totalVariance += explainedVariance[i];
		}
	} else {
		CFeatureVarianceProblem problem( data );
		CArray<double> variance;
		CalcFeaturesVariance( problem, variance );
		for( int i = 0; i < variance.Size(); i++ ) {
			totalVariance += static_cast<float>( variance[i] );
		}
		totalVariance *= static_cast<float>( m * 1. / ( m - 1 ) );
	}

	// calculate explained_variance_ratio
	explainedVarianceRatio.SetSize( total_components );
	for( int i = 0; i < total_components; i++ ) {
		explainedVarianceRatio[i] = explainedVariance[i] / totalVariance;
	}

	// get actual number of components (still not defined in case of PCAC_Float)
	getComponentsNum( explainedVarianceRatio, total_components );

	// calculate noise_variance
	noiseVariance = totalVariance;
	for( int i = 0; i < components; i++ ) {
		noiseVariance -= explainedVariance[i];
	}
	noiseVariance /= max( 1, n - components );

	explainedVariance.SetSize( components );
	explainedVarianceRatio.SetSize( components );
}

void CPca::train( const CFloatMatrixDesc& data, bool isTransform )
{
	int m = data.Height;
	int n = data.Width;
	int k = min( n, m );

	// matrix = data - mean(data)
	CSparseFloatMatrix matrix = subtractMean( data );

	CArray<float> leftVectors;
	CArray<float> rightVectors;
	if( params.SvdSolver == SVD_Sparse ) {
		components = ( params.ComponentsType == PCAC_None ) ? k : static_cast<int>( params.Components );
		SingularValueDecomposition( matrix.GetDesc(), params.SvdSolver, leftVectors, singularValues, rightVectors,
			false, true, components );
	} else {
		SingularValueDecomposition( matrix.GetDesc(), params.SvdSolver, leftVectors, singularValues, rightVectors,
			true, true, 0 );
		// flip signs of u columns and vt rows to obtain deterministic result
		flipSVD( leftVectors, rightVectors, m, ( params.SvdSolver == SVD_Sparse ) ? components : k, n );
	}

	// calculate variance per component
	calculateVariance( data, singularValues, ( params.SvdSolver == SVD_Sparse ) ? components : k );

	singularValues.SetSize( components );
	componentsMatrix = CSparseFloatMatrix( components, n );
	convertToMatrix( rightVectors, componentsMatrix, components, n, n );

	if( isTransform ) {
		if( params.SvdSolver == SVD_Full ) {
			transformedMatrix = CSparseFloatMatrix( components, m, components * m );
			for( int row = 0; row < m; row++ ) {
				for( int col = 0; col < components; col++ ) {
					leftVectors[row * k + col] *= singularValues[col];
				}
			}
			convertToMatrix( leftVectors, transformedMatrix, m, components, k );
		} else {

		}
	}
}

CFloatMatrixDesc CPca::transform( const CFloatMatrixDesc& data )
{
	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( 1, 0 ) );
	CPtr<CDnnBlob> columns;
	CPtr<CDnnBlob> rows;
	CPtr<CDnnBlob> values;
	CSparseMatrixDesc desc = getSparseMatrixDesc( *mathEngine, data, columns, rows, values );
	mathEngine->MultiplySparseMatrixByTransposedMatrix( data->Height, data->Width,
		getSparseMatrixDesc( *mathEngine, data ) );
}

void CPca::Train( const CFloatMatrixDesc& data )
{
	train( data, false );
}

CFloatMatrixDesc CPca::Transform( const CFloatMatrixDesc& data )
{
	return ;
}

CFloatMatrixDesc CPca::TrainTransform( const CFloatMatrixDesc& data )
{
	train( data, true );
	return transformedMatrix.GetDesc();
}
