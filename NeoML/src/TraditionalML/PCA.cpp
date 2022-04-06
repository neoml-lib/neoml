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
#include <NeoML/Random.h>

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

// convert CFloatMatrix to CDnnBlob
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

// change sorting order of singular vectors
static void reverseArrays( CArray<float>& leftVectors, CArray<float>& singularVectors, CArray<float>& rightVectors,
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

// convert CFloatMatrixDesc to internal CSparseMatrixDesc
static CSparseMatrixDesc getSparseMatrixDesc( IMathEngine& mathEngine, const CFloatMatrixDesc& data,
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

// copy blob with reduced width
static void copyNarrowedBlobToArray( int height, CPtr<CDnnBlob> blob, int oldWidth, CArray<float>& array, int newWidth )
{
	float* buffer = blob->GetBuffer<float>( 0, blob->GetDataSize(), true );
	array.Empty();
	if( newWidth < oldWidth ) {
		array.SetBufferSize( height * newWidth );
		for( int i = 0; i < height; i++ ) {
			for( int j = 0; j < newWidth; j++ ) {
				array.Add( buffer[j] );
			}
			buffer += oldWidth;
		}
	} else {
		array.SetSize( height * newWidth );
		blob->CopyTo( array.GetPtr(), height * newWidth );
	}
}

// returns data - mean(data)
static CSparseFloatMatrix subtractMean( const CFloatMatrixDesc& data, CSparseFloatVector& mean, bool calculateMean )
{
	if( calculateMean ) {
		mean = CSparseFloatVector( data.GetRow( 0 ) );
		for( int i = 1; i < data.Height; i++ ) {
			mean += data.GetRow( i );
		}
		mean /= data.Height;
	}
	CSparseFloatMatrix centeredMatrix( data.Width );
	for( int i = 0; i < data.Height; i++ ) {
		CSparseFloatVector cur( data.GetRow( i ) );
		cur -= mean;
		centeredMatrix.AddRow( cur );
	}
	return centeredMatrix;
}

// flip signs of u columns and vt rows to obtain deterministic result
static void flipSVD( CArray<float>& u, CArray<float>& vt, int m, int k, int n, bool hasLeft )
{
	CArray<float> maxValues;
	maxValues.Add( 0, k );

	for( int row = 0; row < k; row++ ) {
		for( int col = 0; col < n; col++ ) {
			if( abs( vt[row * k + col] ) > abs( maxValues[row] ) ) {
				maxValues[row] = vt[row * k + col];
			}
		}
	}
	for( int col = 0; col < k; col++ ) {
		maxValues[col] = ( ( maxValues[col] >= 0 ) ? 1.f : -1.f );
	}

	if( hasLeft ) {
		for( int row = 0; row < m; row++ ) {
			for( int col = 0; col < k; col++ ) {
				u[row * k + col] *= maxValues[col];
			}
		}
	}

	for( int row = 0; row < k; row++ ) {
		for( int col = 0; col < n; col++ ) {
			vt[row * n + col] *= maxValues[row];
		}
	}
}

// convert CDnnBlob to CSparseFloatMatrix
static void convertToMatrix( float* input, CSparseFloatMatrix& matrix, int matrixHeight, int components, int matrixWidth )
{
	CFloatVectorDesc row;
	row.Size = components;

	for( int i = 0; i < matrixHeight; i++ ) {
		row.Values = input + i * matrixWidth;
		matrix.AddRow( row );
	}
}

// transform matrix using principal components
static CSparseFloatMatrix transform( const CFloatMatrixDesc& data, const CArray<float>& componentsMatrix, int components )
{
	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( 1, 0 ) );
	CPtr<CDnnBlob> columns;
	CPtr<CDnnBlob> rows;
	CPtr<CDnnBlob> values;
	CSparseMatrixDesc desc = getSparseMatrixDesc( *mathEngine, data, columns, rows, values );
	CPtr<CDnnBlob> rightVectors = CDnnBlob::CreateVector( *mathEngine, CT_Float, data.Width * components );
	rightVectors->CopyFrom( componentsMatrix.GetPtr() );
	CPtr<CDnnBlob> transformed = CDnnBlob::CreateVector( *mathEngine, CT_Float, data.Height * components );
	mathEngine->MultiplySparseMatrixByTransposedMatrix( data.Height, data.Width, components,
		desc, rightVectors->GetData(), transformed->GetData() );

	CSparseFloatMatrix transformedResult = CSparseFloatMatrix( components, data.Height );
	convertToMatrix( transformed->GetBuffer<float>( 0, data.Height * components, true ),
		transformedResult, data.Height, components, components );
	return transformedResult;
}

namespace NeoML {

void RandomizedSingularValueDecomposition( const CFloatMatrixDesc& data,
	CArray<float>& leftVectors_, CArray<float>& singularValues_, CArray<float>& rightVectors_,
	bool returnLeftVectors, bool returnRightVectors, int components,
	int iterationCount, int overSamples, int seed )
{
	CRandom rand( seed );
	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( 1, 0 ) );

	CPtr<CDnnBlob> columns;
	CPtr<CDnnBlob> rows;
	CPtr<CDnnBlob> values;
	CSparseMatrixDesc dataDesc = getSparseMatrixDesc( *mathEngine, data, columns, rows, values );

	int wideSide = max( data.Height, data.Width );
	int thinSide = min( components + overSamples, min( data.Height, data.Width ) );
	int tempSize = wideSide * thinSide;
	CObjectArray<CDnnBlob> tempIterationMatrix;
	tempIterationMatrix.Add( { CDnnBlob::CreateVector( *mathEngine, CT_Float, tempSize ), CDnnBlob::CreateVector( *mathEngine, CT_Float, tempSize ) } );
	float* buffer = tempIterationMatrix[0]->GetBuffer<float>( 0, tempSize, false );
	for( int i = 0; i < tempSize; i++ ) {
		buffer[i] = rand.Normal( 0., 1. );
	}
	mathEngine->ReleaseBuffer( tempIterationMatrix[0]->GetData(), buffer, true );

	for( int i = 0; i < iterationCount; i++ ) {
		mathEngine->MultiplySparseMatrixByMatrix( data.Height, data.Width, thinSide, dataDesc, tempIterationMatrix[0]->GetData(), tempIterationMatrix[1]->GetData() );
		mathEngine->MultiplyTransposedSparseMatrixByMatrix( data.Height, data.Width, thinSide, dataDesc, tempIterationMatrix[1]->GetData(), tempIterationMatrix[0]->GetData() );
	}
	mathEngine->MultiplySparseMatrixByMatrix( data.Height, data.Width, thinSide, dataDesc, tempIterationMatrix[0]->GetData(), tempIterationMatrix[1]->GetData() );
	CFloatHandle qHandle = tempIterationMatrix[0]->GetData();
	mathEngine->QRFactorization( data.Height, thinSide, tempIterationMatrix[1]->GetData(), &qHandle, nullptr, true, true, false );
	mathEngine->MultiplyTransposedMatrixBySparseMatrix( data.Height, thinSide, data.Width, tempIterationMatrix[0]->GetData(), dataDesc, tempIterationMatrix[1]->GetData() );

	CPtr<CDnnBlob> leftVectors;
	if( returnLeftVectors ) {
		leftVectors = CDnnBlob::CreateMatrix( *mathEngine, CT_Float, thinSide, thinSide );
	} else {
		leftVectors = CDnnBlob::CreateVector( *mathEngine, CT_Float, 1 );
	}
	CPtr<CDnnBlob> rightVectors;
	if( returnRightVectors ) {
		rightVectors = CDnnBlob::CreateMatrix( *mathEngine, CT_Float, thinSide, data.Width );
	} else {
		rightVectors = CDnnBlob::CreateVector( *mathEngine, CT_Float, 1 );
	}
	CPtr<CDnnBlob> singularValues = CDnnBlob::CreateVector( *mathEngine, CT_Float, thinSide );
	CPtr<CDnnBlob> superb = CDnnBlob::CreateVector( *mathEngine, CT_Float, thinSide );

	mathEngine->SingularValueDecomposition( tempIterationMatrix[1]->GetData(), thinSide, data.Width, leftVectors->GetData(),
		singularValues->GetData(), rightVectors->GetData(), superb->GetData(), returnLeftVectors, returnRightVectors );
	if( returnLeftVectors ) {
		mathEngine->MultiplyMatrixByMatrix( 1, tempIterationMatrix[0]->GetData(), data.Height, thinSide, leftVectors->GetData(), thinSide, tempIterationMatrix[1]->GetData(), data.Height * thinSide );
		copyNarrowedBlobToArray( data.Height, tempIterationMatrix[1], thinSide, leftVectors_, components );
	}
	singularValues_.SetSize( components );
	singularValues->CopyTo( singularValues_.GetPtr(), components );
	if( returnRightVectors ) {
		rightVectors_.SetSize( components * data.Width );
		rightVectors->CopyTo( rightVectors_.GetPtr(), components * data.Width );
	}
}

void SingularValueDecomposition( const CFloatMatrixDesc& data, const TSvd& svdSolver,
	CArray<float>& leftVectors_, CArray<float>& singularValues_, CArray<float>& rightVectors_,
	bool returnLeftVectors, bool returnRightVectors, int components_ )
{
	if( svdSolver == SVD_Sparse && ( returnLeftVectors == returnRightVectors ) ) {
		// exactly one flag must be true
		NeoAssert( false );
	}
	int width = data.Width;
	int height = data.Height;

	int components = components_;
	if( svdSolver == SVD_Full || components == 0 ) {
		components = min( height, width );
	}

	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( 1, 0 ) );
	CPtr<CDnnBlob> leftVectors;
	if( returnLeftVectors ) {
		leftVectors = CDnnBlob::CreateMatrix( *mathEngine, CT_Float, components, height );
	} else {
		leftVectors = CDnnBlob::CreateVector( *mathEngine, CT_Float, 1 );
	}
	CPtr<CDnnBlob> rightVectors;
	if( returnRightVectors ) {
		rightVectors = CDnnBlob::CreateMatrix( *mathEngine, CT_Float, components, width );
	} else {
		rightVectors = CDnnBlob::CreateVector( *mathEngine, CT_Float, 1 );
	}
	CPtr<CDnnBlob> singularValues = CDnnBlob::CreateVector( *mathEngine, CT_Float, components );
	CPtr<CDnnBlob> superb = CDnnBlob::CreateVector( *mathEngine, CT_Float, components );
	if( svdSolver == SVD_Full ) {
		CPtr<CDnnBlob> a = convertToBlob( *mathEngine, data );
		mathEngine->SingularValueDecomposition( a->GetData(), height, width, leftVectors->GetData(),
			singularValues->GetData(), rightVectors->GetData(), superb->GetData(), returnLeftVectors, returnRightVectors );
	} else {
		CPtr<CDnnBlob> columns;
		CPtr<CDnnBlob> rows;
		CPtr<CDnnBlob> values;
		CSparseMatrixDesc desc = getSparseMatrixDesc( *mathEngine, data, columns, rows, values );
		mathEngine->SparseSingularValueDecomposition( desc, height, width, leftVectors->GetData(),
			singularValues->GetData(), rightVectors->GetData(), superb->GetData(),
			components, returnLeftVectors );
		leftVectors = leftVectors->GetTransposed( 0, 1 );
	}

	if( returnLeftVectors ) {
		copyNarrowedBlobToArray( height, leftVectors, components, leftVectors_, components_ );
	}
	singularValues_.SetSize( components_ );
	singularValues->CopyTo( singularValues_.GetPtr(), components_ );
	if( returnRightVectors ) {
		rightVectors_.SetSize( components_ * width );
		rightVectors->CopyTo( rightVectors_.GetPtr(), components_ * width );
	}
	if( svdSolver == SVD_Sparse ) {
		reverseArrays( leftVectors_, singularValues_, rightVectors_, height, components, width,
			returnLeftVectors, returnRightVectors );
	}
}

}

CPca::CPca( const CParams& _params ) :
	params( _params )
{
	NeoAssert( ( params.ComponentsType == PCAC_None ) ||
		( ( params.ComponentsType == PCAC_Int ) && ( params.Components > 0 ) ) ||
		( ( params.ComponentsType == PCAC_Float ) && ( 0 < params.Components ) && ( params.Components < 1 ) ) );
	NeoAssert( ( params.SvdSolver == SVD_Full ) || ( params.ComponentsType != PCAC_Float ) );
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
	CSparseFloatMatrix matrix = subtractMean( data, meanVector, true );

	CArray<float> leftVectors;
	if( params.SvdSolver == SVD_Sparse ) {
		components = ( params.ComponentsType == PCAC_None ) ? k : static_cast<int>( params.Components );
		SingularValueDecomposition( matrix.GetDesc(), params.SvdSolver, leftVectors, singularValues, componentsMatrix,
			false, true, components );
	} else {
		SingularValueDecomposition( matrix.GetDesc(), params.SvdSolver, leftVectors, singularValues, componentsMatrix,
			true, true, 0 );
	}

	// flip signs of u columns and vt rows to obtain deterministic result
	flipSVD( leftVectors, componentsMatrix, m, ( params.SvdSolver == SVD_Sparse ) ? components : k, n, params.SvdSolver == SVD_Full );

	// calculate variance per component
	calculateVariance( data, singularValues, ( params.SvdSolver == SVD_Sparse ) ? components : k );

	singularValues.SetSize( components );
	componentsMatrix.SetSize( components * n );

	if( isTransform ) {
		if( params.SvdSolver == SVD_Full ) {
			transformedMatrix = CSparseFloatMatrix( components, m );
			for( int row = 0; row < m; row++ ) {
				for( int col = 0; col < components; col++ ) {
					leftVectors[row * k + col] *= singularValues[col];
				}
			}
			convertToMatrix( leftVectors.GetPtr(), transformedMatrix, m, components, k );
		} else {
			transformedMatrix = transform( matrix.GetDesc(), componentsMatrix, components );
		}
	}
}

CSparseFloatMatrix CPca::GetComponents()
{
	const int componentWidth = componentsMatrix.Size() / components;
	CSparseFloatMatrix matrix = CSparseFloatMatrix( components, componentWidth );
	convertToMatrix( componentsMatrix.GetPtr(), matrix, components, componentWidth, componentWidth );
	return matrix;
}

void CPca::Train( const CFloatMatrixDesc& data )
{
	train( data, false );
}

CSparseFloatMatrixDesc CPca::TrainTransform( const CFloatMatrixDesc& data )
{
	train( data, true );
	return transformedMatrix.GetDesc();
}

CSparseFloatMatrixDesc CPca::Transform( const CFloatMatrixDesc& data )
{
	// matrix = data - mean(data)
	CSparseFloatMatrix matrix = subtractMean( data, meanVector, false );
	transformedMatrix = transform( matrix.GetDesc(), componentsMatrix, components );
	return transformedMatrix.GetDesc();
}

void CPca::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );
	if( archive.IsStoring() ) {
		archive << params.Components;
		archive << static_cast<int>( params.ComponentsType );
		archive << static_cast<int>( params.SvdSolver );
		archive << components << noiseVariance;
		archive << singularValues;
		archive << explainedVariance << explainedVarianceRatio;
		archive << componentsMatrix;
		archive << meanVector;
	} else if( archive.IsLoading() ) {
		CParams serializedParams;
		archive >> serializedParams.Components;
		int x;
		archive >> x;
		serializedParams.ComponentsType = static_cast<TComponents>( x );
		archive >> x;
		serializedParams.SvdSolver = static_cast<TSvd>( x );
		params = serializedParams;
		archive >> components >> noiseVariance;
		archive >> singularValues;
		archive >> explainedVariance >> explainedVarianceRatio;
		archive >> componentsMatrix;
		archive >> meanVector;
	} else {
		NeoAssert( false );
	}
}
