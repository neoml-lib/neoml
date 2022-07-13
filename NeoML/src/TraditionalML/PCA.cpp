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

// convert CFloatMatrixDesc to internal CSparseMatrixDesc
static CSparseMatrixDesc getSparseMatrixDesc( IMathEngine& mathEngine, const CFloatMatrixDesc& data,
	CPtr<CDnnBlob>& columns, CPtr<CDnnBlob>& rows, CPtr<CDnnBlob>& values )
{
	const int height = data.Height;
	const int width = data.Width;
	CSparseMatrixDesc desc;
	desc.ElementCount = ( data.Columns == nullptr ) ? height * width : data.PointerE[height - 1];
	columns = CDnnBlob::CreateVector( mathEngine, CT_Int, desc.ElementCount );
	rows = CDnnBlob::CreateVector( mathEngine, CT_Int, height + 1 );
	values = CDnnBlob::CreateVector( mathEngine, CT_Float, desc.ElementCount );
	if( data.Columns != nullptr ) {
		columns->CopyFrom( data.Columns );
		int* rowBuffer = rows->GetBuffer<int>( 0, height + 1, false );
		for( int i = 0; i < height; i++ ) {
			rowBuffer[i] = data.PointerB[i];
		}
		rowBuffer[height] = data.PointerE[height - 1];
		rows->ReleaseBuffer( rowBuffer, true );
	} else {
		desc.ElementCount = height * width;
		int* colBuffer = columns->GetBuffer<int>( 0, height * width, false );
		int* rowBuffer = rows->GetBuffer<int>( 0, height + 1, false );
		for( int i = 0; i < height; i++ ) {
			rowBuffer[i] = i * width;
			for( int j = 0; j < width; j++ ) {
				colBuffer[i * width + j] = j;
			}
		}
		rowBuffer[height] = height * width;
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
static void flipSVD( CArray<float>& u, CArray<float>& vt, int m, int k, int n )
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
	NeoAssert( components > 0 );
	NeoAssert( components <= min( data.Height, data.Width ) );
	CRandom rand( seed );
	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( 1, 0 ) );

	CPtr<CDnnBlob> columns;
	CPtr<CDnnBlob> rows;
	CPtr<CDnnBlob> values;
	CSparseMatrixDesc dataDesc = getSparseMatrixDesc( *mathEngine, data, columns, rows, values );

	const bool transpose = data.Height < data.Width;
	const int bigSide = max( data.Height, data.Width );
	const int smallSide = min( data.Height, data.Width );
	const int reducedSide = min( components + overSamples, smallSide );
	const int tempSize = bigSide * reducedSide;
	CObjectArray<CDnnBlob> tempIterationMatrix;
	tempIterationMatrix.Add( { CDnnBlob::CreateVector( *mathEngine, CT_Float, tempSize ), CDnnBlob::CreateVector( *mathEngine, CT_Float, tempSize ) } );
	float* buffer = tempIterationMatrix[0]->GetBuffer<float>( 0, tempSize, false );
	for( int i = 0; i < tempSize; i++ ) {
		buffer[i] = static_cast<float>( rand.Normal( 0., 1. ) );
	}
	mathEngine->ReleaseBuffer( tempIterationMatrix[0]->GetData(), buffer, true );

	if( transpose ) {
		for( int i = 0; i < iterationCount; i++ ) {
			mathEngine->MultiplyTransposedSparseMatrixByMatrix( data.Height, data.Width, reducedSide, dataDesc, tempIterationMatrix[0]->GetData(), tempIterationMatrix[1]->GetData() );
			mathEngine->MultiplySparseMatrixByMatrix( data.Height, data.Width, reducedSide, dataDesc, tempIterationMatrix[1]->GetData(), tempIterationMatrix[0]->GetData() );
		}
		mathEngine->MultiplyTransposedSparseMatrixByMatrix( data.Height, data.Width, reducedSide, dataDesc, tempIterationMatrix[0]->GetData(), tempIterationMatrix[1]->GetData() );
		CFloatHandle qHandle = tempIterationMatrix[0]->GetData();
		mathEngine->QRFactorization( bigSide, reducedSide, tempIterationMatrix[1]->GetData(), &qHandle, nullptr, true, true, false );
		mathEngine->MultiplyTransposedMatrixBySparseMatrix( bigSide, reducedSide, smallSide, tempIterationMatrix[0]->GetData(), dataDesc, tempIterationMatrix[1]->GetData(), true );
	} else {
		for( int i = 0; i < iterationCount; i++ ) {
			mathEngine->MultiplySparseMatrixByMatrix( data.Height, data.Width, reducedSide, dataDesc, tempIterationMatrix[0]->GetData(), tempIterationMatrix[1]->GetData() );
			mathEngine->MultiplyTransposedSparseMatrixByMatrix( data.Height, data.Width, reducedSide, dataDesc, tempIterationMatrix[1]->GetData(), tempIterationMatrix[0]->GetData() );
		}
		mathEngine->MultiplySparseMatrixByMatrix( data.Height, data.Width, reducedSide, dataDesc, tempIterationMatrix[0]->GetData(), tempIterationMatrix[1]->GetData() );
		CFloatHandle qHandle = tempIterationMatrix[0]->GetData();
		mathEngine->QRFactorization( bigSide, reducedSide, tempIterationMatrix[1]->GetData(), &qHandle, nullptr, true, true, false );
		mathEngine->MultiplyTransposedMatrixBySparseMatrix( bigSide, reducedSide, smallSide, tempIterationMatrix[0]->GetData(), dataDesc, tempIterationMatrix[1]->GetData(), false );
	}

	const bool calculateLeftVectors = transpose ? returnRightVectors : returnLeftVectors;
	const bool calculateRightVectors = transpose ? returnLeftVectors : returnRightVectors;
	CPtr<CDnnBlob> leftVectors;
	if( calculateLeftVectors ) {
		leftVectors = CDnnBlob::CreateMatrix( *mathEngine, CT_Float, reducedSide, reducedSide );
	} else {
		leftVectors = CDnnBlob::CreateVector( *mathEngine, CT_Float, 1 );
	}
	CPtr<CDnnBlob> rightVectors;
	if( calculateRightVectors ) {
		rightVectors = CDnnBlob::CreateMatrix( *mathEngine, CT_Float, reducedSide, smallSide );
	} else {
		rightVectors = CDnnBlob::CreateVector( *mathEngine, CT_Float, 1 );
	}
	CPtr<CDnnBlob> singularValues = CDnnBlob::CreateVector( *mathEngine, CT_Float, reducedSide );
	CPtr<CDnnBlob> superb = CDnnBlob::CreateVector( *mathEngine, CT_Float, reducedSide );

	mathEngine->SingularValueDecomposition( tempIterationMatrix[1]->GetData(), reducedSide, smallSide, leftVectors->GetData(),
		singularValues->GetData(), rightVectors->GetData(), superb->GetData(), calculateLeftVectors, calculateRightVectors );
	singularValues_.SetSize( components );
	singularValues->CopyTo( singularValues_.GetPtr(), components );

	if( transpose ) {
		if( returnLeftVectors ) {
			CPtr<CDnnBlob> transposed = rightVectors->GetTransposed( 0, 1 );
			leftVectors_.SetSize( components * smallSide );
			copyNarrowedBlobToArray( smallSide, transposed, reducedSide, leftVectors_, components );
		}

		if( returnRightVectors ) {
			mathEngine->MultiplyMatrixByMatrix( 1, tempIterationMatrix[0]->GetData(), bigSide, reducedSide, leftVectors->GetData(), reducedSide, tempIterationMatrix[1]->GetData(), bigSide * reducedSide );
			tempIterationMatrix[1]->ReinterpretDimensions( { bigSide, reducedSide } );
			tempIterationMatrix[0]->ReinterpretDimensions( { reducedSide, bigSide } );
			tempIterationMatrix[0]->TransposeFrom( tempIterationMatrix[1], CBlobDesc::MaxDimensions - 2, CBlobDesc::MaxDimensions - 1 );
			rightVectors_.SetSize( components * bigSide );
			tempIterationMatrix[0]->CopyTo( rightVectors_.GetPtr(), components * bigSide );
		}
	} else {
		if( returnLeftVectors ) {
			mathEngine->MultiplyMatrixByMatrix( 1, tempIterationMatrix[0]->GetData(), bigSide, reducedSide, leftVectors->GetData(), reducedSide, tempIterationMatrix[1]->GetData(), bigSide * reducedSide );
			copyNarrowedBlobToArray( data.Height, tempIterationMatrix[1], reducedSide, leftVectors_, components );
		}

		if( returnRightVectors ) {
			rightVectors_.SetSize( components * smallSide );
			rightVectors->CopyTo( rightVectors_.GetPtr(), components * smallSide );
		}
	}
}

void SingularValueDecomposition( const CFloatMatrixDesc& data,
	CArray<float>& leftVectors_, CArray<float>& singularValues_, CArray<float>& rightVectors_,
	bool returnLeftVectors, bool returnRightVectors, int resultComponents )
{
	const int height = data.Height;
	const int width = data.Width;
	NeoAssert( resultComponents >= 0 );
	NeoAssert( resultComponents <= min( height, width ) );

	const int components = min( height, width );
	resultComponents = ( resultComponents == 0 ) ? components : resultComponents;

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
	CPtr<CDnnBlob> a = convertToBlob( *mathEngine, data );
	mathEngine->SingularValueDecomposition( a->GetData(), height, width, leftVectors->GetData(),
		singularValues->GetData(), rightVectors->GetData(), superb->GetData(), returnLeftVectors, returnRightVectors );

	if( returnLeftVectors ) {
		copyNarrowedBlobToArray( height, leftVectors, components, leftVectors_, resultComponents );
	}
	singularValues_.SetSize( resultComponents );
	singularValues->CopyTo( singularValues_.GetPtr(), resultComponents );
	if( returnRightVectors ) {
		rightVectors_.SetSize( resultComponents * width );
		rightVectors->CopyTo( rightVectors_.GetPtr(), resultComponents * width );
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
	const int height = data.Height;
	const int width = data.Width;

	// calculate explained_variance
	explainedVariance.SetSize( total_components );
	for( int i = 0; i < total_components; i++ ) {
		explainedVariance[i] = s[i] * s[i] / ( height - 1 );
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
		totalVariance *= static_cast<float>( height * 1. / ( height - 1 ) );
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
	noiseVariance /= max( 1, width - components );

	explainedVariance.SetSize( components );
	explainedVarianceRatio.SetSize( components );
}

void CPca::train( const CFloatMatrixDesc& data, bool isTransform )
{
	const int height = data.Height;
	const int width = data.Width;
	const int k = min( height, width );

	// matrix = data - mean(data)
	CSparseFloatMatrix matrix = subtractMean( data, meanVector, true );

	CArray<float> leftVectors;
	if( params.SvdSolver == SVD_Full ) {
		components = k;
		SingularValueDecomposition( matrix.GetDesc(), leftVectors, singularValues, componentsMatrix,
			true, true );
	} else if( params.SvdSolver == SVD_Randomized ) {
		components = ( params.ComponentsType == PCAC_None ) ? k : static_cast<int>( params.Components );
		RandomizedSingularValueDecomposition( matrix.GetDesc(), leftVectors, singularValues, componentsMatrix,
			true, true, components );
	} else {
		NeoAssert( false );
	}

	// flip signs of u columns and vt rows to obtain deterministic result
	flipSVD( leftVectors, componentsMatrix, height, components, width );

	// calculate variance per component
	calculateVariance( data, singularValues, components );

	singularValues.SetSize( components );
	componentsMatrix.SetSize( components * width );

	if( isTransform ) {
		if( params.SvdSolver == SVD_Full ) {
			transformedMatrix = CSparseFloatMatrix( components, height );
			for( int row = 0; row < height; row++ ) {
				for( int col = 0; col < components; col++ ) {
					leftVectors[row * k + col] *= singularValues[col];
				}
			}
			convertToMatrix( leftVectors.GetPtr(), transformedMatrix, height, components, k );
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
