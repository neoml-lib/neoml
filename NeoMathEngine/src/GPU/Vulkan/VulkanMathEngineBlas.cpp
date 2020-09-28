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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_VULKAN

#include <VulkanMathEngine.h>
#include <VulkanShader.h>
#include <MathEngineCommon.h>
#include <VulkanShader.h>
#include <VulkanDll.h>

namespace NeoML {

// Include the shader code
#include <shaders/generated/Transpose.h>
#include <shaders/generated/Matrix2InterleavedAdreno.h>
#include <shaders/generated/MultiplyMatrixInterleavedAdreno.h>
#include <shaders/generated/MultiplyMatrixInterleavedBoardersAdreno.h>
#include <shaders/generated/AddVectorToMatrixRowsAdreno.h>
#include <shaders/generated/SetVectorToMatrixRowsAdreno.h>
#include <shaders/generated/SetVectorToMatrixRows.h>
#include <shaders/generated/RowMultiplyMatrixByMatrix.h>
#include <shaders/generated/SumMatrixRows.h>
#include <shaders/generated/SumMatrixColumns.h>
#include <shaders/generated/MultiplyMatrixByDiagMatrixAdreno.h>
#include <shaders/generated/MultiplyMatrixByDiagMatrix.h>
#include <shaders/generated/MultiplyDiagMatrixByMatrixAdreno.h>
#include <shaders/generated/MultiplyDiagMatrixByMatrix.h>
#include <shaders/generated/MultiplyDiagMatrixByMatrixAndAdd.h>
#include <shaders/generated/MultiplyMatrixByMatrix.h>
#include <shaders/generated/MultiplySparseMatrixByTransposedMatrix.h>
#include <shaders/generated/MultiplyTransposedMatrixBySparseMatrix.h>
#include <shaders/generated/BatchMultiplyMatrixByMatrixBorders.h>
#include <shaders/generated/BatchMultiplyMatrixByTransposedMatrix.h>
#include <shaders/generated/BatchMultiplyMatrixByTransposedMatrixBorders.h>
#include <shaders/generated/BatchMultiplyTransposedMatrixByMatrix.h>
#include <shaders/generated/BatchMultiplyTransposedMatrixByMatrixBorders.h>
#include <shaders/generated/BatchInitAddMultiplyMatrixByTransposedMatrix.h>
#include <shaders/generated/BatchInitMultiplyMatrixByTransposedMatrixBorders.h>
#include <shaders/generated/SetVectorToMatrixElements.h>
#include <shaders/generated/FindMaxValueInRows.h>
#include <shaders/generated/FindMaxValueInRowsNoIndices.h>
#include <shaders/generated/FindMaxValueInColumns.h>
#include <shaders/generated/LookupAndSum.h>
#include <shaders/generated/AddMatrixElementsToVector.h>
#include <shaders/generated/AddMatrixElementsToVectorEx.h>
#include <shaders/generated/AddVectorToMatrixColumnsInt.h>
#include <shaders/generated/AddVectorToMatrixColumnsFloatAdreno.h>
#include <shaders/generated/AddVectorToMatrixColumnsFloat.h>
#include <shaders/generated/BatchAddVectorToMatrixRows.h>
#include <shaders/generated/EnumBinarizationFloat.h>
#include <shaders/generated/EnumBinarizationInt.h>
#include <shaders/generated/BitSetBinarization.h>
#include <shaders/generated/MatrixLogSumExpByRows.h>
#include <shaders/generated/MatrixSoftmaxByRows.h>
#include <shaders/generated/MatrixSoftmaxByColumns.h>
#include <shaders/generated/MatrixSpreadRowsFloat.h>
#include <shaders/generated/MatrixSpreadRowsFloatAdd.h>
#include <shaders/generated/MatrixSpreadRowsInt.h>
#include <shaders/generated/FindMaxValueInColumnsNoIndices.h>
#include <shaders/generated/FindMinValueInColumns.h>
#include <shaders/generated/VectorMultichannelLookupAndCopyFloat.h>
#include <shaders/generated/VectorMultichannelCopyFloat.h>
#include <shaders/generated/VectorMultichannelLookupAndCopyInt.h>
#include <shaders/generated/VectorMultichannelCopyInt.h>

//------------------------------------------------------------------------------------------------------------

inline int Floor( int val, int discret )
{
	assert( discret > 0 );
	if( val > 0 ) {
		return val / discret;
	}
	return ( val - discret + 1 ) / discret;
}

inline int FloorTo( int val, int discret )
{
	return Floor( val, discret ) * discret;
}

inline int Ceil( int val, int discret )
{
	assert( discret > 0 );
	if( val > 0 ) {
		return ( val + discret - 1 ) / discret;
	}
	return val / discret;
}

//------------------------------------------------------------------------------------------------------------

void CVulkanMathEngine::TransposeMatrix( int batchSize, const CConstFloatHandle& firstHandle,
	int height, int medium, int width, int channels, const CFloatHandle& resultHandle, int /*resultBufferSize*/ )
{
	int vectorSize = batchSize * height * medium * width * channels;
	CMemoryHandle bufs[2] = { firstHandle, resultHandle };
	size_t sizes[2] = { vectorSize * sizeof(float), vectorSize * sizeof(float) };

	PARAM_STRUCT(Transpose) param = { height, medium, width, channels, batchSize };

	runVectorShader( shaderLoader->GET_SHADER_DATA(Transpose, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, vectorSize );
}

void CVulkanMathEngine::TransposeMatrix( int batchSize, const CConstIntHandle& firstHandle,
	int height, int medium, int width, int channels, const CIntHandle& resultHandle, int /*resultBufferSize*/ )
{
	int vectorSize = batchSize * height * medium * width * channels;
	CMemoryHandle bufs[2] = { firstHandle, resultHandle };
	size_t sizes[2] = { vectorSize * sizeof(int), vectorSize * sizeof(int) };

	PARAM_STRUCT(Transpose) param = { height, medium, width, channels, batchSize };

	runVectorShader( shaderLoader->GET_SHADER_DATA(Transpose, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, vectorSize );
}

void CVulkanMathEngine::MultiplyMatrixByTransposedMatrix( const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
	const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize )
{
	if( device->Type == VDT_Adreno ) {
		batchMultiplyMatrixByMatrixAdreno( false, 1, firstHandle,
			firstHeight, firstWidth, firstRowSize, false, secondHandle, secondHeight, firstWidth,
			secondRowSize, true, resultHandle, resultRowSize, resultBufferSize );
	} else {
		batchMultiplyMatrixByTransposedMatrix( false, 1, firstHandle, firstHeight, firstWidth, firstRowSize,
			secondHandle, secondHeight, secondRowSize, resultHandle, resultRowSize, resultBufferSize );
	}
}

void CVulkanMathEngine::MultiplyMatrixByTransposedMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, const CConstFloatHandle& secondHandle, int secondHeight, const CFloatHandle& resultHandle, int resultBufferSize)
{
	if( device->Type == VDT_Adreno ) {
		batchMultiplyMatrixByMatrixAdreno( false, batchSize, firstHandle,
			firstHeight, firstWidth, firstWidth, false, secondHandle, secondHeight, firstWidth,
			firstWidth, true, resultHandle, secondHeight, resultBufferSize );
	} else {
		batchMultiplyMatrixByTransposedMatrix( false, batchSize, firstHandle, firstHeight, firstWidth, firstWidth,
			secondHandle, secondHeight, firstWidth, resultHandle, secondHeight, resultBufferSize );
	}
}

void CVulkanMathEngine::MultiplySparseMatrixByTransposedMatrix( int firstHeight, int firstWidth, int secondHeight,
	const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstDesc.Rows.GetMathEngine() == this );
	ASSERT_EXPR( firstDesc.Columns.GetMathEngine() == this );
	ASSERT_EXPR( firstDesc.Values.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	CMemoryHandle bufs[5] = { firstDesc.Rows, firstDesc.Columns, firstDesc.Values, secondHandle, resultHandle };
	size_t sizes[5] = { ( firstHeight + 1 ) * sizeof( int ), firstDesc.ElementCount * sizeof( int ), firstDesc.ElementCount * sizeof( int ),
		secondHeight * firstWidth * sizeof( float ), firstHeight * secondHeight * sizeof( float ) };

	PARAM_STRUCT( MultiplySparseMatrixByTransposedMatrix ) param = { firstHeight, firstWidth, secondHeight };

	runShader( shaderLoader->GET_SHADER_DATA( MultiplySparseMatrixByTransposedMatrix, true, 0, 0, 5 ),
		&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 5, secondHeight, firstHeight, 1 );
}

void CVulkanMathEngine::MultiplyTransposedMatrixBySparseMatrixAndAdd( int firstHeight, int firstWidth, int secondWidth,
	const CConstFloatHandle& firstHandle, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondDesc.Rows.GetMathEngine() == this );
	ASSERT_EXPR( secondDesc.Columns.GetMathEngine() == this );
	ASSERT_EXPR( secondDesc.Values.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	CMemoryHandle bufs[5] = { firstHandle, secondDesc.Rows, secondDesc.Columns, secondDesc.Values, resultHandle };
	size_t sizes[5] = { firstHeight * firstWidth * sizeof( float ), ( firstHeight + 1 ) * sizeof( int ), secondDesc.ElementCount * sizeof( int ),
		secondDesc.ElementCount * sizeof( float ), firstWidth * secondWidth * sizeof( float ) };

	PARAM_STRUCT( MultiplyTransposedMatrixBySparseMatrix ) param = { firstHeight, firstWidth, secondWidth };

	runVectorShader( shaderLoader->GET_SHADER_DATA( MultiplyTransposedMatrixBySparseMatrix, true, 0, 0, 5 ),
		&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 5, firstWidth );
}

void CVulkanMathEngine::MultiplyTransposedMatrixByMatrixAndAdd( const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondWidth, int secondRowSize,
	const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize )
{
	if( device->Type == VDT_Adreno ) {
		batchMultiplyMatrixByMatrixAdreno( true, 1, firstHandle, firstHeight, firstWidth, firstRowSize, true,
			secondHandle, firstHeight, secondWidth, secondRowSize, false, resultHandle, resultRowSize, resultBufferSize );
	} else {
		batchMultiplyTransposedMatrixByMatrix( true, 1, firstHandle, firstHeight, firstWidth, firstRowSize,
			secondHandle, secondWidth, secondRowSize, resultHandle, resultRowSize, resultBufferSize );
	}
}

void CVulkanMathEngine::MultiplyTransposedMatrixByMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int resultBufferSize)
{
	if( device->Type == VDT_Adreno ) {
		batchMultiplyMatrixByMatrixAdreno( false, batchSize, firstHandle, firstHeight, firstWidth, firstWidth, true,
			secondHandle, firstHeight, secondWidth, secondWidth, false, resultHandle, secondWidth, resultBufferSize );
	} else {
		batchMultiplyTransposedMatrixByMatrix( false, batchSize, firstHandle, firstHeight, firstWidth, firstWidth,
			secondHandle, secondWidth, secondWidth, resultHandle, secondWidth, resultBufferSize );
	}
}

void CVulkanMathEngine::MultiplyDiagMatrixByMatrix( const CConstFloatHandle& firstHandle, int firstSize,
	const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( resultBufferSize >= firstSize * secondWidth );

	if( device->Type == VDT_Adreno ) {
		const CVulkanImage* samplers[] = { &batchVectorToImage( 1, firstHandle, firstSize, TVI_MatrixLeft ) };

		CMemoryHandle bufs[2] = { secondHandle, resultHandle };
		size_t sizes[2] = { firstSize * secondWidth * sizeof( float ), firstSize * secondWidth * sizeof( float ) };

		PARAM_STRUCT( MultiplyDiagMatrixByMatrixAdreno ) param = { firstSize, secondWidth };

		runShader( shaderLoader->GET_SHADER_DATA( MultiplyDiagMatrixByMatrixAdreno, true, 0, 1, 2 ),
			&param, sizeof( param ), 0, 0, samplers, 1, bufs, sizes, 2, Ceil( firstSize, 4 ), secondWidth, 1 );
	} else {
		const size_t matrixSize = firstSize * secondWidth;
		CMemoryHandle bufs[3] = { firstHandle, secondHandle, resultHandle };
		size_t sizes[3] = { firstSize * sizeof( float ), matrixSize * sizeof( float ), matrixSize * sizeof( float ) };

		PARAM_STRUCT( MultiplyDiagMatrixByMatrix ) param = { firstSize, secondWidth };

		runShader( shaderLoader->GET_SHADER_DATA( MultiplyDiagMatrixByMatrix, true, 0, 0, 3 ),
			&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 3, Ceil( firstSize, 4 ), secondWidth, 1 );
	}
}

void CVulkanMathEngine::Multiply1DiagMatrixByMatrix( int, const CConstFloatHandle&, int, const CConstFloatHandle&,
	int, const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::MultiplyMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int resultBufferSize )
{
	if( device->Type == VDT_Adreno ) {
		batchMultiplyMatrixByMatrixAdreno( false, batchSize, firstHandle, firstHeight, firstWidth, firstWidth, false,
			secondHandle, firstWidth, secondWidth, secondWidth, false,
			resultHandle, secondWidth, resultBufferSize );
	} else {
		multiplyMatrixByMatrix( false, batchSize, firstHandle, firstHeight, firstWidth, firstWidth,
			secondHandle, secondWidth, secondWidth, resultHandle, secondWidth, resultBufferSize );
	}
}

void CVulkanMathEngine::addVectorToMatrixRowsAdreno( int /*batchSize*/,
	const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle )
{
	const size_t matrixSize = matrixHeight * matrixWidth * sizeof( float );
	const CVulkanImage* samplers[] = { &batchVectorToImage( 1, vectorHandle, matrixWidth, TVI_FreeTerm ) };

	CMemoryHandle bufs[2] = { matrixHandle, resultHandle };
	size_t sizes[2] = { matrixSize, matrixSize };

	PARAM_STRUCT( AddVectorToMatrixRowsAdreno ) param = { matrixHeight, matrixWidth };

	runShader( shaderLoader->GET_SHADER_DATA( AddVectorToMatrixRowsAdreno, true, 0, 1, 2 ),
		&param, sizeof( param ), 0, 0, samplers, 1, bufs, sizes, 2, Ceil( matrixWidth, 4 ), matrixHeight, 1 );
}

void CVulkanMathEngine::AddVectorToMatrixRows( int batchSize,
	const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle )
{
	const size_t matrixSize = matrixHeight * matrixWidth * sizeof(float);
	const size_t batchMatrixSize = matrixSize * batchSize;
	CMemoryHandle bufs[3] = { matrixHandle, vectorHandle, resultHandle };
	size_t sizes[3] = { batchMatrixSize, batchSize * matrixWidth * sizeof(float), batchMatrixSize };

	PARAM_STRUCT( BatchAddVectorToMatrixRows ) param = { batchSize, matrixHeight, matrixWidth };

	runShader( shaderLoader->GET_SHADER_DATA( BatchAddVectorToMatrixRows, true, 0, 0, 3 ),
		&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 3, matrixWidth, Ceil( matrixHeight, 4 ), batchSize );
}

void CVulkanMathEngine::AddVectorToMatrixColumns( const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle )
{
	if( device->Type == VDT_Adreno ) {
		const CVulkanImage* samplers[] = { &batchVectorToImage( 1, vectorHandle, matrixHeight, TVI_FreeTerm ) };

		CMemoryHandle bufs[2] = { matrixHandle, resultHandle };
		size_t sizes[2] = { matrixHeight * matrixWidth * sizeof( float ),
			matrixHeight * matrixWidth * sizeof( float ) };

		PARAM_STRUCT( AddVectorToMatrixColumnsFloatAdreno ) param = { matrixHeight, matrixWidth };

		runShader( shaderLoader->GET_SHADER_DATA( AddVectorToMatrixColumnsFloatAdreno, true, 0, 1, 2 ),
			&param, sizeof( param ), 0, 0, samplers, 1, bufs, sizes, 2, matrixWidth, Ceil( matrixHeight, 4 ), 1 );
	} else {
		CMemoryHandle bufs[3] = { matrixHandle, vectorHandle, resultHandle };
		size_t sizes[3] = { matrixHeight * matrixWidth * sizeof( int ), matrixHeight * sizeof( int ),
			matrixHeight * matrixWidth * sizeof( int ) };

		PARAM_STRUCT( AddVectorToMatrixColumnsFloat ) param = { matrixHeight, matrixWidth };

		runShader( shaderLoader->GET_SHADER_DATA( AddVectorToMatrixColumnsFloat, true, 0, 0, 3 ),
			&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 3, Ceil( matrixWidth, 4 ), matrixHeight, 1 );
	}
}

void CVulkanMathEngine::AddVectorToMatrixColumns( const CConstIntHandle& matrixHandle, const CIntHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstIntHandle& vectorHandle )
{
	// At the moment the image works only with float, so keep the vectorHandle as a buffer
	CMemoryHandle bufs[3] = { matrixHandle, vectorHandle, resultHandle };
	size_t sizes[3] = { matrixHeight * matrixWidth * sizeof(int), matrixHeight * sizeof(int),
		matrixHeight * matrixWidth * sizeof(int) };

	PARAM_STRUCT(AddVectorToMatrixColumnsInt) param = { matrixHeight, matrixWidth };

	runShader( shaderLoader->GET_SHADER_DATA(AddVectorToMatrixColumnsInt, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, Ceil(matrixWidth, 4), matrixHeight, 1);
}


void CVulkanMathEngine::SubVectorFromMatrixColumns( const CConstFloatHandle&, const CFloatHandle&, int, int, const CConstFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::SetVectorToMatrixRows( const CFloatHandle& resultHandle, int matrixHeight,
	int matrixWidth, const CConstFloatHandle& vectorHandle )
{
	if( device->Type == VDT_Adreno ) {
		const CVulkanImage* samplers[] = { &batchVectorToImage( 1, vectorHandle, matrixWidth, TVI_FreeTerm ) };

		CMemoryHandle bufs[1] = { resultHandle };
		size_t sizes[1] = { matrixHeight * matrixWidth * sizeof( float ) };

		PARAM_STRUCT( SetVectorToMatrixRowsAdreno ) param = { matrixHeight, matrixWidth };

		runShader( shaderLoader->GET_SHADER_DATA( SetVectorToMatrixRowsAdreno, true, 0, 1, 1 ),
			&param, sizeof( param ), 0, 0, samplers, 1, bufs, sizes, 1, Ceil( matrixWidth, 4 ), matrixHeight, 1 );
	} else {
		CMemoryHandle bufs[2] = { vectorHandle, resultHandle };
		size_t sizes[2] = { matrixWidth * sizeof( float ), matrixHeight * matrixWidth * sizeof( float ) };

		PARAM_STRUCT( SetVectorToMatrixRows ) param = { matrixHeight, matrixWidth };

		runShader( shaderLoader->GET_SHADER_DATA( SetVectorToMatrixRows, true, 0, 0, 2 ),
			&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 2, Ceil( matrixWidth, 4 ), matrixHeight, 1 );
	}
}

void CVulkanMathEngine::AddVectorToMatrixElements( const CFloatHandle&, int, int, const CConstIntHandle&, const CConstFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::AddVectorToMatrixElements( const CFloatHandle&, int, int, const CConstIntHandle&, const CConstIntHandle&,
	const CConstFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::SetVectorToMatrixElements( const CFloatHandle& matrixHandle, int height, int width,
	const CConstIntHandle& rowIndicesHandle, const CConstIntHandle& columnIndicesHandle,
	const CConstFloatHandle& vectorHandle, int vectorSize )
{
	CMemoryHandle bufs[4] = { rowIndicesHandle, columnIndicesHandle, vectorHandle, matrixHandle };
	size_t sizes[4] = { vectorSize * sizeof(int), vectorSize * sizeof(int), vectorSize * sizeof(float), height * width * sizeof(float) };

	PARAM_STRUCT(SetVectorToMatrixElements) param = { width };

	runShader( shaderLoader->GET_SHADER_DATA(SetVectorToMatrixElements, true, 0, 0, 4),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 4, vectorSize, 1, 1 );
}

void CVulkanMathEngine::EltwiseLogSumExpVectorToMatrixElements( const CFloatHandle&, int, int, const CConstIntHandle&,
	const CConstFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::EltwiseLogSumExpVectorToMatrixElements( const CFloatHandle&, int, int,
	const CConstIntHandle&, const CConstIntHandle&, const CConstFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
	const CConstIntHandle& indices, const CFloatHandle& result, int vectorSize)
{
	ASSERT_EXPR(vectorSize >= height);

	CMemoryHandle bufs[3] = { matrix, indices, result };
	size_t sizes[3] = { height * width * sizeof(float), height * sizeof(int), height * sizeof(float) };

    PARAM_STRUCT(AddMatrixElementsToVector) param = { height, width };

	runShader( shaderLoader->GET_SHADER_DATA(AddMatrixElementsToVector, true, 0, 0, 3), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 3, Ceil(height, 4), 1, 1 );
}

void CVulkanMathEngine::AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
	const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
	const CFloatHandle& result, int vectorSize)
{	
	CMemoryHandle bufs[4] = { matrix, rowIndices, columnIndices, result };
	size_t sizes[4] = { height * width * sizeof(float), vectorSize * sizeof(int), vectorSize * sizeof(int), vectorSize * sizeof(float) };

    PARAM_STRUCT(AddMatrixElementsToVectorEx) param = { vectorSize, width };

	runShader( shaderLoader->GET_SHADER_DATA(AddMatrixElementsToVectorEx, true, 0, 0, 4), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 4, Ceil(vectorSize, 4), 1, 1 );
}

void CVulkanMathEngine::AddMatrixElementsToMatrix( const CConstFloatHandle&, int, int, const CFloatHandle&, const CConstIntHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::MultiplyDiagMatrixByMatrixAndAdd( int batchSize, const CConstFloatHandle& firstHandle,
	int firstSize, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle )
{
	CMemoryHandle bufs[3] = { firstHandle, secondHandle, resultHandle };
	size_t sizes[3] = { batchSize * firstSize * sizeof(float), batchSize * firstSize * secondWidth * sizeof(float), firstSize * secondWidth * sizeof(float) };

	PARAM_STRUCT(MultiplyDiagMatrixByMatrixAndAdd) param =
		{ batchSize, firstSize, secondWidth };

	runShader( shaderLoader->GET_SHADER_DATA(MultiplyDiagMatrixByMatrixAndAdd, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, secondWidth, firstSize, 1 );
}

void CVulkanMathEngine::FindMaxValueInRows( const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CFloatHandle& resultHandle, const CIntHandle& columnIndices, int vectorSize )
{
	ASSERT_EXPR( vectorSize >= matrixHeight );

	CMemoryHandle bufs[3] = { matrixHandle, resultHandle, columnIndices };
	size_t sizes[3] = { matrixHeight * matrixWidth * sizeof(float), matrixHeight * sizeof(float), matrixHeight * sizeof(int) };

	PARAM_STRUCT(FindMaxValueInRows) param =
		{ matrixHeight, matrixWidth };

	runShader( shaderLoader->GET_SHADER_DATA(FindMaxValueInRows, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, 1, matrixHeight, 1 );
}

void CVulkanMathEngine::FindMaxValueInRows( const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( vectorSize >= matrixHeight );

	CMemoryHandle bufs[2] = { matrixHandle, resultHandle };
	size_t sizes[2] = { matrixHeight * matrixWidth * sizeof(float), matrixHeight * sizeof(float) };

	PARAM_STRUCT(FindMaxValueInRowsNoIndices) param =
		{ matrixHeight, matrixWidth };

	runShader( shaderLoader->GET_SHADER_DATA(FindMaxValueInRowsNoIndices, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, 1, matrixHeight, 1 );
}

void CVulkanMathEngine::FindMaxValueInColumns( int batchSize, const CConstFloatHandle& matrixHandle, int matrixHeight,
	int matrixWidth, const CFloatHandle& resultHandle, const CIntHandle& rowIndices, int vectorSize )
{
	ASSERT_EXPR( vectorSize >= batchSize * matrixWidth );

	CMemoryHandle bufs[3] = { matrixHandle, resultHandle, rowIndices };
	size_t sizes[3] = { batchSize * matrixHeight * matrixWidth * sizeof(float), batchSize * matrixWidth * sizeof(float), batchSize * matrixWidth * sizeof(int) };

	PARAM_STRUCT(FindMaxValueInColumns) param =
		{ batchSize, matrixHeight, matrixWidth };

	runShader( shaderLoader->GET_SHADER_DATA(FindMaxValueInColumns, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, matrixWidth, batchSize, 1 );
}

void CVulkanMathEngine::FindMinValueInColumns( const CConstFloatHandle& matrixHandle, int matrixHeight,
	int matrixWidth, const CFloatHandle& resultHandle, const CIntHandle& columnIndices )
{
	CMemoryHandle bufs[3] = { matrixHandle, resultHandle, columnIndices };
	size_t sizes[3] = { matrixHeight * matrixWidth * sizeof( float ), matrixWidth * sizeof( float ), matrixWidth * sizeof( int ) };

	PARAM_STRUCT( FindMinValueInColumns ) param =
		{ matrixHeight, matrixWidth };

	runShader( shaderLoader->GET_SHADER_DATA( FindMinValueInColumns, true, 0, 0, 3 ),
		&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 2, matrixWidth, 1, 1 );
}

void CVulkanMathEngine::findMaxValueInColumns( const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth )
{
	CMemoryHandle bufs[2] = { matrixHandle, resultHandle };
	size_t sizes[2] = { matrixHeight * matrixWidth * sizeof(float), matrixWidth * sizeof(float) };

	PARAM_STRUCT(FindMaxValueInColumnsNoIndices) param =
		{ matrixHeight, matrixWidth };

	runShader( shaderLoader->GET_SHADER_DATA(FindMaxValueInColumnsNoIndices, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, matrixWidth, 1, 1 );	
}

void CVulkanMathEngine::VectorMultichannelLookupAndCopy( int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
	const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CFloatHandle& outputHandle, int outputChannelCount )
{
    ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( outputHandle.GetMathEngine() == this );

    int outputChannel = 0;
    for( int i = 0; i < lookupCount; ++i ) {
		CMemoryHandle bufs[3] = { inputHandle, lookupHandles[i], outputHandle };
		size_t sizes[3] = { batchSize * channelCount * sizeof(float), 
			lookupDimensions[i].VectorSize * lookupDimensions[i].VectorCount * sizeof(float), batchSize * outputChannelCount * sizeof(int) };

		PARAM_STRUCT(VectorMultichannelLookupAndCopyFloat) param =
			{ batchSize, i, channelCount, outputChannelCount, lookupDimensions[i].VectorSize, outputChannel };

		runShader( shaderLoader->GET_SHADER_DATA(VectorMultichannelLookupAndCopyFloat, true, 0, 0, 3),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, Ceil(batchSize, 4), lookupDimensions[i].VectorSize, 1 );	
        
        outputChannel += lookupDimensions[i].VectorSize;
    }
    
    if( lookupCount < channelCount ) {
    	CMemoryHandle bufs[2] = { inputHandle, outputHandle };
		size_t sizes[2] = { batchSize * channelCount * sizeof(float), batchSize * outputChannelCount * sizeof(int)  };

		PARAM_STRUCT(VectorMultichannelCopyFloat) param =
			{ batchSize, channelCount, outputChannelCount, lookupCount, outputChannel, channelCount - lookupCount };

		runShader( shaderLoader->GET_SHADER_DATA(VectorMultichannelCopyFloat, true, 0, 0, 2),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, Ceil(batchSize, 4), channelCount - lookupCount, 1 );	
	}
}

void CVulkanMathEngine::VectorMultichannelLookupAndCopy( int batchSize, int channelCount, const CConstIntHandle& inputHandle,
	const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CFloatHandle& outputHandle, int outputChannelCount )
{
    ASSERT_EXPR( inputHandle.GetMathEngine() == this );
	ASSERT_EXPR( outputHandle.GetMathEngine() == this );

    int outputChannel = 0;
    for( int i = 0; i < lookupCount; ++i ) {
		CMemoryHandle bufs[3] = { inputHandle, lookupHandles[i], outputHandle };
		size_t sizes[3] = { batchSize * channelCount * sizeof(int), 
			lookupDimensions[i].VectorSize * lookupDimensions[i].VectorCount * sizeof(float), batchSize * outputChannelCount * sizeof(int) };

		PARAM_STRUCT(VectorMultichannelLookupAndCopyInt) param =
			{ batchSize, i, channelCount, outputChannelCount, lookupDimensions[i].VectorSize, outputChannel };

		runShader( shaderLoader->GET_SHADER_DATA(VectorMultichannelLookupAndCopyInt, true, 0, 0, 3),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, Ceil(batchSize, 4), lookupDimensions[i].VectorSize, 1 );	
        
        outputChannel += lookupDimensions[i].VectorSize;
    }
    
    if( lookupCount < channelCount ) {
    	CMemoryHandle bufs[2] = { inputHandle, outputHandle };
		size_t sizes[2] = { batchSize * channelCount * sizeof(float), batchSize * outputChannelCount * sizeof(int)  };

		PARAM_STRUCT(VectorMultichannelCopyInt) param =
			{ batchSize, channelCount, outputChannelCount, lookupCount, outputChannel, channelCount - lookupCount };

		runShader( shaderLoader->GET_SHADER_DATA(VectorMultichannelCopyInt, true, 0, 0, 2),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, Ceil(batchSize, 4), channelCount - lookupCount, 1 );	
	}
}

void CVulkanMathEngine::VectorMultichannelLookupAndAddToTable( int, int, const CConstFloatHandle&,
	const CFloatHandle*, const CLookupDimension*, int, const CConstFloatHandle&, const CConstFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::VectorMultichannelLookupAndAddToTable( int, int, const CConstIntHandle&,
	const CFloatHandle*, const CLookupDimension*, int, const CConstFloatHandle&, const CConstFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::RowMultiplyMatrixByMatrix( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	int height, int width, const CFloatHandle& result )
{
	const int batchSize = 1;
	const int yStep = width;
	const int xStep = 1;

	CMemoryHandle bufs[3] = { firstHandle, secondHandle, result };
	int matrixSize = height * width;
	int resultSize = batchSize * ((yStep == 1) ? width : height);
	size_t sizes[3] = { batchSize * matrixSize * sizeof(float),
		matrixSize * sizeof(float), resultSize * sizeof(float) };

	PARAM_STRUCT(RowMultiplyMatrixByMatrix) param = { height, width, yStep, xStep };

	const CVulkanShaderData& shaderData = shaderLoader->GET_SHADER_DATA(RowMultiplyMatrixByMatrix, true, 0, 0, 3);

	runShader( shaderData, &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 3, shaderData.GroupSizeX, height, batchSize );
}

void CVulkanMathEngine::MatrixSpreadRows( const CConstFloatHandle& sourceHandle, int height, int width,
	const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
	const CConstFloatHandle& fillValue )
{
	if( fillValue.IsNull() ) {
        VectorFill(resultHandle, 0.0f, resultHeight * width);
    } else {
        VectorFill(resultHandle, resultHeight * width, fillValue);
    }
    
	CMemoryHandle bufs[3] = { sourceHandle, indexHandle, resultHandle };
	size_t sizes[3] = { height * width * sizeof( float ),
		height * sizeof( int ), width * resultHeight * sizeof( float ) };

	PARAM_STRUCT( MatrixSpreadRowsFloat ) param = { height, width };

	runShader( shaderLoader->GET_SHADER_DATA( MatrixSpreadRowsFloat, true, 0, 0, 3 ), &param, sizeof( param ),
		0, 0, 0, 0, bufs, sizes, 3, Ceil( width, 8 ), height, 1 );
}

void CVulkanMathEngine::MatrixSpreadRowsAdd( const CConstFloatHandle& sourceHandle, int height, int width,
	const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle )
{
	CMemoryHandle bufs[3] = { sourceHandle, indexHandle, resultHandle };
	size_t sizes[3] = { height * width * sizeof( float ),
		height * sizeof( int ), width * resultHeight * sizeof( float ) };

	PARAM_STRUCT( MatrixSpreadRowsFloatAdd ) param = { height, width };

	runShader( shaderLoader->GET_SHADER_DATA( MatrixSpreadRowsFloatAdd, true, 0, 0, 3 ), &param, sizeof( param ),
		0, 0, 0, 0, bufs, sizes, 3, Ceil( width, 8 ), height, 1 );
}

void CVulkanMathEngine::MatrixSpreadRows( const CConstIntHandle& sourceHandle, int height, int width,
	const CIntHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
	const CConstIntHandle& fillValue )
{
	if( fillValue.IsNull() ) {
        VectorFill(resultHandle, 0, resultHeight * width);
    } else {
        VectorFill(resultHandle, resultHeight * width, fillValue);
    }
    
	CMemoryHandle bufs[3] = { sourceHandle, indexHandle, resultHandle };
	size_t sizes[3] = { height * width * sizeof(int),
		height * sizeof(int), width * resultHeight * sizeof(int) };

	PARAM_STRUCT(MatrixSpreadRowsInt) param = { height, width };

	runShader( shaderLoader->GET_SHADER_DATA(MatrixSpreadRowsInt, true, 0, 0, 3), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 3, Ceil(width, 8), height, 1 );
}

void CVulkanMathEngine::SumMatrixRowsAdd( int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth )
{
	CMemoryHandle bufs[2] = { matrixHandle, resultHandle };
	size_t sizes[2] = { batchSize * matrixHeight * matrixWidth * sizeof(float),
		batchSize * matrixWidth * sizeof(float) };

	PARAM_STRUCT(SumMatrixRows) param = { matrixWidth, matrixHeight, batchSize, 1 };

	runShader(shaderLoader->GET_SHADER_DATA(SumMatrixRows, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, matrixWidth, 1, batchSize);
}

void CVulkanMathEngine::SumMatrixRows( int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth )
{
	CMemoryHandle bufs[2] = { matrixHandle, resultHandle };
	size_t sizes[2] = { batchSize * matrixHeight * matrixWidth * sizeof(float),
		batchSize * matrixWidth * sizeof(float) };

	PARAM_STRUCT(SumMatrixRows) param = { matrixWidth, matrixHeight, batchSize, 0 };

	runShader(shaderLoader->GET_SHADER_DATA(SumMatrixRows, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, matrixWidth, 1, batchSize);
}

void CVulkanMathEngine::SumMatrixColumns( const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth )
{
	CMemoryHandle bufs[2] = { matrixHandle, resultHandle };
	size_t sizes[2] = { matrixHeight * matrixWidth * sizeof(float), matrixHeight * sizeof(float) };

	PARAM_STRUCT(SumMatrixColumns) param = { matrixWidth, matrixHeight };

	runShader(shaderLoader->GET_SHADER_DATA(SumMatrixColumns, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, matrixHeight, 1, 1);
}

void CVulkanMathEngine::MatrixLogSumExpByRows( const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result, int resultSize )
{
	ASSERT_EXPR( resultSize >= height );

	CMemoryHandle bufs[2] = { matrix, result };
	size_t sizes[2] = { height * width * sizeof(float), height * sizeof(float) };

	PARAM_STRUCT(MatrixLogSumExpByRows) param = { height, width };

	const CVulkanShaderData& shaderData = shaderLoader->GET_SHADER_DATA(MatrixLogSumExpByRows, true, 0, 0, 2);
	runShader( shaderData, &param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, shaderData.GroupSizeX, height, 1 );
}

void CVulkanMathEngine::MatrixSoftmaxByRows( const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result)
{
	CMemoryHandle bufs[2] = { matrix, result };
	size_t sizes[2] = { height * width * sizeof(float), height * width * sizeof(float) };

	PARAM_STRUCT(MatrixSoftmaxByRows) param = { height, width };

	const CVulkanShaderData& shaderData = shaderLoader->GET_SHADER_DATA(MatrixSoftmaxByRows, true, 0, 0, 2);
	runShader( shaderData, &param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, 1, height, 1 );
}

void CVulkanMathEngine::MatrixSoftmaxDiffOpByRows( const CConstFloatHandle&, const CConstFloatHandle&, int, int, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::MatrixLogSumExpByColumns( const CConstFloatHandle&, int, int, const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::MatrixSoftmaxByColumns( const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result )
{
	CMemoryHandle bufs[2] = { matrix, result };
	size_t sizes[2] = { height * width * sizeof(float), height * width * sizeof(float) };

	PARAM_STRUCT(MatrixSoftmaxByColumns) param = { height, width };

	const CVulkanShaderData& shaderData = shaderLoader->GET_SHADER_DATA(MatrixSoftmaxByColumns, true, 0, 0, 2);
	runShader( shaderData, &param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2, width, 1, 1 );
}

void CVulkanMathEngine::MatrixSoftmaxDiffOpByColumns(const CConstFloatHandle&, const CConstFloatHandle&, int, int, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::MultiplyMatrixByDiagMatrix( const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int resultBufferSize )
{
	const int batchSize = 1;
	const int secondBatchSize = 1;
	const bool toAdd = false;
	int matrixSize = batchSize * firstHeight * firstWidth;

	ASSERT_EXPR( resultBufferSize >= matrixSize );

	if( device->Type == VDT_Adreno ) {
		const CVulkanImage* samplers[] =
		{ &batchVectorToImage( secondBatchSize, secondHandle, firstWidth, TVI_DiagMatrix ) };

		CMemoryHandle bufs[2] = { firstHandle, resultHandle };
		size_t sizes[2] = { matrixSize * sizeof( float ), matrixSize * sizeof( float ) };

		PARAM_STRUCT( MultiplyMatrixByDiagMatrixAdreno ) param =
		{ batchSize, secondBatchSize, firstHeight, firstWidth, ( toAdd ? 1 : 0 ) };

		runShader( shaderLoader->GET_SHADER_DATA( MultiplyMatrixByDiagMatrixAdreno, true, 0, 1, 2 ),
			&param, sizeof( param ), 0, 0, samplers, 1, bufs, sizes, 2, Ceil( firstWidth, 4 ), batchSize * firstHeight, 1 );
	} else {
		CMemoryHandle bufs[3] = { firstHandle, secondHandle, resultHandle };
		size_t sizes[3] = { matrixSize * sizeof( float ), firstWidth * sizeof( float ), matrixSize * sizeof( float ) };

		PARAM_STRUCT( MultiplyMatrixByDiagMatrix ) param =
		{ batchSize, secondBatchSize, firstHeight, firstWidth, ( toAdd ? 1 : 0 ) };

		runShader( shaderLoader->GET_SHADER_DATA( MultiplyMatrixByDiagMatrix, true, 0, 0, 3 ),
			&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 3, Ceil( firstWidth, 4 ), batchSize * firstHeight, 1 );
	}
}

//------------------------------------------------------------------------------------------------------------
// private methods

void CVulkanMathEngine::multiplyMatrixByMatrix( bool toAdd, int batchSize, const CConstFloatHandle& firstHandle,
	int firstHeight, int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondWidth, 
	int secondRowSize, const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize )
{	
    ASSERT_EXPR( firstWidth <= firstRowSize );
    ASSERT_EXPR( secondWidth <= secondRowSize );
    ASSERT_EXPR( secondWidth <= resultRowSize );
    ASSERT_EXPR( firstHeight * resultRowSize <= resultBufferSize );

	CMemoryHandle bufs[3] = { firstHandle, secondHandle, resultHandle };
	size_t sizes[3] = { batchSize * firstHeight * firstWidth * sizeof(float), 
						batchSize * firstWidth * secondWidth * sizeof(float), 
						batchSize * firstHeight * secondWidth * sizeof(float) };
	
	if( firstHeight >= 4 && secondWidth >= 4 ) {
		PARAM_STRUCT(MultiplyMatrixByMatrix) param = { batchSize, firstHeight, firstWidth, firstRowSize,
			secondWidth, secondRowSize, resultRowSize, toAdd ? 1 : 0 };
		runShader(shaderLoader->GET_SHADER_DATA(MultiplyMatrixByMatrix, true, 0, 0, 3),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, secondWidth / 4, firstHeight / 4, batchSize);
	}

	int leftOffset = secondWidth - secondWidth % 4;
    int topOffset = firstHeight - firstHeight % 4;
    int count = secondWidth * firstHeight - leftOffset * topOffset;
    if( count > 0 ) {
		PARAM_STRUCT(BatchMultiplyMatrixByMatrixBorders) param = { batchSize, firstHeight, firstWidth, firstRowSize, 
			secondWidth, secondRowSize, resultRowSize, leftOffset, topOffset, toAdd ? 1 : 0 };
		runShader( shaderLoader->GET_SHADER_DATA(BatchMultiplyMatrixByMatrixBorders, true, 0, 0, 3),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, count, batchSize, 1 );
	}
}

void CVulkanMathEngine::batchMultiplyMatrixByTransposedMatrix( bool toAdd, int batchSize, const CConstFloatHandle& firstHandle, 
	int firstHeight, int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondHeight,
	int secondRowSize, const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize )
{
    ASSERT_EXPR( firstWidth <= firstRowSize );
    ASSERT_EXPR( firstWidth <= secondRowSize );
    ASSERT_EXPR( secondHeight <= resultRowSize );
    ASSERT_EXPR( ( firstHeight - 1 ) * resultRowSize + secondHeight <= resultBufferSize );

	CMemoryHandle bufs[3] = { firstHandle, secondHandle, resultHandle };
	size_t sizes[3] = { batchSize * firstHeight * firstWidth * sizeof(float), 
						batchSize * firstWidth * secondHeight * sizeof(float), 
						batchSize * firstHeight * secondHeight * sizeof(float) };
	
	if( firstHeight >= 4 && secondHeight >= 4 ) {
		PARAM_STRUCT(BatchMultiplyMatrixByTransposedMatrix) param = { batchSize, firstHeight, firstWidth, firstRowSize,
			secondHeight, secondRowSize, resultRowSize,  ( toAdd ) ? 1 : 0 };
		runShader( shaderLoader->GET_SHADER_DATA( BatchMultiplyMatrixByTransposedMatrix, true, 0, 0, 3),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, firstHeight / 4, secondHeight / 4, batchSize );
	}

	int leftOffset = secondHeight - secondHeight % 4;
	int topOffset = firstHeight - firstHeight % 4;
	int count = secondHeight * firstHeight - leftOffset * topOffset;
	if( count > 0 ) {
		PARAM_STRUCT(BatchMultiplyMatrixByTransposedMatrixBorders) param = { batchSize, firstHeight, firstWidth, firstRowSize,
			secondHeight, secondRowSize, resultRowSize, leftOffset, topOffset, toAdd };
		runShader( shaderLoader->GET_SHADER_DATA(BatchMultiplyMatrixByTransposedMatrixBorders, true, 0, 0, 3),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, count, batchSize, 1 );
	}
}

void CVulkanMathEngine::blobConvolution1x1s1( int batchSize, const CConstFloatHandle& initHandle,
	const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth, int firstRowSize,
	const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
	const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize )
{
	ASSERT_EXPR( firstWidth <= firstRowSize );
	ASSERT_EXPR( firstWidth <= secondRowSize );
	ASSERT_EXPR( secondHeight <= resultRowSize );
	ASSERT_EXPR( ( firstHeight - 1 ) * resultRowSize + secondHeight <= resultBufferSize );

	CMemoryHandle bufs[4] = { initHandle, firstHandle, secondHandle, resultHandle };
	size_t sizes[4] = { batchSize * secondHeight * sizeof( float ),
		batchSize * firstHeight * firstWidth * sizeof( float ),
		batchSize * firstWidth * secondHeight * sizeof( float ),
		batchSize * firstHeight * secondHeight * sizeof( float ) };

	if( firstHeight >= 4 && secondHeight >= 4 ) {
		PARAM_STRUCT( BatchInitAddMultiplyMatrixByTransposedMatrix ) param = { batchSize, firstHeight, firstWidth, firstRowSize,
			secondHeight, secondRowSize, resultRowSize };
		runShader( shaderLoader->GET_SHADER_DATA( BatchInitAddMultiplyMatrixByTransposedMatrix, true, 0, 0, 4 ),
			&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 4, firstHeight / 4, secondHeight / 4, batchSize );
	}

	int leftOffset = secondHeight - secondHeight % 4;
	int topOffset = firstHeight - firstHeight % 4;
	int count = secondHeight * firstHeight - leftOffset * topOffset;
	if( count > 0 ) {
		PARAM_STRUCT( BatchInitMultiplyMatrixByTransposedMatrixBorders ) param = { batchSize, firstHeight, firstWidth, firstRowSize,
			secondHeight, secondRowSize, resultRowSize, leftOffset, topOffset };
		runShader( shaderLoader->GET_SHADER_DATA( BatchInitMultiplyMatrixByTransposedMatrixBorders, true, 0, 0, 4 ),
			&param, sizeof( param ), 0, 0, 0, 0, bufs, sizes, 4, count, batchSize, 1 );
	}
}

void CVulkanMathEngine::batchMultiplyTransposedMatrixByMatrix( bool toAdd, int batchSize, const CConstFloatHandle& firstHandle, 
	int firstHeight, int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondWidth,
	int secondRowSize, const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize )
{	
    ASSERT_EXPR( firstWidth <= firstRowSize );
    ASSERT_EXPR( secondWidth <= secondRowSize );
    ASSERT_EXPR( secondWidth <= resultRowSize );
    ASSERT_EXPR( ( firstWidth - 1 ) * resultRowSize + secondWidth <= resultBufferSize );

	CMemoryHandle bufs[3] = { firstHandle, secondHandle, resultHandle };
	size_t sizes[3] = { batchSize * firstHeight * firstWidth * sizeof(float), 
						batchSize * firstHeight * secondWidth * sizeof(float), 
						batchSize * firstWidth * secondWidth * sizeof(float) };
	
	if( firstWidth >= 4 && secondWidth >= 4 ) {
		PARAM_STRUCT(BatchMultiplyTransposedMatrixByMatrix) param = { batchSize, firstHeight, firstWidth, firstRowSize,
			secondWidth, secondRowSize, resultRowSize, toAdd ? 1 : 0 };
		runShader(shaderLoader->GET_SHADER_DATA(BatchMultiplyTransposedMatrixByMatrix, true, 0, 0, 3),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, secondWidth / 4, firstWidth / 4, batchSize);
	}

	int leftOffset = secondWidth - secondWidth % 4;
	int topOffset = firstWidth - firstWidth % 4;
	int count = secondWidth * firstWidth - leftOffset * topOffset;
	if( count > 0 ) {
		PARAM_STRUCT(BatchMultiplyTransposedMatrixByMatrixBorders) param = { batchSize, firstHeight, firstWidth, firstRowSize, 
			secondWidth, secondRowSize, resultRowSize, leftOffset, topOffset, toAdd ? 1 : 0 };
		runShader( shaderLoader->GET_SHADER_DATA(BatchMultiplyTransposedMatrixByMatrixBorders, true, 0, 0, 3),
			&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, count, batchSize, 1 );
	}
}

struct CInterleavedMatrixDesc {
	const CVulkanImage* Image;
	int ChunkX;
	int ChunkY;
};

void CVulkanMathEngine::matrixToInterleavedAdreno( int batchSize, CConstFloatHandle from, int height, int width, int rowSize,
	bool isTrans, int imageId, CInterleavedMatrixDesc& result )
{
	ASSERT_EXPR( device->Type == VDT_Adreno );
	ASSERT_EXPR( device->IsImageBased );

	int resHeight4 = Ceil((isTrans ? width : height), 4);
	int resWidth4 = Ceil((isTrans ? height : width), 4);

	// Split the matrix into chunks if it doesn't fit into the 2D limitations
	int imageHeight = batchSize * resHeight4;
	int imageWidth = resWidth4 * 4;

	int chunkY = imageHeight;
	int chunkX = imageWidth;

	int maxImageDimension2D = device->Properties.limits.maxImageDimension2D;
	if( imageHeight > maxImageDimension2D ) {
		// The matrix is too high
		int chunkCount = Ceil(imageHeight, maxImageDimension2D);
		chunkY = imageHeight = maxImageDimension2D;
		imageWidth *= chunkCount;
	} else if( imageWidth > maxImageDimension2D ) {
		// The matrix is too wide
		int limit = FloorTo(maxImageDimension2D, 4); // the width should be a multiple of 4
		int chunkCount = Ceil(imageWidth, limit);
		chunkX = imageWidth = limit;
		imageHeight *= chunkCount;
	}

	const CVulkanImage* images[] = { getTmpImage( (TTmpVulkanImage)imageId, imageWidth, imageHeight ) };

	CMemoryHandle bufs[1] = { from };
	size_t sizes[1] = { ((batchSize * height - 1) * rowSize + width) * sizeof(float) };

	PARAM_STRUCT( Matrix2InterleavedAdreno ) param = { { chunkX, chunkY }, batchSize, height, width, rowSize,
		(isTrans ? 1 : 0) };

	runShader( shaderLoader->GET_SHADER_DATA( Matrix2InterleavedAdreno, true, 1, 0, 1),
		&param, sizeof(param), images, 1, 0, 0, bufs, sizes, 1, resWidth4, batchSize * resHeight4, 1 );

	result = { images[0], chunkX, chunkY };
}

void CVulkanMathEngine::batchMultiplyMatrixByMatrixAdreno( bool toAdd, int batchSize,
	const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth, int firstRowSize, bool isFirstTrans,
	const CConstFloatHandle& secondHandle, int secondHeight, int secondWidth, int secondRowSize,
	bool isSecondTrans, const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize )
{
	ASSERT_EXPR( device->Type == VDT_Adreno );
	ASSERT_EXPR( device->IsImageBased );

	ASSERT_EXPR(firstWidth<= firstRowSize);
	ASSERT_EXPR(secondWidth<= secondRowSize);

	int resHeight = isFirstTrans ? firstWidth : firstHeight;
	ASSERT_EXPR(resHeight > 0);
	int medium = isFirstTrans ? firstHeight : firstWidth;
	ASSERT_EXPR(medium > 0);
	int resWidth = isSecondTrans ? secondHeight : secondWidth;
	ASSERT_EXPR(resWidth > 0);

	ASSERT_EXPR(resWidth <= resultRowSize);
	ASSERT_EXPR((isSecondTrans ? secondWidth : secondHeight) == medium);
	ASSERT_EXPR((batchSize * resHeight - 1) * resultRowSize + resWidth <= resultBufferSize);

	//////////////////////////////////////////////////////
	// Convert the matrices into interleaved format
	int resHeight4 = Ceil(resHeight, 4);
	int medium4 = Ceil(medium, 4);
	int resWidth4 = Ceil(resWidth, 4);

	CInterleavedMatrixDesc imageFirst;
	matrixToInterleavedAdreno(batchSize, firstHandle, firstHeight, firstWidth, firstRowSize, isFirstTrans, TVI_MatrixLeft, imageFirst);
	CInterleavedMatrixDesc imageSecond;
	matrixToInterleavedAdreno(batchSize, secondHandle, secondHeight, secondWidth, secondRowSize, isSecondTrans, TVI_MatrixRight, imageSecond);

	///////////////////////////////////////////////////////////
	// Perform multiplication
	const CVulkanImage* samplers[] = { imageFirst.Image, imageSecond.Image };

	CMemoryHandle bufs[1] = { resultHandle };
	size_t sizes[1] = { ((batchSize * resHeight - 1) * resultRowSize + resWidth) * sizeof(float) };

	int resHeight4Low = resHeight / 4;
	int resWidth4Low = resWidth / 4;

	if( resHeight4Low > 0 && resWidth4Low > 0 ) {
		// Start processing the main matrix
		PARAM_STRUCT( MultiplyMatrixInterleavedAdreno ) param =
		{ { imageFirst.ChunkX, imageFirst.ChunkY }, { imageSecond.ChunkX, imageSecond.ChunkY },
			batchSize, resHeight, medium4, resWidth, (toAdd ? 1 : 0), resultRowSize };

		runShader( shaderLoader->GET_SHADER_DATA( MultiplyMatrixInterleavedAdreno, true, 0, 2, 1),
			&param, sizeof(param), 0, 0, samplers, 2, bufs, sizes, 1, resWidth4Low, batchSize * resHeight4Low, 1 );
	}

	if( resHeight4 > resHeight4Low && resWidth4Low > 0 ) {
		// Start processing the bottom without the bottom right corner
		PARAM_STRUCT( MultiplyMatrixInterleavedBoardersAdreno ) param =
		{ { imageFirst.ChunkX, imageFirst.ChunkY }, { imageSecond.ChunkX, imageSecond.ChunkY },
			batchSize, resHeight, medium4, resWidth, (toAdd ? 1 : 0), resultRowSize,
			0, resWidth4Low, resHeight4Low, resHeight4 };

		runShader( shaderLoader->GET_SHADER_DATA( MultiplyMatrixInterleavedBoardersAdreno, true, 0, 2, 1),
			&param, sizeof(param), 0, 0, samplers, 2, bufs, sizes, 1, resWidth4Low, batchSize, 1 );
	}

	if( resWidth4 > resWidth4Low && resHeight4Low > 0 ) {
		// Start processing the right without the bottom right corner
		PARAM_STRUCT( MultiplyMatrixInterleavedBoardersAdreno ) param =
		{ { imageFirst.ChunkX, imageFirst.ChunkY }, { imageSecond.ChunkX, imageSecond.ChunkY },
			batchSize, resHeight, medium4, resWidth, (toAdd ? 1 : 0), resultRowSize,
			resWidth4Low, resWidth4, 0, resHeight4Low };

		runShader( shaderLoader->GET_SHADER_DATA( MultiplyMatrixInterleavedBoardersAdreno, true, 0, 2, 1),
			&param, sizeof(param), 0, 0, samplers, 2, bufs, sizes, 1, 1, batchSize * resHeight4Low, 1 );
	}

	if( resWidth4 > resWidth4Low && resHeight4 > resHeight4Low ) {
		// Start processing the bottom right corner
		PARAM_STRUCT( MultiplyMatrixInterleavedBoardersAdreno ) param =
		{ { imageFirst.ChunkX, imageFirst.ChunkY }, { imageSecond.ChunkX, imageSecond.ChunkY },
			batchSize, resHeight, medium4, resWidth, (toAdd ? 1 : 0), resultRowSize,
			resWidth4Low, resWidth4, resHeight4Low, resHeight4 };

		runShader( shaderLoader->GET_SHADER_DATA( MultiplyMatrixInterleavedBoardersAdreno, true, 0, 2, 1),
			&param, sizeof(param), 0, 0, samplers, 2, bufs, sizes, 1, 1, batchSize, 1 );
	}
}

void CVulkanMathEngine::LookupAndSum(const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
	const CConstFloatHandle& tableHandle, int vectorSize, const CFloatHandle& result)
{
	CMemoryHandle bufs[3] = { indicesHandle, tableHandle, result };
	size_t sizes[3] = { batchSize * indexCount * sizeof(int),
		vectorSize * indexCount * sizeof(float), batchSize * vectorSize * sizeof(float) };

	PARAM_STRUCT(LookupAndSum) param = { batchSize, indexCount, vectorSize };

	runShader( shaderLoader->GET_SHADER_DATA(LookupAndSum, true, 0, 0, 3), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 3, vectorSize, batchSize, 1 );
}

void CVulkanMathEngine::LookupAndAddToTable( const CConstIntHandle&, int, int, const CConstFloatHandle&, int, const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::EnumBinarization( int batchSize, const CConstFloatHandle& inputHandle, int enumSize,
	const CFloatHandle& resultHandle )
{
	VectorFill(resultHandle, 0.0f, batchSize * enumSize);

	CMemoryHandle bufs[2] = { inputHandle, resultHandle };
	size_t sizes[4] = { batchSize * sizeof(float), batchSize * enumSize * sizeof(float) };

    PARAM_STRUCT(EnumBinarizationFloat) param = { batchSize, enumSize };

	runShader( shaderLoader->GET_SHADER_DATA(EnumBinarizationFloat, true, 0, 0, 2), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 2, Ceil(batchSize, 4), 1, 1 );
}

void CVulkanMathEngine::EnumBinarization( int batchSize, const CConstIntHandle& inputHandle, int enumSize,
	const CFloatHandle& resultHandle )
{
	VectorFill(resultHandle, 0, batchSize * enumSize);

	CMemoryHandle bufs[2] = { inputHandle, resultHandle };
	size_t sizes[4] = { batchSize * sizeof(int), batchSize * enumSize * sizeof(float) };

    PARAM_STRUCT(EnumBinarizationInt) param = { batchSize, enumSize };

	runShader( shaderLoader->GET_SHADER_DATA(EnumBinarizationInt, true, 0, 0, 2), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 2, Ceil(batchSize, 4), 1, 1 );
}

void CVulkanMathEngine::BitSetBinarization( int batchSize, int bitSetSize,
	const CConstIntHandle& inputHandle, int outputVectorSize, const CFloatHandle& resultHandle )
{
	const int BitsPerElement = sizeof( int ) * CHAR_BIT;
	ASSERT_EXPR( bitSetSize * BitsPerElement >= outputVectorSize );

	CMemoryHandle bufs[2] = { inputHandle, resultHandle };
	size_t sizes[4] = { batchSize * bitSetSize * sizeof(int), batchSize * outputVectorSize * sizeof(float) };

    PARAM_STRUCT(BitSetBinarization) param = { bitSetSize, outputVectorSize };

	runVectorShader( shaderLoader->GET_SHADER_DATA(BitSetBinarization, true, 0, 0, 2), &param, sizeof(param),
		0, 0, 0, 0, bufs, sizes, 2, Ceil(batchSize * outputVectorSize , 4) );
}

void CVulkanMathEngine::MultiplyLookupMatrixByLookupVector( int, const CLookupMatrix&, const CLookupVector&, const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::MultiplyTransposedLookupMatrixByVector( int, const CLookupMatrix&,
	const CConstFloatHandle&, const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::MultiplyTransposedLookupMatrixByVectorAndAdd( int, const CLookupMatrix&,
	const CConstFloatHandle&, const CFloatHandle&, int )
{
	ASSERT_EXPR( false );
}

void CVulkanMathEngine::MultiplyVectorByTransposedLookupVectorAndAddToTable( int,const CFloatHandle&, int, int,
	const CConstIntHandle&, const CConstFloatHandle&, int, const CLookupVector& )
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_VULKAN
