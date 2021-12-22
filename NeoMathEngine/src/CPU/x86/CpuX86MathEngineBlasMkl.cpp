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

#ifdef NEOML_USE_SSE

#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <CpuX86.h>
#include <float.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <NeoMathEngine/SimdMathEngine.h>

#ifdef NEOML_USE_MKL
#if FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
#include <mkl.h>
#else
#error Unknown platform
#endif
#else
#include <CPUInfo.h>
#include <MatrixMultiplyingInterleavedCommon/MatrixMultiplying.h>

// Cache sizes
// Find the acceptable values or get them from CPU info
static constexpr CCPUInfo CpuInfo( 0x60000, 0x180000, 0x900000 );
#endif
#include <MatrixMultiplyingInterleavedCommon/CpuMemoryHelper.h>

namespace NeoML {

void CCpuMathEngine::multiplyMatrixByMatrix( const float* first, int firstHeight,
	int firstWidth, int firstRowSize, const float* second, int secondWidth, int secondRowSize,
	float* result, int resultRowSize )
{
	ASSERT_EXPR( firstWidth <= firstRowSize );
	ASSERT_EXPR( secondWidth <= secondRowSize );
	ASSERT_EXPR( secondWidth <= resultRowSize );

	if( customSgemmFunction != nullptr ) {
		nullify( result, firstHeight, secondWidth, resultRowSize );
		customSgemmFunction( false, false, this, first, firstRowSize, second, secondRowSize,
			result, resultRowSize, firstHeight, secondWidth, firstWidth );
	} else {
#ifdef NEOML_USE_MKL
		cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, firstHeight, secondWidth, firstWidth,
			1, first, firstRowSize, second, secondRowSize, 0, result, resultRowSize );
#else
		nullify( result, firstHeight, secondWidth, resultRowSize );
		MultiplyMatrix<false, false, CTmpMemoryHandler>( this, CpuInfo, first, firstRowSize, second, secondRowSize,
			result, resultRowSize, firstHeight, secondWidth, firstWidth );
#endif
	}
}

void CCpuMathEngine::multiplyMatrixByMatrixAndAdd( const float* first, int firstHeight,
	int firstWidth, int firstRowSize, const float* second, int secondWidth, int secondRowSize,
	float* result, int resultRowSize )
{
	ASSERT_EXPR( firstWidth <= firstRowSize );
	ASSERT_EXPR( secondWidth <= resultRowSize );

	if( customSgemmFunction != nullptr ) {
		customSgemmFunction( false, false, this, first, firstRowSize, second, secondRowSize,
			result, resultRowSize, firstHeight, secondWidth, firstWidth );
	} else {
#ifdef NEOML_USE_MKL
		cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, firstHeight, secondWidth, firstWidth,
			1, first, firstRowSize, second, secondRowSize, 1, result, resultRowSize );
#else
		MultiplyMatrix<false, false, CTmpMemoryHandler>( this, CpuInfo, first, firstRowSize, second, secondRowSize,
			result, resultRowSize, firstHeight, secondWidth, firstWidth );
#endif
	}
}

void CCpuMathEngine::multiplyMatrixByTransposedMatrix(const float* first, int firstHeight,
	int firstWidth, int firstRowSize, const float* second, int secondHeight, int secondRowSize,
	float* result, int resultRowSize)
{
	ASSERT_EXPR(firstWidth <= firstRowSize);
	ASSERT_EXPR(firstWidth <= secondRowSize);

	if( customSgemmFunction != nullptr ) {
		nullify( result, firstHeight, secondHeight, resultRowSize );
		customSgemmFunction( false, true, this, first, firstRowSize, second, secondRowSize,
			result, resultRowSize, firstHeight, secondHeight, firstWidth );
	} else {
#ifdef NEOML_USE_MKL
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, firstHeight, secondHeight, firstWidth,
			1, first, firstRowSize, second, secondRowSize, 0, result, resultRowSize);
#else
		nullify( result, firstHeight, secondHeight, resultRowSize );
		MultiplyMatrix<false, true, CTmpMemoryHandler>( this, CpuInfo, first, firstRowSize, second, secondRowSize,
			result, resultRowSize, firstHeight, secondHeight, firstWidth );
#endif
	}
}

void CCpuMathEngine::multiplyMatrixByTransposedMatrixAndAdd( const float* first, int firstHeight,
	int firstWidth, int firstRowSize, const float* second, int secondHeight, int secondRowSize,
	float* result, int resultRowSize )
{
	if( customSgemmFunction != nullptr ) {
		customSgemmFunction( false, true, this, first, firstRowSize, second, secondRowSize,
			result, resultRowSize, firstHeight, secondHeight, firstWidth );
	} else  {
#ifdef NEOML_USE_MKL
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, firstHeight, secondHeight, firstWidth,
			1, first, firstRowSize, second, secondRowSize, 1, result, resultRowSize);
#else
		MultiplyMatrix<false, true, CTmpMemoryHandler>( this, CpuInfo, first, firstRowSize, second, secondRowSize,
			result, resultRowSize, firstHeight, secondHeight, firstWidth );
#endif
	}
}

// result = first * T(second). The result size is firstHeight * secondHeight:
void CCpuMathEngine::MultiplySparseMatrixByTransposedMatrix( int firstHeight, int firstWidth, int secondHeight,
	const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstDesc.Rows.GetMathEngine() == this );
	ASSERT_EXPR( firstDesc.Columns.GetMathEngine() == this );
	ASSERT_EXPR( firstDesc.Values.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	int* firstRows = GetRaw( firstDesc.Rows );
	int* firstColumns = GetRaw( firstDesc.Columns );
	float* values = GetRaw( firstDesc.Values );
	const float* second = GetRaw( secondHandle );
	float* res = GetRaw( resultHandle );

#ifdef NEOML_USE_MKL
	sparse_matrix_t sparseMatrix;

	ASSERT_EXPR( mkl_sparse_s_create_csr( &sparseMatrix, SPARSE_INDEX_BASE_ZERO, firstHeight, firstWidth, firstRows,
		firstRows + 1, firstColumns, values ) ==  SPARSE_STATUS_SUCCESS );

	matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;

	ASSERT_EXPR( mkl_sparse_s_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.f, sparseMatrix, descr, SPARSE_LAYOUT_COLUMN_MAJOR,
		second, secondHeight, firstWidth, 0, res, firstHeight ) == SPARSE_STATUS_SUCCESS );
	mkl_simatcopy( 'r', 't', secondHeight, firstHeight, 1, res, firstHeight, secondHeight );

	ASSERT_EXPR( mkl_sparse_destroy( sparseMatrix ) == SPARSE_STATUS_SUCCESS );
#else
	for( int col = 0; col < secondHeight; ++col ) {
		float* result = res;
		for( int row = 0; row < firstHeight; ++row ) {
			float resultVal = 0;
			for( int ind = firstRows[row]; ind < firstRows[row + 1]; ++ind ) {
				resultVal += values[ind] * second[firstColumns[ind]];
			}
			result[col] = resultVal;
			result += secondHeight;
		}
		second += firstWidth;
	}
#endif
}

// result = result + T(first) * second. The result size is firstWidth * secondWidth:
void CCpuMathEngine::MultiplyTransposedMatrixBySparseMatrixAndAdd( int firstHeight, int firstWidth, int secondWidth,
	const CConstFloatHandle& firstHandle, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondDesc.Rows.GetMathEngine() == this );
	ASSERT_EXPR( secondDesc.Columns.GetMathEngine() == this );
	ASSERT_EXPR( secondDesc.Values.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	int* secondRows = GetRaw( secondDesc.Rows );
	int* secondColumns = GetRaw( secondDesc.Columns );
	float* secondValues = GetRaw( secondDesc.Values );
	float* result = GetRaw( resultHandle );

#ifdef NEOML_USE_MKL
	CFloatHandleStackVar transposedFirst( mathEngine(), firstHeight * firstWidth );
	mkl_somatcopy( 'r', 't', firstHeight, firstWidth, 1, first, firstWidth,
		GetRaw( transposedFirst.GetHandle() ), firstHeight );
	
	sparse_matrix_t sparseMatrix;

	ASSERT_EXPR( mkl_sparse_s_create_csr( &sparseMatrix, SPARSE_INDEX_BASE_ZERO, firstHeight, secondWidth, secondRows,
		secondRows + 1, secondColumns, secondValues ) == SPARSE_STATUS_SUCCESS );

	matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;

	ASSERT_EXPR( mkl_sparse_s_mm( SPARSE_OPERATION_TRANSPOSE, 1.f, sparseMatrix, descr, SPARSE_LAYOUT_COLUMN_MAJOR,
		GetRaw( transposedFirst.GetHandle() ), firstWidth, firstHeight, 1, result, secondWidth ) == SPARSE_STATUS_SUCCESS );

	ASSERT_EXPR( mkl_sparse_destroy( sparseMatrix ) == SPARSE_STATUS_SUCCESS );
#else
	for( int row = 0; row < firstHeight; ++row ) {
		for( int ind = secondRows[row]; ind < secondRows[row + 1]; ++ind ) {
			for( int col = 0; col < firstWidth; ++col ) {
				result[col * secondWidth + secondColumns[ind]] += first[col] * secondValues[ind];
			}
		}
		first += firstWidth;
	}
#endif
}

void CCpuMathEngine::multiplyTransposedMatrixByMatrix(const float* first, int firstHeight,
	int firstWidth, const float* second, int secondWidth,
	float* result)
{
	if( customSgemmFunction != nullptr ) {
		auto firstRowSize = firstWidth;
		auto secondRowSize = secondWidth;
		auto resultRowSize = secondWidth;
		nullify( result, firstWidth, secondWidth );
		customSgemmFunction( true, false, this, first, firstRowSize, second, secondRowSize,
					 result, resultRowSize, firstWidth, secondWidth, firstHeight );
	} else {
#ifdef NEOML_USE_MKL
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, firstWidth, secondWidth, firstHeight,
			1, first, firstWidth, second, secondWidth, 0, result, secondWidth);
#else
		auto firstRowSize = firstWidth;
		auto secondRowSize = secondWidth;
		auto resultRowSize = secondWidth;
		nullify( result, firstWidth, secondWidth );
		MultiplyMatrix<true, false, CTmpMemoryHandler>( this, CpuInfo, first, firstRowSize, second, secondRowSize,
			result, resultRowSize, firstWidth, secondWidth, firstHeight );
#endif
	}
}

void CCpuMathEngine::multiplyTransposedMatrixByMatrixAndAdd(const float* first,
	int firstHeight, int firstWidth, int firstRowSize,
	const float* second, int secondWidth, int secondRowSize,
	float* result, int resultRowSize)
{
	ASSERT_EXPR(firstWidth <= firstRowSize);
	ASSERT_EXPR(secondWidth <= secondRowSize);
	ASSERT_EXPR(secondWidth <= resultRowSize);
	if( customSgemmFunction != nullptr ) {
		customSgemmFunction( true, false, this, first, firstRowSize, second, secondRowSize,
					 result, resultRowSize, firstWidth, secondWidth, firstHeight );
	} else {
#ifdef NEOML_USE_MKL
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, firstWidth, secondWidth, firstHeight,
			1, first, firstRowSize, second, secondRowSize, 1, result, resultRowSize);
#else
		MultiplyMatrix<true, false, CTmpMemoryHandler>( this, CpuInfo, first, firstRowSize, second, secondRowSize,
			result, resultRowSize, firstWidth, secondWidth, firstHeight );
#endif
	}
}

} // namespace NeoML

#endif // NEOML_USE_SSE
