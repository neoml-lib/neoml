/* Copyright © 2017-2023 ABBYY

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
#include <CpuMathEnginePrivate.h>

#ifdef NEOML_USE_MKL
#if FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
#include <mkl.h>
#else
#error Unknown platform
#endif
#else  // !NEOML_USE_MKL
#include <CPUInfo.h>
#include <MatrixMultiplyingInterleavedCommon/MatrixMultiplying.h>

// Cache sizes
// Find the acceptable values or get them from CPU info
static constexpr CCPUInfo CpuInfo( 0x60000, 0x180000, 0x900000 );
#endif // !NEOML_USE_MKL
#include <MatrixMultiplyingInterleavedCommon/CpuMemoryHelper.h>

#ifdef NEOML_USE_MLAS
#include "mlas/inc/mlas.h"
#endif

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
#if defined( NEOML_USE_MKL ) && defined( NEOML_USE_MLAS )
		if( CCPUInfo::IsAMD ) {
			MlasGemm( MlasNoTrans, MlasNoTrans, static_cast<size_t>( firstHeight ), static_cast<size_t>( secondWidth ),
				static_cast<size_t>( firstWidth ), 1, first, static_cast<size_t>( firstRowSize ), second,
				static_cast<size_t>( secondRowSize ), 0, result, static_cast<size_t>( resultRowSize ), nullptr );
		} else {
			cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, firstHeight, secondWidth, firstWidth,
				1, first, firstRowSize, second, secondRowSize, 0, result, resultRowSize );
		}
#elif defined( NEOML_USE_MKL )
		cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, firstHeight, secondWidth, firstWidth,
			1, first, firstRowSize, second, secondRowSize, 0, result, resultRowSize );
#elif defined( NEOML_USE_MLAS )
		MlasGemm( MlasNoTrans, MlasNoTrans, static_cast<size_t>( firstHeight ), static_cast<size_t>( secondWidth ),
			static_cast<size_t>( firstWidth ), 1, first, static_cast<size_t>( firstRowSize ), second,
			static_cast<size_t>( secondRowSize ), 0, result, static_cast<size_t>( resultRowSize ), nullptr );
#else // !NEOML_USE_MKL && !NEOML_USE_MLAS
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
#if defined( NEOML_USE_MKL ) && defined( NEOML_USE_MLAS )
		if( CCPUInfo::IsAMD ) {
			MlasGemm( MlasNoTrans, MlasNoTrans, static_cast<size_t>( firstHeight ), static_cast<size_t>( secondWidth ),
				static_cast<size_t>( firstWidth ), 1, first, static_cast<size_t>( firstRowSize ), second,
				static_cast<size_t>( secondRowSize ), 1, result, static_cast<size_t>( resultRowSize ), nullptr );
		} else {
			cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, firstHeight, secondWidth, firstWidth,
				1, first, firstRowSize, second, secondRowSize, 1, result, resultRowSize );
		}
#elif defined( NEOML_USE_MKL )
		cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, firstHeight, secondWidth, firstWidth,
			1, first, firstRowSize, second, secondRowSize, 1, result, resultRowSize );
#elif defined( NEOML_USE_MLAS )
		MlasGemm( MlasNoTrans, MlasNoTrans, static_cast<size_t>( firstHeight ), static_cast<size_t>( secondWidth ),
			static_cast<size_t>( firstWidth ), 1, first, static_cast<size_t>( firstRowSize ), second,
			static_cast<size_t>( secondRowSize ), 1, result, static_cast<size_t>( resultRowSize ), nullptr );
#else // !NEOML_USE_MKL && !NEOML_USE_MLAS
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
#if defined( NEOML_USE_MKL ) && defined( NEOML_USE_MLAS )
		if( CCPUInfo::IsAMD ) {
			MlasGemm( MlasNoTrans, MlasTrans, static_cast<size_t>( firstHeight ), static_cast<size_t>( secondHeight ),
				static_cast<size_t>( firstWidth ), 1, first, static_cast<size_t>( firstRowSize ), second,
				static_cast<size_t>( secondRowSize ), 0, result, static_cast<size_t>( resultRowSize ), nullptr );
		} else {
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, firstHeight, secondHeight, firstWidth,
				1, first, firstRowSize, second, secondRowSize, 0, result, resultRowSize);
		}
#elif defined( NEOML_USE_MKL )
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, firstHeight, secondHeight, firstWidth,
			1, first, firstRowSize, second, secondRowSize, 0, result, resultRowSize);
#elif defined( NEOML_USE_MLAS )
		MlasGemm( MlasNoTrans, MlasTrans, static_cast<size_t>( firstHeight ), static_cast<size_t>( secondHeight ),
			static_cast<size_t>( firstWidth ), 1, first, static_cast<size_t>( firstRowSize ), second,
			static_cast<size_t>( secondRowSize ), 0, result, static_cast<size_t>( resultRowSize ), nullptr );
#else // !NEOML_USE_MKL && !NEOML_USE_MLAS
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
	} else {
#if defined( NEOML_USE_MKL ) && defined( NEOML_USE_MLAS )
		if( CCPUInfo::IsAMD ) {
			MlasGemm( MlasNoTrans, MlasTrans, static_cast<size_t>( firstHeight ), static_cast<size_t>( secondHeight ),
				static_cast<size_t>( firstWidth ), 1, first, static_cast<size_t>( firstRowSize ), second,
				static_cast<size_t>( secondRowSize ), 1, result, static_cast<size_t>( resultRowSize ), nullptr );
		} else {
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, firstHeight, secondHeight, firstWidth,
				1, first, firstRowSize, second, secondRowSize, 1, result, resultRowSize);
		}
#elif defined( NEOML_USE_MKL )
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, firstHeight, secondHeight, firstWidth,
			1, first, firstRowSize, second, secondRowSize, 1, result, resultRowSize);
#elif defined( NEOML_USE_MLAS )
		MlasGemm( MlasNoTrans, MlasTrans, static_cast<size_t>( firstHeight ), static_cast<size_t>( secondHeight ),
			static_cast<size_t>( firstWidth ), 1, first, static_cast<size_t>( firstRowSize ), second,
			static_cast<size_t>( secondRowSize ), 1, result, static_cast<size_t>( resultRowSize ), nullptr );
#else  // !NEOML_USE_MKL && !NEOML_USE_MLAS
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
#else  // !NEOML_USE_MKL
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
#endif // !NEOML_USE_MKL
}

// result = T(first) * second, second is transposed if isTransposedSparse
void CCpuMathEngine::MultiplyTransposedMatrixBySparseMatrix( int firstHeight, int firstWidth, int resultWidth,
	const CConstFloatHandle& firstHandle, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle,
	bool isTransposedSparse )
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

	const int sparseRows = isTransposedSparse ? resultWidth : firstHeight;
	const int sparseCols = isTransposedSparse ? firstHeight : resultWidth;
	ASSERT_EXPR( mkl_sparse_s_create_csr( &sparseMatrix, SPARSE_INDEX_BASE_ZERO, sparseRows, sparseCols, secondRows,
		secondRows + 1, secondColumns, secondValues ) == SPARSE_STATUS_SUCCESS );

	matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;

	ASSERT_EXPR( mkl_sparse_s_mm( isTransposedSparse ? SPARSE_OPERATION_NON_TRANSPOSE : SPARSE_OPERATION_TRANSPOSE,
		1.f, sparseMatrix, descr, SPARSE_LAYOUT_COLUMN_MAJOR, GetRaw( transposedFirst.GetHandle() ), firstWidth, firstHeight,
		0.f, result, resultWidth ) == SPARSE_STATUS_SUCCESS );

	ASSERT_EXPR( mkl_sparse_destroy( sparseMatrix ) == SPARSE_STATUS_SUCCESS );
#else  // !NEOML_USE_MKL
	nullify( result, firstWidth, resultWidth );
	if( isTransposedSparse ) {
		for( int row = 0; row < resultWidth; ++row ) {
			for( int ind = secondRows[row]; ind < secondRows[row + 1]; ++ind ) {
				const float* firstPtr = first + secondColumns[ind] * firstWidth;
				for( int col = 0; col < firstWidth; ++col ) {
					result[col * resultWidth + row] += firstPtr[col] * secondValues[ind];
				}
			}
		}
	} else {
		for( int row = 0; row < firstHeight; ++row ) {
			for( int ind = secondRows[row]; ind < secondRows[row + 1]; ++ind ) {
				for( int col = 0; col < firstWidth; ++col ) {
					result[col * resultWidth + secondColumns[ind]] += first[col] * secondValues[ind];
				}
			}
			first += firstWidth;
		}
	}
#endif // !NEOML_USE_MKL
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
#else  // !NEOML_USE_MKL
	for( int row = 0; row < firstHeight; ++row ) {
		for( int ind = secondRows[row]; ind < secondRows[row + 1]; ++ind ) {
			for( int col = 0; col < firstWidth; ++col ) {
				result[col * secondWidth + secondColumns[ind]] += first[col] * secondValues[ind];
			}
		}
		first += firstWidth;
	}
#endif // !NEOML_USE_MKL
}

// result = first * second. The result size is firstHeight * secondWidth:
void CCpuMathEngine::MultiplySparseMatrixByMatrix( int firstHeight, int firstWidth, int secondWidth,
	const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstDesc.Rows.GetMathEngine() == this );
	ASSERT_EXPR( firstDesc.Columns.GetMathEngine() == this );
	ASSERT_EXPR( firstDesc.Values.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( firstWidth > 0 );
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

	ASSERT_EXPR( mkl_sparse_s_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.f, sparseMatrix, descr, SPARSE_LAYOUT_ROW_MAJOR,
		second, secondWidth, secondWidth, 0, res, secondWidth ) == SPARSE_STATUS_SUCCESS );

	ASSERT_EXPR( mkl_sparse_destroy( sparseMatrix ) == SPARSE_STATUS_SUCCESS );
#else  // !NEOML_USE_MKL
	nullify( res, firstHeight, secondWidth );
	for( int row = 0; row < firstHeight; ++row ) {
		float* result = res + row * secondWidth;
		for( int ind = firstRows[row]; ind < firstRows[row + 1]; ++ind ) {
			const float* dense = second + firstColumns[ind] * secondWidth;
			for( int col = 0; col < secondWidth; ++col ) {
				result[col] += values[ind] * dense[col];
			}
		}
	}
#endif // !NEOML_USE_MKL
}

// result = T(first) * second. The result size is firstWidth * secondWidth:
void CCpuMathEngine::MultiplyTransposedSparseMatrixByMatrix( int firstHeight, int firstWidth, int secondWidth,
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

	ASSERT_EXPR( mkl_sparse_s_mm( SPARSE_OPERATION_TRANSPOSE, 1.f, sparseMatrix, descr, SPARSE_LAYOUT_ROW_MAJOR,
		second, secondWidth, secondWidth, 0, res, secondWidth ) == SPARSE_STATUS_SUCCESS );

	ASSERT_EXPR( mkl_sparse_destroy( sparseMatrix ) == SPARSE_STATUS_SUCCESS );
#else  // !NEOML_USE_MKL
	nullify( res, firstWidth, secondWidth );
	for( int row = 0; row < firstHeight; ++row ) {
		for( int ind = firstRows[row]; ind < firstRows[row + 1]; ++ind ) {
			const float* dense = second + row * secondWidth;
			float* resRow = res + firstColumns[ind] * secondWidth;
			for( int col = 0; col < secondWidth; ++col ) {
				resRow[col] += values[ind] * dense[col];
			}
		}
	}
#endif // !NEOML_USE_MKL
}

void CCpuMathEngine::QRFactorization( int height, int width, const CFloatHandle& matrixHandle, const CFloatHandle* qHandle, const CFloatHandle* rHandle,
	bool inplace, bool returnQ, bool returnR )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( returnQ == false || qHandle != nullptr  );
	ASSERT_EXPR( inplace == true || returnR == false || rHandle != nullptr  );
	ASSERT_EXPR( height > 0 );
	ASSERT_EXPR( width > 0 );
	CCpuExecutionScope scope;

#ifdef NEOML_USE_MKL
	float* matrix = GetRaw( matrixHandle );
	float* q = nullptr;
	if( qHandle != nullptr ) {
		ASSERT_EXPR( qHandle->GetMathEngine() == this );
		q = GetRaw( *qHandle );
	}
	float* r = nullptr;
	if( rHandle != nullptr ) {
		ASSERT_EXPR( rHandle->GetMathEngine() == this );
		r = GetRaw( *rHandle );
	}

	CFloatHandleStackVar tempR( mathEngine(), ( returnR || !inplace ) ? height * width : 1 );
	if( inplace ) {
		r = matrix;
	} else {
		if( r == nullptr ) {
			r = GetRaw( tempR.GetHandle() );
		}
		memcpy( r, matrix, height * width * sizeof( float ) );
	}

	const int reflectors = std::min( height, width );
	CFloatHandleStackVar tau( mathEngine(), reflectors );
	LAPACKE_sgeqrf( LAPACK_ROW_MAJOR, height, width, r, width, GetRaw( tau.GetHandle() ) );
	if( returnQ ) {
		float* temp = r;
		if( returnR ) {
			temp = GetRaw( tempR.GetHandle() );
			memcpy( temp, r, height * width * sizeof( float ) );
		}
		LAPACKE_sorgqr( LAPACK_ROW_MAJOR, height, reflectors, reflectors, temp, width, GetRaw( tau.GetHandle() ) );
		for( int i = 0; i < height; i++ ) {
			for( int j = 0; j < reflectors; j++ ) {
				*q++ = temp[j];
			}
			temp += width;
		}
	}
	if( returnR ) {
		float* rPtr = r;
		for( int i = 0; i < height; i++ ) {
			const int diag = std::min( i, width );
			for( int j = 0; j < diag; j++ ) {
				rPtr[j] = 0.f;
			}
			rPtr += width;
		}
	}
#else  // !NEOML_USE_MKL
	ASSERT_EXPR( false );
#endif // !NEOML_USE_MKL
}

void CCpuMathEngine::LUFactorization( int height, int width, const CFloatHandle& matrixHandle )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( height > 0 );
	ASSERT_EXPR( width > 0 );
	CCpuExecutionScope scope;

#ifdef NEOML_USE_MKL
	const int minDim = std::max( 1, std::min( height, width ) );
	CIntHandleStackVar ipivVar( *this, minDim );

	float* matrix = GetRaw( matrixHandle );
	ASSERT_EXPR( LAPACKE_sgetrf( LAPACK_ROW_MAJOR, height, width, matrix, width, GetRaw( ipivVar.GetHandle() ) ) == 0 );
	
	// Fill main diagonal with 1 and upper half with zeroes
	for( int rowIndex = 0; rowIndex < minDim; ++rowIndex ) {
		float* row = matrix + width * rowIndex;
		row[rowIndex] = 1.f;
		if( rowIndex + 1 < width ) {
			vectorFill( row + rowIndex + 1, 0.f, width - rowIndex - 1 );
		}
	}

	// Apply permutation matrix (ipiv in reversed order)
	const int* ipiv = GetRaw(ipivVar.GetHandle());
	for( int rowIndex = minDim - 1; rowIndex >= 0; --rowIndex ) {
		int origIndex = ipiv[rowIndex] - 1;
		if( origIndex != rowIndex ) {
			// Need to swap rows...
			float* row = matrix + rowIndex * width;
			float* origRow = matrix + origIndex * width;
			int toSwap = width;
			while( toSwap >= 4 ) {
				__m128 curr = LoadSse4( row );
				StoreSse4( LoadSse4( origRow ), row );
				StoreSse4( curr, origRow );
				row += 4;
				origRow += 4;
				toSwap -= 4;
			}
			if( toSwap > 0 ) {
				__m128 curr = LoadSse( row, toSwap );
				StoreSse( LoadSse( origRow, toSwap ), row, toSwap );
				StoreSse( curr, origRow, toSwap );
			}
		}
	}

#else  // !NEOML_USE_MKL
	ASSERT_EXPR( false );
#endif // !NEOML_USE_MKL
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
#else  // !NEOML_USE_MKL
		auto firstRowSize = firstWidth;
		auto secondRowSize = secondWidth;
		auto resultRowSize = secondWidth;
		nullify( result, firstWidth, secondWidth );
		MultiplyMatrix<true, false, CTmpMemoryHandler>( this, CpuInfo, first, firstRowSize, second, secondRowSize,
			result, resultRowSize, firstWidth, secondWidth, firstHeight );
#endif // !NEOML_USE_MKL
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
#else  // !NEOML_USE_MKL
		MultiplyMatrix<true, false, CTmpMemoryHandler>( this, CpuInfo, first, firstRowSize, second, secondRowSize,
			result, resultRowSize, firstWidth, secondWidth, firstHeight );
#endif // !NEOML_USE_MKL
	}
}

void CCpuMathEngine::SingularValueDecomposition( const CFloatHandle& a, int height, int width, const CFloatHandle& u, const CFloatHandle& s,
	const CFloatHandle& vt, const CFloatHandle& superb, bool returnLeftVectors, bool returnRightVectors )
{
	ASSERT_EXPR( a.GetMathEngine() == this );
	ASSERT_EXPR( u.GetMathEngine() == this );
	ASSERT_EXPR( s.GetMathEngine() == this );
	ASSERT_EXPR( vt.GetMathEngine() == this );
	ASSERT_EXPR( superb.GetMathEngine() == this );
	ASSERT_EXPR( height > 0 );
	ASSERT_EXPR( width > 0 );

#ifdef NEOML_USE_MKL
	const int ldu = std::min( height, width );
	ASSERT_EXPR( LAPACKE_sgesvd( LAPACK_ROW_MAJOR, returnLeftVectors ? 'S' : 'N', returnRightVectors ? 'S' : 'N',
		height, width, GetRaw( a ), width, GetRaw( s ), GetRaw(u), ldu, GetRaw( vt ), width, GetRaw( superb ) ) == 0 );
#else  // !NEOML_USE_MKL
	( void )returnLeftVectors;
	( void )returnRightVectors;
	ASSERT_EXPR( false );
#endif // !NEOML_USE_MKL
}

} // namespace NeoML

#endif // NEOML_USE_SSE
