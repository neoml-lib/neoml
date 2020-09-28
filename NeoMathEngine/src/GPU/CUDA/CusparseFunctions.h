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

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <cusparse.h>

namespace NeoML {

// The functions of the cusparse library used by the CUDA implementation of MathEngine
struct CCusparse {
	// typedef for convenience
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseCreate ) ( cusparseHandle_t *handle );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseDestroy ) ( cusparseHandle_t handle );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseSetStream ) ( cusparseHandle_t handle, cudaStream_t streamId );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseCreateMatDescr ) ( cusparseMatDescr_t *descrA );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseDestroyMatDescr ) ( cusparseMatDescr_t descrA );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseSetMatType ) ( cusparseMatDescr_t descrA, cusparseMatrixType_t type );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseSetMatIndexBase ) ( cusparseMatDescr_t descrA, cusparseIndexBase_t base );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseScsrmm ) ( cusparseHandle_t handle, cusparseOperation_t transA,
		int m, int n, int k, int nnz, const float *alpha, const cusparseMatDescr_t descrA, const float  *csrSortedValA,
		const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *B, int ldb, const float *beta, float *C,
		int ldc );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseScsrmm2 ) ( cusparseHandle_t handle, cusparseOperation_t transA,
		cusparseOperation_t transB, int m, int n, int k, int nnz, const float *alpha, const cusparseMatDescr_t descrA,
		const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *B, int ldb,
		const float *beta, float *C, int ldc );
	typedef const char*( CUSPARSEAPI *TCusparseGetErrorString ) ( cusparseStatus_t status );

	TCusparseCreate Create;
	TCusparseDestroy Destroy;
	TCusparseSetStream SetStream;
	TCusparseCreateMatDescr CreateMatDescr;
	TCusparseDestroyMatDescr DestroyMatDescr;
	TCusparseSetMatType SetMatType;
	TCusparseSetMatIndexBase SetMatIndexBase;
	TCusparseScsrmm Scsrmm;
	TCusparseScsrmm2 Scsrmm2;
	TCusparseGetErrorString GetErrorString;
};

} // namespace NeoML

#endif // NEOML_USE_CUDA
