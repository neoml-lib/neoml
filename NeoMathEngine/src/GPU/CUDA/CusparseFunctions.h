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
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseCreateCsr ) ( cusparseSpMatDescr_t* spMatDescr, int64_t rows,
		int64_t cols, int64_t nnz, void* csrRowOffsets, void* csrColInd,  void* csrValues,
		cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase,
		cudaDataType valueType );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseDestroySpMat ) ( cusparseSpMatDescr_t spMatDescr );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseCreateDnMat ) ( cusparseDnMatDescr_t* dnMatDescr, int64_t rows,
		int64_t cols, int64_t ld, void* values, cudaDataType valueType, cusparseOrder_t order );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseDestroyDnMat ) ( cusparseDnMatDescr_t dnMatDescr );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseSpMM_bufferSize ) ( cusparseHandle_t handle, cusparseOperation_t opA,
		cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, const void* beta,
		cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, size_t* bufferSizer );
	typedef cusparseStatus_t( CUSPARSEAPI *TCusparseSpMM ) ( cusparseHandle_t handle, cusparseOperation_t opA,
		cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, const void* beta,
		cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer );

	TCusparseCreate Create;
	TCusparseDestroy Destroy;
	TCusparseSetStream SetStream;
	TCusparseCreateCsr CreateCsr;
	TCusparseDestroySpMat DestroySpMat;
	TCusparseCreateDnMat CreateDnMat;
	TCusparseDestroyDnMat DestroyDnMat;
	TCusparseSpMM_bufferSize SpMM_bufferSize;
	TCusparseSpMM SpMM;
};

} // namespace NeoML

#endif // NEOML_USE_CUDA
