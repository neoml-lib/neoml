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

#include <cublas_v2.h>

namespace NeoML {

// The cublas functions used in CUDA implementation of the MathEngine
struct CCublas {
	// typedef for convenience
	typedef cublasStatus_t( CUBLASWINAPI *TCublasCreate ) ( cublasHandle_t *handle );
	typedef cublasStatus_t( CUBLASWINAPI *TCublasDestroy ) ( cublasHandle_t handle );
	typedef cublasStatus_t( CUBLASWINAPI *TCublasSetStream ) ( cublasHandle_t handle, cudaStream_t streamId );
	typedef cublasStatus_t( CUBLASWINAPI *TCublasSetMathMode ) ( cublasHandle_t handle, cublasMath_t mode );
	typedef cublasStatus_t( CUBLASWINAPI *TCublasSetPointerMode ) ( cublasHandle_t handle, cublasPointerMode_t mode );
	typedef cublasStatus_t( CUBLASWINAPI *TCublasSetAtomicsMode ) ( cublasHandle_t handle, cublasAtomicsMode_t mode );
	typedef cublasStatus_t( CUBLASWINAPI *TCublasSdot ) ( cublasHandle_t handle, int n, const float *x, int incx,
		const float *y, int incy, float *result );
	typedef cublasStatus_t( CUBLASWINAPI *TCublasSaxpy ) ( cublasHandle_t handle, int n, const float *alpha,
		const float *x, int incx, float *y, int incy );
	typedef cublasStatus_t( CUBLASWINAPI *TCublasSgemm ) ( cublasHandle_t handle, cublasOperation_t transa,
		cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B,
		int ldb, const float *beta, float *C, int ldc );
	typedef cublasStatus_t( CUBLASWINAPI *TCublasSgemmStridedBatched ) ( cublasHandle_t handle, cublasOperation_t transa,
		cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA,
		const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC,
		int batchCount );

	TCublasCreate Create;
	TCublasDestroy Destroy;
	TCublasSetStream SetStream;
	TCublasSetMathMode SetMathMode;
	TCublasSetPointerMode SetPointerMode;
	TCublasSetAtomicsMode SetAtomicsMode;
	TCublasSdot Sdot;
	TCublasSaxpy Saxpy;
	TCublasSgemm Sgemm;
	TCublasSgemmStridedBatched SgemmStridedBatched;
};

} // namespace NeoML

#endif // NEOML_USE_CUDA
