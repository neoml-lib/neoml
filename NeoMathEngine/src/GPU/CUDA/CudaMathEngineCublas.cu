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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <CudaMathEngine.h>
#include <CudaCommon.h>
#include <CudaAssert.h>
#include <CublasFunctions.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>

#include <cuda_runtime_api.h>

namespace NeoML {

void CCudaMathEngine::VectorDotProduct(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
    int vectorSize, const CFloatHandle& resultHandle)
{
    ASSERT_EXPR( firstHandle.GetMathEngine() == this );
    ASSERT_EXPR( secondHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	ASSERT_CUBLAS( cublas->Sdot( cublasHandle, vectorSize, GetRaw( firstHandle ), 1,
		GetRaw( secondHandle ), 1, GetRaw( resultHandle ) ) );
}

void CCudaMathEngine::VectorMultiplyAndAdd( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );

	const float* first = GetRaw( firstHandle );
	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );
	const float* mult = GetRaw( multHandle );

	if( result != first ) {
		ASSERT_CUDA( cudaMemcpy( result, first, vectorSize * sizeof( float ), cudaMemcpyDeviceToDevice ) );
	}
	ASSERT_CUBLAS( cublas->Saxpy( cublasHandle, vectorSize, mult, second, 1, result, 1 ) );
}

void CCudaMathEngine::MultiplyMatrixByTransposedMatrix( const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
	const CFloatHandle& resultHandle, int resultRowSize, int )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	ASSERT_CUBLAS( cublas->Sgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, secondHeight, firstHeight, firstWidth,
		CCudaConst::One, GetRaw( secondHandle ), secondRowSize, GetRaw( firstHandle ), firstRowSize, CCudaConst::Zero,
		GetRaw( resultHandle ), resultRowSize ) );
}

void CCudaMathEngine::MultiplyMatrixByTransposedMatrix( int batchSize, const CConstFloatHandle& firstHandle,
	int firstHeight, int firstWidth, const CConstFloatHandle& secondHandle, int secondHeight,
	const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	ASSERT_CUBLAS( cublas->SgemmStridedBatched( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, secondHeight,
		firstHeight, firstWidth, CCudaConst::One, GetRaw( secondHandle ), firstWidth, firstWidth * secondHeight,
		GetRaw( firstHandle ), firstWidth, firstHeight * firstWidth, CCudaConst::Zero, GetRaw( resultHandle ),
		secondHeight, secondHeight * firstHeight, batchSize ) );
}

void CCudaMathEngine::MultiplyTransposedMatrixByMatrixAndAdd( const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondWidth, int secondRowSize,
	const CFloatHandle& resultHandle, int resultRowSize, int )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	ASSERT_CUBLAS( cublas->Sgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, secondWidth, firstWidth, firstHeight,
		CCudaConst::One, GetRaw( secondHandle ), secondRowSize, GetRaw( firstHandle ), firstRowSize, CCudaConst::One,
		GetRaw( resultHandle ), resultRowSize ) );
}

void CCudaMathEngine::MultiplyTransposedMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	ASSERT_CUBLAS( cublas->SgemmStridedBatched( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, secondWidth, firstWidth,
		firstHeight, CCudaConst::One, GetRaw(secondHandle), secondWidth, firstHeight * secondWidth, GetRaw(firstHandle),
		firstWidth, firstHeight * firstWidth, CCudaConst::Zero, GetRaw(resultHandle), secondWidth, firstWidth * secondWidth,
		batchSize ) );
}

void CCudaMathEngine::MultiplyMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth,
	const CFloatHandle& resultHandle, int )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	if( batchSize == 1 ) {
		ASSERT_CUBLAS( cublas->Sgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, secondWidth, firstHeight, firstWidth,
			CCudaConst::One, GetRaw( secondHandle ), secondWidth, GetRaw( firstHandle ), firstWidth, CCudaConst::Zero,
			GetRaw( resultHandle ), secondWidth ) );
	} else {
		ASSERT_CUBLAS( cublas->SgemmStridedBatched( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, secondWidth, firstHeight, firstWidth,
			CCudaConst::One, GetRaw( secondHandle ), secondWidth, firstWidth * secondWidth, GetRaw( firstHandle ), firstWidth,
			firstHeight * firstWidth, CCudaConst::Zero, GetRaw( resultHandle ), secondWidth, secondWidth * firstHeight, batchSize ) );
	}
}

void CCudaMathEngine::multiplyMatrixByTransposedMatrixAndAdd(const CConstFloatHandle& firstHandle,
	int firstHeight, int firstWidth, int firstRowSize,
	const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
	const CFloatHandle& resultHandle, int resultRowSize)
{
	ASSERT_CUBLAS( cublas->Sgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, secondHeight, firstHeight, firstWidth,
		CCudaConst::One, GetRaw( secondHandle ), secondRowSize, GetRaw( firstHandle ), firstRowSize, CCudaConst::One,
		GetRaw( resultHandle ), resultRowSize ) );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
