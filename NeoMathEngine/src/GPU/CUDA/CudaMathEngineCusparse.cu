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
#include <CudaAssert.h>
#include <CudaDevice.h>
#include <CudaCommon.h>
#include <CusparseFunctions.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>

namespace NeoML {

// result = first * T(second). The result size is firstHeight * secondHeight:
void CCudaMathEngine::MultiplySparseMatrixByTransposedMatrix( int firstHeight, int firstWidth, int secondHeight,
	const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	CFloatHandleStackVar tResult( mathEngine(), firstHeight * secondHeight );
	CFloatHandle tResultPtr = tResult.GetHandle();

	cusparseSpMatDescr_t firstCuDesc = 0;
	int* firstRows = GetRaw( firstDesc.Rows );
	int* firstColumns = GetRaw( firstDesc.Columns );
	float* firstValues = GetRaw( firstDesc.Values );

	ASSERT_CUSPARSE( cusparse->CreateCsr( &firstCuDesc, firstHeight, firstWidth, firstDesc.ElementCount,
		firstRows, firstColumns, firstValues, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
		CUDA_R_32F ) );

	float* secondValues = const_cast<float*>( GetRaw( secondHandle ) );
	cusparseDnMatDescr_t secondDesc = 0;
	ASSERT_CUSPARSE( cusparse->CreateDnMat( &secondDesc, firstWidth, secondHeight, firstWidth,
		secondValues, CUDA_R_32F, CUSPARSE_ORDER_COL ) );
	
	cusparseDnMatDescr_t resultDesc = 0;
	float* resultValues = GetRaw( tResultPtr );
	ASSERT_CUSPARSE( cusparse->CreateDnMat( &resultDesc, firstHeight, secondHeight, firstHeight,
		resultValues, CUDA_R_32F, CUSPARSE_ORDER_COL ) );

	float alpha = 1.0;
	float beta = 0.0;

	size_t bufferSize = 0;
	ASSERT_CUSPARSE( cusparse->SpMM_bufferSize( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		CUSPARSE_OPERATION_NON_TRANSPOSE, static_cast<void*>( &alpha ), firstCuDesc, secondDesc, static_cast<void*>( &beta ),
		resultDesc, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize ) );
	
	CFloatHandleStackVar buffer( mathEngine(), bufferSize / sizeof( float ) );
	ASSERT_CUSPARSE( cusparse->SpMM( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, firstCuDesc, secondDesc, &beta, resultDesc, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, GetRaw( buffer ) ) );

	ASSERT_CUSPARSE( cusparse->DestroyDnMat( resultDesc ) );
	ASSERT_CUSPARSE( cusparse->DestroyDnMat( secondDesc ) );
	ASSERT_CUSPARSE( cusparse->DestroySpMat( firstCuDesc ) );

	TransposeMatrix( 1, tResultPtr, secondHeight, 1, firstHeight, 1, resultHandle, static_cast<int>( tResult.Size() ) );
}

void CCudaMathEngine::MultiplyTransposedMatrixBySparseMatrix( int firstHeight, int firstWidth, int secondWidth,
	const CConstFloatHandle& firstHandle, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle,
	bool isTransposedSparse )
{
	ASSERT_EXPR( false );
}

// result = result + T(first) * second. The result size is firstWidth * secondWidth:
void CCudaMathEngine::MultiplyTransposedMatrixBySparseMatrixAndAdd( int firstHeight, int firstWidth, int secondWidth,
	const CConstFloatHandle& first, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( first.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	// C = A * B = T( T(B) * A )

	// Transpose first
	CFloatHandleStackVar tFirst( mathEngine(), firstHeight * firstWidth );
	CFloatHandle tFirstPtr = tFirst.GetHandle();
	TransposeMatrix( 1, first, firstHeight, 1, firstWidth, 1, tFirstPtr, static_cast<int>( tFirst.Size() ) );

	cusparseDnMatDescr_t tFirstDesc = 0;
	void* firstValues = GetRaw( tFirst );
	ASSERT_CUSPARSE( cusparse->CreateDnMat( &tFirstDesc, firstHeight, firstWidth, firstHeight,
		firstValues, CUDA_R_32F, CUSPARSE_ORDER_COL ) );
	
	cusparseSpMatDescr_t secondCuDesc = 0;
	int* secondtRows = GetRaw( secondDesc.Rows );
	int* secondColumns = GetRaw( secondDesc.Columns );
	float* secondValues = GetRaw( secondDesc.Values );
	ASSERT_CUSPARSE( cusparse->CreateCsr( &secondCuDesc, firstHeight, secondWidth, secondDesc.ElementCount,
		secondtRows, secondColumns, secondValues, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
		CUDA_R_32F ) );

	cusparseDnMatDescr_t resultDesc = 0;
	float* resultValues = GetRaw( resultHandle );
	ASSERT_CUSPARSE( cusparse->CreateDnMat( &resultDesc, secondWidth, firstWidth, secondWidth,
		resultValues, CUDA_R_32F, CUSPARSE_ORDER_COL ) );
	float alpha = 1.0;
	float beta = 1.0;

	size_t bufferSize = 0;
	ASSERT_CUSPARSE( cusparse->SpMM_bufferSize( cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, secondCuDesc, tFirstDesc, &beta, resultDesc, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize ) );

	CFloatHandleStackVar buffer( mathEngine(), bufferSize / sizeof( float ) );
	ASSERT_CUSPARSE( cusparse->SpMM( cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, secondCuDesc, tFirstDesc, &beta, resultDesc, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, GetRaw( buffer ) ) );

	ASSERT_CUSPARSE( cusparse->DestroyDnMat( resultDesc ) );
	ASSERT_CUSPARSE( cusparse->DestroySpMat( secondCuDesc ) );
	ASSERT_CUSPARSE( cusparse->DestroyDnMat( tFirstDesc ) );
}

void CCudaMathEngine::MultiplySparseMatrixByMatrix( int firstHeight, int firstWidth, int secondWidth,
	const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( false );
}

void CCudaMathEngine::MultiplyTransposedSparseMatrixByMatrix( int firstHeight, int firstWidth, int secondWidth,
	const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
