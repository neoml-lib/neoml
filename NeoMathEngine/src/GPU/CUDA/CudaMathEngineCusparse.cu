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

	CFloatHandleStackVar tResult( mathEngine(), firstHeight * secondHeight );
	CFloatHandle tResultPtr = tResult.GetHandle();

	cusparseMatDescr_t description = 0;
	ASSERT_CUSPARSE( cusparse->CreateMatDescr( &description ) );
	ASSERT_CUSPARSE( cusparse->SetMatType( description, CUSPARSE_MATRIX_TYPE_GENERAL ) );
	ASSERT_CUSPARSE( cusparse->SetMatIndexBase( description, CUSPARSE_INDEX_BASE_ZERO ) );

	const int* firstRows = GetRaw( firstDesc.Rows );
	const float* firstValues = GetRaw( firstDesc.Values );
	const int* firstColumns = GetRaw( firstDesc.Columns );
	float alpha = 1.0;
	float beta = 0.0;

	ASSERT_CUSPARSE( cusparse->Scsrmm( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, firstHeight, secondHeight, firstWidth,
		firstDesc.ElementCount, &alpha, description, firstValues, firstRows, firstColumns, GetRaw( secondHandle ), firstWidth,
		&beta, GetRaw( tResultPtr ), firstHeight ) );

	ASSERT_CUSPARSE( cusparse->DestroyMatDescr( description ) );

	TransposeMatrix( 1, tResultPtr, secondHeight, 1, firstHeight, 1, resultHandle, static_cast<int>( tResult.Size() ) );
}

// result = result + T(first) * second. The result size is firstWidth * secondWidth:
void CCudaMathEngine::MultiplyTransposedMatrixBySparseMatrixAndAdd( int firstHeight, int firstWidth, int secondWidth,
	const CConstFloatHandle& first, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( first.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	// C = A * B = T( T(B) * A )

	// Transpose first
	CFloatHandleStackVar tFirst( mathEngine(), firstHeight * firstWidth );
	CFloatHandle tFirstPtr = tFirst.GetHandle();
	TransposeMatrix( 1, first, firstHeight, 1, firstWidth, 1, tFirstPtr, static_cast<int>( tFirst.Size() ) );

	cusparseMatDescr_t description = 0;
	ASSERT_CUSPARSE( cusparse->CreateMatDescr( &description ) );
	ASSERT_CUSPARSE( cusparse->SetMatType( description, CUSPARSE_MATRIX_TYPE_GENERAL ) );
	ASSERT_CUSPARSE( cusparse->SetMatIndexBase( description, CUSPARSE_INDEX_BASE_ZERO ) );

	// Calculate T( T(B) * A ):
	const int* secondRows = GetRaw( secondDesc.Rows );
	const float* secondValues = GetRaw( secondDesc.Values );
	const int* secondColumns = GetRaw( secondDesc.Columns );
	float alpha = 1.0;
	float beta = 1.0;

	ASSERT_CUSPARSE( cusparse->Scsrmm2( cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		firstHeight, firstWidth, secondWidth, secondDesc.ElementCount, &alpha, description, secondValues, secondRows,
		secondColumns, GetRaw( tFirstPtr ), firstHeight, &beta, GetRaw( resultHandle ), secondWidth ) );

	ASSERT_CUSPARSE( cusparse->DestroyMatDescr( description ) );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
