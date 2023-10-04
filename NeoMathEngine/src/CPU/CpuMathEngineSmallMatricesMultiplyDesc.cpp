/* Copyright © 2023 ABBYY

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

#include <CpuMathEngine.h>
#include <MemoryHandleInternal.h>
#include <CpuMathEngineSmallMatricesMultiplyDesc.h>

#ifdef NEOML_USE_MKL
#if FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
#include <mkl.h>
#else
#error Unknown platform
#endif
#endif // NEOML_USE_MKL

//-------------------------------------------------------------------------------------------------------------------------

namespace NeoML {

inline CSmallMatricesMultiplyDesc* CCpuMathEngine::InitSmallMatricesMultiplyDesc(
	int firstHeight, int firstWidth, int secondWidth, int secondRowSize, int resultWidth,
	bool resultAdd, bool trans1, bool trans2 ) const
{
	return new CCpuSmallMatricesMultiplyDesc(
		firstHeight, firstWidth, secondWidth, secondRowSize, resultWidth,
		resultAdd, trans1, trans2 );
}

inline bool CCpuMathEngine::smallMatricesMultiply( const CSmallMatricesMultiplyDesc* desc,
	const float* first, const float* second, float* result ) const
{
	PRESUME_EXPR( first != nullptr );
	PRESUME_EXPR( second != nullptr );
	PRESUME_EXPR( result != nullptr );

#if FINE_BIT( FINE_64_BIT ) && defined( NEOML_USE_MKL )

	if( desc != nullptr ) {
		PRESUME_EXPR( dynamic_cast<const CCpuSmallMatricesMultiplyDesc*>( desc ) != nullptr );
		const CCpuSmallMatricesMultiplyDesc* mulDesk = static_cast<const CCpuSmallMatricesMultiplyDesc*>( desc );

		if( mulDesk->MklJitter != nullptr ) {
			PRESUME_EXPR( mulDesk->MklJitter != nullptr && mulDesk->MklKernel != nullptr );

			// Repeatedly execute the SGemm Kernel
			mulDesk->MklKernel( mulDesk->MklJitter, ( float* )first, ( float* )second, result );
			return true;
		}
	}

#else  // !FINE_64_BIT || !NEOML_USE_MKL
	( void )desc;
	( void )first;
	( void )second;
	( void )result;
#endif // !FINE_64_BIT || !NEOML_USE_MKL

	return false;
}

//-------------------------------------------------------------------------------------------------------------------------

CCpuSmallMatricesMultiplyDesc::CCpuSmallMatricesMultiplyDesc(
	int firstHeight, int firstWidth, int secondWidth, int secondRowSize, int resultWidth,
	bool resultAdd, bool trans1, bool trans2 )
{
	ASSERT_EXPR( firstHeight > 0 );
	ASSERT_EXPR( firstWidth > 0 );
	ASSERT_EXPR( secondWidth > 0 );
	ASSERT_EXPR( secondRowSize > 0 );
	ASSERT_EXPR( resultWidth > 0 );

#if FINE_BIT( FINE_64_BIT ) && defined( NEOML_USE_MKL )

	// Empirical upper limit of a matrix size to effective JIT optimization
	static constexpr int MaxMatrixSize = 128;

	if( firstHeight <= MaxMatrixSize
		&& ( trans2 || ( firstWidth <= MaxMatrixSize && secondWidth <= MaxMatrixSize ) )
		&& resultWidth <= MaxMatrixSize )
	{
		// Create jitter handle and generate SGemm Kernel
		auto status = mkl_jit_create_sgemm(
			&MklJitter,
			MKL_ROW_MAJOR,
			( trans1 ? MKL_TRANS : MKL_NOTRANS ),
			( trans2 ? MKL_TRANS : MKL_NOTRANS ),
			/*m*/( trans1 ? firstWidth : firstHeight ),
			/*n*/( trans2 ? resultWidth : secondWidth ),
			/*k*/( trans1 ? firstHeight : firstWidth ),
			1.f,
			firstWidth,
			secondRowSize,
			( resultAdd ? 1.f : 0.f ),
			resultWidth );
		ASSERT_EXPR( status != MKL_JIT_ERROR );

		// Get kernel associated with jitter handle
		MklKernel = mkl_jit_get_sgemm_ptr( MklJitter );
		ASSERT_EXPR( MklKernel != nullptr );
	}

#else  // !FINE_64_BIT || !NEOML_USE_MKL
	( void )firstHeight;
	( void )firstWidth;
	( void )secondWidth;
	( void )secondRowSize;
	( void )resultWidth;
	( void )resultAdd;
	( void )trans1;
	( void )trans2;
#endif // !FINE_64_BIT || !NEOML_USE_MKL
}

CCpuSmallMatricesMultiplyDesc::~CCpuSmallMatricesMultiplyDesc()
{
	if( MklJitter ) {
#ifdef NEOML_USE_MKL
		// Destroy the created jitter
		mkl_jit_destroy( MklJitter );
#endif // NEOML_USE_MKL
	}
}

} // namespace NeoML

