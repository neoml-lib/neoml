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

#include <cublas.h>

#define ASSERT_CUDA( expr ) \
	do { \
		int _err_ = ( int ) ( expr ); \
		if( _err_ != cudaSuccess ) { \
			NeoML::IMathEngineExceptionHandler* exceptionHandler = GetMathEngineExceptionHandler(); \
			if( exceptionHandler != nullptr ) { \
				generateAssert( exceptionHandler, cudaGetErrorString( static_cast<cudaError_t>( _err_ ) ), __UNICODEFILE__, __LINE__, _err_ ); \
			} else { \
				throw std::logic_error( #expr ); \
			} \
		} \
	} while(0)

// This macro requires 'CCusparse* cusparse' variable
#define ASSERT_CUSPARSE( expr ) \
	do { \
		int _err_ = ( int ) ( expr ); \
		if( _err_ != CUSPARSE_STATUS_SUCCESS ) { \
			NeoML::IMathEngineExceptionHandler* exceptionHandler = GetMathEngineExceptionHandler(); \
			if( exceptionHandler != nullptr ) { \
				generateAssert( exceptionHandler, cusparse->GetErrorString( static_cast<cusparseStatus_t>( _err_ ) ), __UNICODEFILE__, __LINE__, _err_ ); \
			} else { \
				throw std::logic_error( #expr ); \
			} \
		} \
	} while(0)

// There is no GetErrorString function in cuBLAS API
// That's why using our own version
inline const char* cublasGetErrorString( cublasStatus_t status )
{
	switch( status ) {
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "CUBLAS_STATUS_NOT_SUPPORTED";
		case CUBLAS_STATUS_LICENSE_ERROR:
			return "CUBLAS_STATUS_LICENSE_ERROR";
		default:
			// There is no such value in cublas docs...
			return "Unknown CUBLAS error!";
	}
}

#define ASSERT_CUBLAS( expr ) \
	do { \
		int _err_ = ( int ) ( expr ); \
		if( _err_ != CUBLAS_STATUS_SUCCESS ) { \
			NeoML::IMathEngineExceptionHandler* exceptionHandler = GetMathEngineExceptionHandler(); \
			if( exceptionHandler != nullptr ) { \
				generateAssert( exceptionHandler, cublasGetErrorString( static_cast<cublasStatus_t>( _err_ ) ), __UNICODEFILE__, __LINE__, _err_ ); \
			} else { \
				throw std::logic_error( #expr ); \
			} \
		} \
	} while(0)

#endif // NEOML_USE_CUDA
