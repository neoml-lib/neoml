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

#include <NeoMathEngine/Platforms.h>

#if FINE_ARCHITECTURE( FINE_X86 ) || FINE_ARCHITECTURE( FINE_X64 )
#define NEOML_USE_SSE
#endif

#if FINE_ARCHITECTURE( FINE_ARM ) || FINE_ARCHITECTURE( FINE_ARM64 )
#define NEOML_USE_NEON
#define NEOML_USE_OWN_BLAS
#endif

#if FINE_PLATFORM( FINE_IOS )
#define NEOML_USE_METAL
#endif // FINE_PLATFORM( FINE_IOS )

#if FINE_PLATFORM( FINE_WINDOWS )

#ifndef FME_LIB_VERSION_SUFFIX

#if defined(_DEBUG)
	#define FME_CONFIGURATION_NAME "Debug"
#elif defined(_FINAL)
	#define FME_CONFIGURATION_NAME "Final"
#else
	#define FME_CONFIGURATION_NAME "Release"
#endif

#if FINE_BIT( FINE_64_BIT )
	#define FME_PLATFORM_NAME "x64"
#elif FINE_BIT( FINE_32_BIT )
	#define FME_PLATFORM_NAME "Win32"
#endif

#define FME_LIB_VERSION_SUFFIX "." FME_PLATFORM_NAME "." FME_CONFIGURATION_NAME
#endif

#include <climits>
#include <cstddef>

#if defined( FINEOBJ_VERSION ) && !defined( CMAKE_INTDIR )
	#ifndef BUILD_NEOMATHENGINE
		#pragma comment( lib, "NeoMathEngine" FME_LIB_VERSION_SUFFIX ".lib" )
	#else // ifdef BUILD_NEOMATHENGINE

		#ifdef NEOML_USE_MKL

			#ifdef _WIN64
				#pragma comment( lib, "mkl_intel_lp64.lib" )
			#else
				#pragma comment( lib, "mkl_intel_c.lib" )
			#endif

			#pragma comment( lib, "mkl_core.lib" )
			#pragma comment( lib, "mkl_sequential.lib" )

		#endif

		#if defined( NEOML_USE_CUDA )
			#pragma comment( lib, "cudart_static.lib" )
		#endif

	#endif // BUILD_NEOMATHENGINE
#endif // FINEOBJ_VERSION && !CMAKE_INTDIR

#endif // FINE_PLATFORM( FINE_WINDOWS )

#if FINE_PLATFORM( FINE_WINDOWS )
#define FME_DLL_IMPORT __declspec( dllimport )
#define FME_DLL_EXPORT __declspec( dllexport )
#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_IOS )
#define FME_DLL_IMPORT __attribute__((visibility("default")))
#define FME_DLL_EXPORT __attribute__((visibility("default")))
#else
#error "Platform isn't supported!"
#endif

#if defined( STATIC_NEOMATHENGINE )
	#define NEOMATHENGINE_API
#elif defined( BUILD_NEOMATHENGINE )
	#define NEOMATHENGINE_API FME_DLL_EXPORT
#else
	#define NEOMATHENGINE_API FME_DLL_IMPORT
#endif

#if FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_IOS )
	#ifndef MEMORY_ALLOCATION_ALIGNMENT
		#define MEMORY_ALLOCATION_ALIGNMENT (2 * sizeof(void*))
	#endif
#endif

