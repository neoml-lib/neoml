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

#if FINE_PLATFORM( FINE_WINDOWS )
	#ifndef NEOML_LIB_VERSION_SUFFIX
		#if defined(_DEBUG)
			#define NEOML_CONFIGURATION_NAME "Debug"
		#elif defined(_FINAL)
			#define NEOML_CONFIGURATION_NAME "Final"
		#else
			#define NEOML_CONFIGURATION_NAME "Release"
		#endif

		#if FINE_BIT( FINE_64_BIT )
			#define NEOML_PLATFORM_NAME "x64"
		#elif FINE_BIT( FINE_32_BIT )
			#define NEOML_PLATFORM_NAME "Win32"
		#endif

		#define NEOML_LIB_VERSION_SUFFIX "." NEOML_PLATFORM_NAME "." NEOML_CONFIGURATION_NAME
	#endif

	#if defined( FINEOBJ_VERSION ) && !defined( CMAKE_INTDIR )
		#pragma comment( lib, "NeoOnnx" NEOML_LIB_VERSION_SUFFIX ".lib" )
	#endif // FINEOBJ_VERSION
#endif // FINE_PLATFORM( FINE_WINDOWS )

#if defined( STATIC_NEOONNX )
	#define NEOONNX_API
#elif defined( BUILD_NEOONNX )
	#if FINE_PLATFORM( FINE_WINDOWS )
		#define NEOONNX_API	__declspec( dllexport )
	#elif FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_IOS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
		#define NEOONNX_API	__attribute__( ( visibility("default") ) )
	#endif
#else
	#if FINE_PLATFORM( FINE_WINDOWS )
		#define NEOONNX_API	__declspec( dllimport )
	#elif FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_IOS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
		#define NEOONNX_API	__attribute__( ( visibility("default") ) )
	#endif
#endif
