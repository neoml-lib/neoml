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

#include <FinePlatforms.h>

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

