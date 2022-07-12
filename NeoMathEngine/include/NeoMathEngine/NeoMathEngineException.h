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

#include <NeoMathEngine/CrtAllocatedObject.h>

#define __merge__2( a, b )	a##b
#define __merge__1( a, b )	__merge__2( a, b )
#define __UNICODEFILE__	__merge__1( L, __FILE__ )

namespace NeoML {

// Exception handler interface
// Use it to change the program's reaction to exceptions
class NEOMATHENGINE_API IMathEngineExceptionHandler {
public:
	virtual ~IMathEngineExceptionHandler();
	// An error during a method call
	// The default action is to throw std::logic_error
	virtual void OnAssert( const char* message, const wchar_t* file, int line, int errorCode ) = 0;

	// Memory cannot be allocated on device
	// The default action is to throw std::bad_alloc
	virtual void OnMemoryError() = 0;
};

// Set exception handler interface for whole programm
// Setting this to null means using default exception handling
// Non-default handler must be destroyed by the caller after use
NEOMATHENGINE_API void SetMathEngineExceptionHandler( IMathEngineExceptionHandler* exceptionHandler );

// Get current exception handler interface ( returns default one if handler wasn't set )
NEOMATHENGINE_API IMathEngineExceptionHandler* GetMathEngineExceptionHandler();

} // namespace NeoML

#define ASSERT_ERROR_CODE( expr ) \
	do { \
		int _err_ = (int)(expr); \
		if(_err_ != 0) { \
			NeoML::GetMathEngineExceptionHandler()->OnAssert( #expr, __UNICODEFILE__, __LINE__, _err_ ); \
		} \
	} while(0)

#define ASSERT_EXPR( expr ) \
	do { \
		if(!(expr)) { \
			NeoML::GetMathEngineExceptionHandler()->OnAssert( #expr, __UNICODEFILE__, __LINE__, 0 ); \
		} \
	} while(0)

#ifdef _DEBUG
	#define PRESUME_ERROR_CODE( expr ) ASSERT_ERROR_CODE( expr )
	#define PRESUME_EXPR( expr ) ASSERT_EXPR( expr )
#else
	// presume is turned off in release version
	#define PRESUME_ERROR_CODE( expr )
	#define PRESUME_EXPR( expr )
#endif

#define THROW_MEMORY_EXCEPTION \
	do { \
		GetMathEngineExceptionHandler()->OnMemoryError(); \
	} while(0)
