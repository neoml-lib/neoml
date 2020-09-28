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

#include <NeoML/NeoMLDefs.h>

namespace NeoML {

// Checks the expression and throws the exception with an error message about the network architecture
void NEOML_API CheckArchitecture( bool expression, const char* layerName, const char* message );

// Throws the exception with an internal error message
bool NEOML_API ThrowInternalError( TInternalErrorType errorType, const char* functionName,
	const char* errorText, const wchar_t* fileName, int line, int errorCode );
} // namespace NeoML

#ifdef _DEBUG

#define NeoAssert( expr ) \
if( !( expr ) ) { \
	FineDebugBreak();	\
	if( NeoML::ThrowInternalError( IET_Assert, __FUNCTION__, "", __UNICODEFILE__, __LINE__, 0 ) ) \
		FineBreakPoint(); \
} else \
	( ( void )1 )

#define NeoPresume( expr ) \
if( !( expr ) ) { \
	FineDebugBreak();	\
	if( NeoML::ThrowInternalError( IET_Presume, __FUNCTION__, "", __UNICODEFILE__, __LINE__, 0 ) ) \
		FineBreakPoint(); \
} else \
		( ( void )1 )

#else // Release

#define NeoAssert( expr ) \
if( !( expr ) ) { \
	NeoML::ThrowInternalError( IET_Assert, __FUNCTION__, "", __UNICODEFILE__, __LINE__, 0 ); \
} else \
	( ( void )1 )

// Presume turned off in release version
#define NeoPresume( expr ) while( 0 )( ( void )1 )

#endif // _DEBUG
