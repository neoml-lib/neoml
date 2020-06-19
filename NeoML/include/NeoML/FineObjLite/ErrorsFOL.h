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

#include "FineObjLiteDefs.h"
#include "StringFOL.h"

#if FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_IOS )
#include <signal.h>
#endif

namespace FObj {

#define __merge__2( a, b )	a##b
#define __merge__1( a, b )	__merge__2( a, b )

//------------------------------------------------------------------------------------------------------------
// Exceptions

typedef std::exception CException;
typedef std::logic_error CInternalError;
typedef std::system_error CFileException;
typedef std::bad_alloc CMemoryException;
typedef std::logic_error CCheckException;

static const char* ERR_BAD_ARCHIVE = "%0 is corrupted.";
static const char* ERR_BAD_ARCHIVE_VERSION = "Invalid version of %0.";

inline void check( bool condition, const char* error, const char* param1, const char* param2 = "" )
{
	if( !condition ) {
		const char* params[2] = { param1, param2 };
		throw CCheckException( SubstParam( error, params, 2 ) );
	}
}

inline void ThrowFileException( int errorCode, const char* fileName )
{
	throw CFileException( errorCode, std::iostream_category(), fileName );
}

//------------------------------------------------------------------------------------------------------------
// Assert and Presume definitions

inline void FineBreakPoint()
{
}

#ifdef _DEBUG

inline void FineDebugBreak()
{
#if FINE_PLATFORM( FINE_WINDOWS )
	__debugbreak();
#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_IOS )
	raise( SIGTRAP );
#else
	#error Unknown platform!
#endif
}

#else

inline void FineDebugBreak() {}

#endif // _DEBUG

enum TInternalErrorType {
	IET_Assert,
	IET_AssertLastError,
	IET_Presume,
	IET_PresumeLastError
};

// Generates the "internal error" exception
inline void GenerateInternalError( TInternalErrorType errorType, const char* functionName,
	const char* errorText, const char* fileName, int line, int errorCode )
{
	CString message;
	switch( errorType ) {
		case IET_Assert:
			message = "Internal Program Error:\nAssertion failed: (%0)\n%2, %3\nFunction: %1";
			break; 
		case IET_AssertLastError:
			message = "Internal Program Error:\nAssertion failed: (%0)\n%2, %3.\nFunction: %1\nError code: %4";
			break; 
		case IET_Presume:
			message = "Internal Program Error:\nPresumption failed: (%0)\n%2, %3\nFunction: %1";
			break; 
		case IET_PresumeLastError:
			message = "Internal Program Error:\nPresumption failed: (%0)\n%2, %3.\nFunction: %1\nError code: %4";
			break; 
	};

	CString lineStr = Str( line );
	CString errorCodeStr = Str( errorCode );
	const char* params[5] = { errorText, functionName, fileName, lineStr.data(), errorCodeStr.data() };
	message = SubstParam( message, params, 5 );
	throw CInternalError( message );
}

#ifdef _DEBUG

#define AssertFO( expr ) \
if( !( expr ) ) { \
	FineDebugBreak();	\
	FObj::GenerateInternalError( IET_Assert, __FUNCTION__, #expr, __FILE__, __LINE__, 0 ); \
}

#define PresumeFO( expr ) \
if( !( expr ) ) { \
	FineDebugBreak();	\
	FObj::GenerateInternalError( IET_Presume, __FUNCTION__, #expr, __FILE__, __LINE__, 0 ); \
}

#else // Release

#define AssertFO( expr ) \
if( !( expr ) ) { \
FObj::GenerateInternalError( IET_Assert, "", "", __FILE__, __LINE__, 0 ); \
}

// PresumeFO is turned off for Release version
#define PresumeFO( expr ) while( 0 )( ( void )1 )

#endif // _DEBUG

} // namespace FObj
