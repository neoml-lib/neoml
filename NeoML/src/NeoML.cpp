/* Copyright © 2017-2023 ABBYY

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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

#ifdef NEOML_USE_FINEOBJ

DefineFineError( ERR_NEOML_DNN_BAD_ARCHITECTURE )

void CheckArchitecture( bool expression, const char* layerName, const char* message )
{
	if( !expression ) {
		check( expression, ERR_NEOML_DNN_BAD_ARCHITECTURE, CreateUnicodeString( layerName, CP_UTF8 ),
			CreateUnicodeString( message, CP_UTF8 ) );
	}
}

bool ThrowInternalError( TInternalErrorType errorType, const char* functionName,
	const char* errorText, const wchar_t* fileName, int line, int errorCode )
{
	return GenerateInternalError( errorType, CUnicodeString( functionName, CP_UTF8 ),
		CUnicodeString( errorText, CP_UTF8 ), fileName, line, errorCode );
}

#else

void CheckArchitecture( bool expression, const char* layerName, const char* message )
{
	if( !expression ) {
		const char* params[2] = { layerName, message };
		CString exceptionMessage = SubstParam( "Bad architecture of %0: %1", params, 2 );
		throw CCheckException( exceptionMessage );
	}
}

bool ThrowInternalError( TInternalErrorType errorType, const char* functionName,
	const char* errorText, const wchar_t* fileName, int line, int errorCode )
{
	GenerateInternalError( errorType, functionName, errorText, fileName, line, errorCode );
	return true;
}

#endif

//------------------------------------------------------------------------------------------------------------

// NeoMathEngine exception handler
class CMathEngineExceptionHandler : public IMathEngineExceptionHandler {
public:
	// IMathEngineExceptionHandler interface methods
	void OnAssert( const char* message, const wchar_t* file, int line, int errorCode ) override
	{
#ifdef _DEBUG
		FineDebugBreak();
		if( ThrowInternalError( IET_Assert, "", message, file, line, errorCode ) ) {
			FineBreakPoint();
		}
#else 
		ThrowInternalError( IET_Assert, "", message, file, line, errorCode );
#endif
	}

	void OnMemoryError() override
	{
#ifdef NEOML_USE_FINEOBJ
		throw FINE_DEBUG_NEW CMemoryException();
#else
		throw CMemoryException();
#endif
	}
};

#ifndef NEOML_USE_FINEOBJ
struct CModuleInitializer {
	CModuleInitializer() = default;
	~CModuleInitializer();

	CModuleInitializer( const CModuleInitializer& ) = delete;
	CModuleInitializer& operator=( const CModuleInitializer& ) = delete;
};
#endif //NEOML_USE_FINEOBJ

//------------------------------------------------------------------------------------------------------------

static CMathEngineExceptionHandler fmlExceptionHandler;
static IMathEngine* cpuMathEngine = nullptr;
static CCriticalSection section;
#ifndef NEOML_USE_FINEOBJ
static CModuleInitializer moduleInitializer;
#endif //NEOML_USE_FINEOBJ
//------------------------------------------------------------------------------------------------------------

IMathEngineExceptionHandler* GetExceptionHandler()
{
	return &fmlExceptionHandler;
}

// deprecated
void EnableSingleThreadMode( bool /*enable*/ )
{
}

// deprecated
bool IsSingleThreadModeOn()
{
	return true;
}

// deprecated
IMathEngine& GetSingleThreadCpuMathEngine()
{
	return GetDefaultCpuMathEngine();
}

// deprecated
IMathEngine& GetMultiThreadCpuMathEngine()
{
	return GetDefaultCpuMathEngine();
}

IMathEngine& GetDefaultCpuMathEngine()
{
	CCriticalSectionLock lock( section );
	if( cpuMathEngine == nullptr ) {
		SetMathEngineExceptionHandler( GetExceptionHandler() );
		cpuMathEngine = CreateCpuMathEngine( /*memoryLimit*/0u );
	}
	return *cpuMathEngine;
}

IMathEngine* GetRecommendedGpuMathEngine( size_t memoryLimit )
{
	SetMathEngineExceptionHandler( GetExceptionHandler() );
	return CreateGpuMathEngine( memoryLimit );
}

static void destroyDefaultCpuMathEngine()
{
	CCriticalSectionLock lock( section );
	if( cpuMathEngine != nullptr ) {
		delete cpuMathEngine;
		cpuMathEngine = nullptr;
	}
	CpuMathEngineCleanUp();
}

} // namespace NeoML

#ifndef STATIC_NEOML
// Initializes a dynamic module
void InitializeModule()
{
}

// Deinitializes a dynamic module
void DeinitializeModule()
{
	NeoML::destroyDefaultCpuMathEngine();
	NeoML::DeinitializeNeoMathEngine();
}
#endif // !STATIC_NEOML

#ifndef NEOML_USE_FINEOBJ
NeoML::CModuleInitializer::~CModuleInitializer()
{
	NeoML::destroyDefaultCpuMathEngine();
}
#endif // NEOML_USE_FINEOBJ

//------------------------------------------------------------------------------------------------------------

#ifndef NEOML_USE_FINEOBJ

namespace FObj {

IObject::~IObject()
{}

} // FObj

#endif // NEOML_USE_FINEOBJ
