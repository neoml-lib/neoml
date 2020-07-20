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

#include <common.h>
#pragma hdrstop

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoMathEngine/OpenMP.h>

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
static IMathEngine* singleThreadCpuMathEngine = 0;
static IMathEngine* multiThreadCpuMathEngine = 0;
static bool isSingleThreadModeOn = true;
static CCriticalSection section;
#ifndef NEOML_USE_FINEOBJ
static CModuleInitializer moduleInitializer;
#endif //NEOML_USE_FINEOBJ
//------------------------------------------------------------------------------------------------------------

IMathEngineExceptionHandler* GetExceptionHandler()
{
	return &fmlExceptionHandler;
}

void EnableSingleThreadMode( bool enable )
{
	CCriticalSectionLock lock( section );
	isSingleThreadModeOn = enable;
}

bool IsSingleThreadModeOn()
{
	return isSingleThreadModeOn;
}

IMathEngine& GetSingleThreadCpuMathEngine()
{
	CCriticalSectionLock lock( section );
	if( singleThreadCpuMathEngine == 0 ) {
		SetMathEngineExceptionHandler( GetExceptionHandler() );
		singleThreadCpuMathEngine = CreateCpuMathEngine( 1, 0 );
	}
	return *singleThreadCpuMathEngine;
}

IMathEngine& GetMultiThreadCpuMathEngine()
{
	CCriticalSectionLock lock( section );
	if( multiThreadCpuMathEngine == 0 ) {
		SetMathEngineExceptionHandler( GetExceptionHandler() );
		multiThreadCpuMathEngine = CreateCpuMathEngine( 0, 0 );
	}
	return *multiThreadCpuMathEngine;
}

IMathEngine& GetDefaultCpuMathEngine()
{
	if( IsSingleThreadModeOn() ) {
		return GetSingleThreadCpuMathEngine();
	}
	return GetMultiThreadCpuMathEngine();
}

IMathEngine* GetRecommendedGpuMathEngine( size_t memoryLimit )
{
	SetMathEngineExceptionHandler( GetExceptionHandler() );
	return CreateGpuMathEngine( memoryLimit );
}

static void destroyDefaultCpuMathEngine()
{
	CCriticalSectionLock lock( section );
	if( singleThreadCpuMathEngine != 0 ) {
		delete singleThreadCpuMathEngine;
		singleThreadCpuMathEngine = 0;
	}
	if( multiThreadCpuMathEngine != 0 ) {
		delete multiThreadCpuMathEngine;
		multiThreadCpuMathEngine = 0;
	}
}

} // namespace NeoML

#ifndef STATIC_NEOML
// To avoid mistakes when unloading OpenMP, set the number of threads to 1 and perform any parallel task
static inline void deinitializeOmp()
{
#ifdef NEOML_USE_OMP
	const int prevNumThreads = omp_get_num_threads();
	omp_set_num_threads( 1 );

	const int size = 16;
	CFastArray<float, 16> arr;
	arr.Add( 0, size );

	#pragma omp parallel for
	for( int i = 0; i < size; ++i ) {
		arr[i]++;
	}
	omp_set_num_threads( prevNumThreads );
#endif
}

// Initializes a dynamic module
void InitializeModule()
{
}

// Deinitializes a dynamic module
void DeinitializeModule()
{
	NeoML::destroyDefaultCpuMathEngine();
	deinitializeOmp();
}
#endif // STATIC_NEOML

#ifndef NEOML_USE_FINEOBJ
NeoML::CModuleInitializer::~CModuleInitializer()
{
	NeoML::destroyDefaultCpuMathEngine();
}
#endif //NEOML_USE_FINEOBJ

//------------------------------------------------------------------------------------------------------------

#ifndef NEOML_USE_FINEOBJ

namespace FObj {

IObject::~IObject()
{}

} // FObj

#endif // NEOML_USE_FINEOBJ
