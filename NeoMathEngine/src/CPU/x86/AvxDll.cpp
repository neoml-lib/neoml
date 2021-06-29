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

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <NeoMathEngine/SimdMathEngine.h>
#include <MathEngineCommon.h>

#include <AvxDll.h>
#include <CPUInfo.h>

#include <string>

static std::string getModuleDir()
{
	std::string result;

#if FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_LINUX )
	Dl_info dlInfo;
	auto returnValue = dladdr( reinterpret_cast<void*>( getModuleDir ), &dlInfo );
	ASSERT_EXPR( returnValue != 0 );
		
	constexpr char separator[] = { '/' };
	const auto* dllPath = dlInfo.dli_fname;
	auto it = std::find_end( dllPath, dllPath + strlen( dllPath ), separator, separator + 1 );
		
	result.assign( dllPath, it + 1 );

#elif FINE_PLATFORM( FINE_WINDOWS )

	static_assert( sizeof( TCHAR ) == sizeof( char ), "TCHAR is wide char type!" );
		
	std::vector<char> buffer;
	DWORD copiedChars = 0;
		
	HMODULE handle;
	auto returnValue = GetModuleHandleEx( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, 
		reinterpret_cast<LPCSTR>( getModuleDir ), &handle );
	PRESUME_EXPR( returnValue != 0 );
		
	do {
		buffer.resize( buffer.size() + MAX_PATH );
		copiedChars = GetModuleFileName( handle, buffer.data(), static_cast<DWORD>( buffer.size() ) );
	} while( copiedChars >= buffer.size() );
		
	constexpr char separator[] = {'\\'};
	auto it = std::find_end( buffer.cbegin(), buffer.cbegin() + copiedChars, separator, separator + 1 );

	result.assign( buffer.cbegin(), it + 1 );

#else
	#error "Platform isn't supported!"
#endif
	return result;
}
namespace NeoML {

CAvxDll::CAvxDll() :
	createSimdMathEngineFunc( nullptr )
{
}

CAvxDll::~CAvxDll()
{
	Free();
}

bool CAvxDll::Load()
{
	if( IsLoaded() ) {
		return true;
	}

	if( !isAvxAvailable() ) {
		return false;
	}

	std::string dllPath( getModuleDir() );
#if FINE_PLATFORM( FINE_WINDOWS )
	dllPath += "NeoMathEngineAvx.dll";
#elif FINE_PLATFORM( FINE_LINUX )
	dllPath += "libNeoMathEngineAvx.so";
#elif FINE_PLATFORM( FINE_DARWIN )
	dllPath += "libNeoMathEngineAvx.dylib";
#else
	#error "Platform isn't supported!"
#endif
	ASSERT_EXPR( CDll::Load( dllPath.c_str() ) );

	ASSERT_EXPR( loadFunctions() );

	return true;
}

void CAvxDll::Free()
{
	if( IsLoaded() ) {
		createSimdMathEngineFunc = nullptr;
		CDll::Free();
	}
}

ISimdMathEngine* CAvxDll::CreateSimdMathEngine( IMathEngine* mathEngine, int threadCount )
{
	ASSERT_EXPR( IsLoaded() );

	ISimdMathEngine* simdMathEngine = createSimdMathEngineFunc( mathEngine, threadCount );
	ASSERT_EXPR( simdMathEngine != nullptr );
	return simdMathEngine;
}

bool CAvxDll::loadFunctions()
{
	createSimdMathEngineFunc = GetProcAddress<CreateSimdMathEngineFunc>( CreateSimdMathEngineFuncName );
	return createSimdMathEngineFunc != nullptr;
}

bool CAvxDll::isAvxAvailable()
{
#if defined(_MSC_VER) && _MSC_VER < 1925
	// VS 2015 compiles code which doesn't use all ymm registers. It brings to decrease performance compared to MKL.
	// Therefore we just disable AVX convolution  enhancement in VS2015.
	return false;
#endif

	static bool res = CCPUInfo::IsAvxAndFmaAvailable() && !CCPUInfo::IsAvx512Available();
	return res;
}

} // namespace NeoML
