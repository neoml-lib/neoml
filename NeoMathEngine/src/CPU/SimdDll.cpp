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

#if !FINE_PLATFORM( FINE_IOS )

#if FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_LINUX )
#include <cpuid.h>
#endif
#if FINE_PLATFORM( FINE_WINDOWS )
#include <intrin.h>
#endif

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <NeoMathEngine/SimdMathEngine.h>
#include <MathEngineCommon.h>

#include <SimdDll.h>

namespace NeoML {

CSimdDll::CSimdDll() : createSimdMathEngineFunc( nullptr )
{
}

CSimdDll::~CSimdDll()
{
	Free();
}

bool CSimdDll::Load()
{
	if( IsLoaded() ) {
		return true;
	}

	if( !isSimdAvailable() ) {
		return false;
	}

	bool res = false;
	#if FINE_PLATFORM( FINE_WINDOWS )
	res = CDll::Load( "NeoMathEngineAvx.dll" );
	#elif FINE_PLATFORM( FINE_LINUX )
	res = CDll::Load( "libNeoMathEngineAvx.so" );
	#elif FINE_PLATFORM( FINE_DARWIN )
	res = CDll::Load( Load( "libNeoMathEngineAvx.dylib" );
	#endif
	if( !res ) {
		Free();
		return false;
	}

	if( !loadFunctions() ) {
		CDll::Free();
		return false;
	}
	return true;
}

void CSimdDll::Free()
{
	if( IsLoaded() ) {
		createSimdMathEngineFunc = nullptr;
		CDll::Free();
	}
}

ISimdMathEngine* CSimdDll::CreateSimdMathEngine( IMathEngine* mathEngine, int threadCount )
{
	if( !IsLoaded() ) {
		return nullptr;
	}
	ISimdMathEngine* simdMathEngine = createSimdMathEngineFunc( mathEngine, threadCount );
	ASSERT_EXPR( simdMathEngine != nullptr );
	return simdMathEngine;
}

bool CSimdDll::loadFunctions()
{
	createSimdMathEngineFunc = reinterpret_cast<GetSimdMathEngineFunc>( GetProcAddress( CreateSimdMathEngineFuncName ) );
	return createSimdMathEngineFunc != nullptr;
}

bool CSimdDll::isSimdAvailable()
{
	// Check for AVX
	#if FINE_PLATFORM(FINE_WINDOWS)

	#if _MSC_VER < 1900
	// VS 2015 compiles code which doesn't use all ymm registers. It brings to decrease performance compared to MKL.
	// Therefore we just disable AVX convolution  enhancement in VS2015.
	return false;
	#endif

	int cpuId[4] = { 0, 0, 0, 0 };
	__cpuid( cpuId, 1 );
	#elif FINE_PLATFORM(FINE_LINUX) || FINE_PLATFORM(FINE_DARWIN)
	unsigned int cpuId[4] = { 0, 0, 0, 0 };
	__get_cpuid( 1, cpuId, cpuId + 1, cpuId + 2, cpuId + 3 );
	#elif FINE_PLATFORM(FINE_ANDROID) || FINE_PLATFORM(FINE_IOS)
	unsigned int cpuId[4] = { 0, 0, 0, 0 };
	#else
	#error "Platform isn't supported!"
	#endif

	return ( cpuId[2] & 0x10000000 ) != 0;

}

} // namespace NeoML

#endif //  !FINE_PLATFORM( FINE_IOS )