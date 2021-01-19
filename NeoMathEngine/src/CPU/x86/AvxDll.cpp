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

#if FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_LINUX )
#include <cpuid.h>
#elif FINE_PLATFORM( FINE_WINDOWS )
#include <intrin.h>
#else
#error "Platform isn't supported!"
#endif

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <NeoMathEngine/SimdMathEngine.h>
#include <MathEngineCommon.h>

#include <AvxDll.h>

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

#if FINE_PLATFORM( FINE_WINDOWS )
	ASSERT_EXPR( CDll::Load( "NeoMathEngineAvx.dll" ) );
#elif FINE_PLATFORM( FINE_LINUX )
	ASSERT_EXPR( CDll::Load( "libNeoMathEngineAvx.so" ) );
#elif FINE_PLATFORM( FINE_DARWIN )
	ASSERT_EXPR( CDll::Load( "libNeoMathEngineAvx.dylib" ) );
#else
	#error "Platform isn't supported!"
#endif

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
	createSimdMathEngineFunc = reinterpret_cast<CreateSimdMathEngineFunc>( GetProcAddress( CreateSimdMathEngineFuncName ) );
	return createSimdMathEngineFunc != nullptr;
}

bool CAvxDll::isAvxAvailable()
{
	// Check for AVX
#if FINE_PLATFORM(FINE_WINDOWS)

#if _MSC_VER < 1925
	// VS 2015 compiles code which doesn't use all ymm registers. It brings to decrease performance compared to MKL.
	// Therefore we just disable AVX convolution  enhancement in VS2015.
	return false;
#endif

	int cpuInfo[4] = { 0, 0, 0, 0 };
	int cpuInfoEx[4] = { 0, 0, 0, 0 };
	__cpuid( cpuInfo, 1 );
	__cpuidex( cpuInfoEx, 7, 0 );
#elif FINE_PLATFORM(FINE_LINUX) || FINE_PLATFORM(FINE_DARWIN)
	unsigned int cpuInfo[4] = { 0, 0, 0, 0 };
	unsigned int cpuInfoEx[4] = { 0, 0, 0, 0 };
	__get_cpuid( 1, cpuInfo, cpuInfo + 1, cpuInfo + 2, cpuInfo + 3 );
	__cpuid_count( 7, 0, cpuInfoEx[0], cpuInfoEx[1], cpuInfoEx[2], cpuInfoEx[3] );
#else
	#error "Platform isn't supported!"
#endif

	const unsigned int AvxAndFmaBits = ( 1 << 28 ) + ( 1 << 12 );
	bool AvxAndFmaAreAvailable = ( cpuInfo[2] & AvxAndFmaBits ) == AvxAndFmaBits;
	// Check avx512_f bit in EBX ( any CPU with AVX512 has this bit )
	bool AnyAvx512IsAvailable = cpuInfoEx[1] & ( 1 << 16 );

	return AvxAndFmaAreAvailable && !AnyAvx512IsAvailable;

}

} // namespace NeoML
