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

#if FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_ANDROID )
#include <cpuid.h>
#endif
#if FINE_PLATFORM( FINE_WINDOWS )
#include <intrin.h>
#endif

#include <NeoMathEngineAvxDll.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>

namespace NeoML {

CNeoMathEngineAvxDll::CNeoMathEngineAvxDll() : isLoaded( false )
{
	if( !isAvxAvailable() || !Load( IAvxDll::LibName ) ) {
		return;
	}

	getAvxDllInstFunc = reinterpret_cast<IAvxDll::GetInstanceFunc>( GetProcAddress( IAvxDll::GetInstanceFuncName ) );
	ASSERT_EXPR( getAvxDllInstFunc != nullptr );

	isLoaded = true;
}

CNeoMathEngineAvxDll& CNeoMathEngineAvxDll::GetInstance()
{
	static CNeoMathEngineAvxDll instance;
	return instance;
}

std::unique_ptr<IAvxDll> CNeoMathEngineAvxDll::GetAvxDllInst( const CCommonConvolutionDesc& desc )
{
	if( !isLoaded ) {
		return nullptr;
	}
	std::unique_ptr<IAvxDll> avxDll( getAvxDllInstFunc( 
		desc.Filter.BatchWidth(), desc.Filter.Channels(), desc.Filter.Height(), desc.Filter.Width(),
		desc.Source.Height(), desc.Source.Width(), desc.StrideHeight, desc.StrideWidth,
		desc.DilationHeight, desc.DilationWidth, desc.Result.Height(), desc.Result.Width() ) );
	ASSERT_EXPR( avxDll != nullptr );
	return avxDll;
}

bool CNeoMathEngineAvxDll::isAvxAvailable()
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
}
