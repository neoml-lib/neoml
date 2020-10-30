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
#endif
#if FINE_PLATFORM( FINE_WINDOWS )
#include <intrin.h>
#endif

#if !FINE_PLATFORM( FINE_IOS )
#include <MathEngineDll.h>
#endif

#include <NeoMathEngine/SimdMathEngine.h>
#include <MathEngineCommon.h>

namespace NeoML {

#if !FINE_PLATFORM( FINE_IOS )
// Class support dynamic library which is compiled with AVX for performance increasing.
class CSimdMathEngineLoader : public CDll {
public:
	CSimdMathEngineLoader( const CSimdMathEngineLoader& ) = delete;
	CSimdMathEngineLoader& operator=( const CSimdMathEngineLoader& ) = delete;
	CSimdMathEngineLoader( CSimdMathEngineLoader&& ) = delete;
	CSimdMathEngineLoader& operator=( CSimdMathEngineLoader&& ) = delete;

	static CSimdMathEngineLoader& GetInstance();
	std::unique_ptr<ISimdMathEngine> CreateSimdMathEngine( IMathEngine* mathEngine, int threadCount );

private:
	constexpr static char const* CreateSimdMathEngineFuncName = "CreateSimdMathEngine";
	typedef ISimdMathEngine* ( *GetSimdMathEngineFunc )( IMathEngine* mathEngine, int threadCount );

	GetSimdMathEngineFunc createSimdMathEngineFunc;

	CSimdMathEngineLoader();
	~CSimdMathEngineLoader() = default;

	static bool isSimdAvailable();
};

CSimdMathEngineLoader::CSimdMathEngineLoader()
{
	if( !isSimdAvailable() ) {
		return;
	}
	
	#if FINE_PLATFORM( FINE_WINDOWS )
	bool res = Load( "NeoMathEngineAvx.dll" );
	#elif FINE_PLATFORM( FINE_LINUX )
	bool res = Load( "libNeoMathEngineAvx.so" );
	#elif FINE_PLATFORM( FINE_DARWIN )
	bool res = Load( "libNeoMathEngineAvx.dylib" );
	#else
	bool res = false;
	#endif

	if( !res ) {
		return;
	}

	createSimdMathEngineFunc = reinterpret_cast<GetSimdMathEngineFunc>( GetProcAddress( CreateSimdMathEngineFuncName ) );

	ASSERT_EXPR( createSimdMathEngineFunc != nullptr );
}

CSimdMathEngineLoader& CSimdMathEngineLoader::GetInstance()
{
	static CSimdMathEngineLoader instance;
	return instance;
}

std::unique_ptr<ISimdMathEngine> CSimdMathEngineLoader::CreateSimdMathEngine( IMathEngine* mathEngine, int threadCount )
{
	if( !IsLoaded() ) {
		return nullptr;
	}
	ISimdMathEngine* simdMathEngine = createSimdMathEngineFunc( mathEngine, threadCount );
	ASSERT_EXPR( simdMathEngine != nullptr );
	return std::unique_ptr<ISimdMathEngine>( simdMathEngine );
}

bool CSimdMathEngineLoader::isSimdAvailable()
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
#endif // !FINE_PLATFORM(FINE_IOS)

std::unique_ptr<ISimdMathEngine> ISimdMathEngine::CreateSimdMathEngine( IMathEngine* mathEngine, int threadCount )
{
	#if ( defined(NEOML_USE_SSE) || !defined(NEOML_USE_NEON) ) && !FINE_PLATFORM(FINE_IOS)
	return CSimdMathEngineLoader::GetInstance().CreateSimdMathEngine( mathEngine, threadCount );
	#else
	( void )filterCount;
	( void )channelCount;
	( void )filterHeight;
	( void )filterWidth;
	return nullptr;
	#endif
}

}