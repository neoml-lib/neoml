/* Copyright © 2017-2022 ABBYY Production LLC

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

#include "common.h"

#include <memory>
#include <mutex>
#include <cassert>
#include "windows.h"

using namespace NeoML;

class CMathEngineInstance {
public:
	explicit CMathEngineInstance( int threadCount );
	~CMathEngineInstance();

	CMathEngineInstance( const CMathEngineInstance& ) = delete;
	CMathEngineInstance( const CMathEngineInstance&& ) = delete;
	CMathEngineInstance& operator=( const CMathEngineInstance& ) = delete;
	CMathEngineInstance& operator=( const CMathEngineInstance&& ) = delete;

	IMathEngine& GetMathEngine() { return *mathEngine; }
	ISimdMathEngine& GetSimdMathEngine() { return *simdMathEngine; }

private:
	std::mutex mutex;
	HMODULE avxModule;
	IMathEngine* mathEngine;
	ISimdMathEngine* simdMathEngine;
};

typedef ISimdMathEngine* ( TCreateSimdMathEngine )( IMathEngine*, int );

CMathEngineInstance::CMathEngineInstance( int threadCount ) :
	avxModule( NULL ),
	mathEngine( nullptr ),
	simdMathEngine( nullptr )
{
	avxModule = ::LoadLibrary( "NeoMathEngineAvx.dll" );
	assert( avxModule != nullptr );
	
	FARPROC createSimdME = ::GetProcAddress( avxModule, "CreateSimdMathEngine" );
	assert( createSimdME != nullptr );

	mathEngine = CreateCpuMathEngine( threadCount, 0 );
	assert( mathEngine != nullptr );

	simdMathEngine = reinterpret_cast<TCreateSimdMathEngine*>( createSimdME )( mathEngine, threadCount );
	assert( simdMathEngine != nullptr );
}

CMathEngineInstance::~CMathEngineInstance()
{
	if( simdMathEngine != nullptr ) {
		delete simdMathEngine;
	}

	if( mathEngine != nullptr ) {
		delete mathEngine;
	}

	if( avxModule != nullptr ) {
		::FreeModule( avxModule );
	}
}

static constexpr int meThreadCount = 1;
static CMathEngineInstance meInstance( meThreadCount );

IMathEngine& MathEngine()
{
	return meInstance.GetMathEngine();
}

ISimdMathEngine& SimdMathEngine()
{
	return meInstance.GetSimdMathEngine();
}
