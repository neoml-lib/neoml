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

#include <algorithm>
#include <memory>
#include <string>
#include "TestFixture.h"

namespace NeoMLTest {

static IMathEngine* mathEngine = 0;

IMathEngine& MathEngine()
{
	return *mathEngine;
}

enum class TMathEngineArgType 
{
	Undefined = 0,
	Cpu,
	Gpu,
	Cuda,
	Vulkan,
	Metal
};

//------------------------------------------------------------------------------------------------------------

static void setMathEngine(IMathEngine* newMathEngine)
{
	mathEngine = newMathEngine;
}

template <typename T, std::size_t N>
static bool startsWith( const T* str, const T( &prefix )[N] )
{
	size_t i = 0;
	for( ; i < N && *str != '\0'; ++i, ++str ) {
		if( *str != prefix[i] ) {
			return false;
		}
	}
	return i == N;
}

template <typename T, std::size_t N>
static bool equal( const T* one, const T( &two )[N] )
{
	return startsWith( one, two ) && one[N] == '\0';
}

template <typename T, std::size_t N>
static const T* argValue( int argc, T* argv[], const T( &argument )[N] )
{
	for( int i = 0; i < argc; ++i ) {
		if( startsWith( argv[i], argument ) ) {
			return argv[i] + N;
		}
	}
	return nullptr;
}

#ifdef NEOML_USE_FINEOBJ
using TCharType = wchar_t;
#else
using TCharType = char;
#endif

static constexpr TCharType mathEngineArg[] = { '-', '-', 'M', 'a', 't', 'h', 'E', 'n', 'g', 'i', 'n', 'e', '=' };
static constexpr TCharType threadCount[] = { '-', '-', 'T', 'h', 'r', 'e', 'a', 'd', 'C', 'o', 'u', 'n', 't', '=' };

static constexpr TCharType cpuArg[] = { 'c', 'p', 'u' };
static constexpr TCharType gpu[] = { 'g', 'p', 'u' };
static constexpr TCharType cuda[] = { 'c', 'u', 'd', 'a' };
static constexpr TCharType vulkan[] = { 'v', 'u', 'l', 'k', 'a', 'n' };
static constexpr TCharType metal[] = { 'm', 'e', 't', 'a', 'l' };

template<typename T>
TMathEngineArgType getMathEngineArgType( int argc, T* argv[] )
{
	const T* rawArg = argValue( argc, argv, mathEngineArg );

	if( rawArg != nullptr ) {
		if( equal( rawArg, cpuArg ) ) {
			return TMathEngineArgType::Cpu;
		} else if( equal( rawArg, gpu ) ) {
			return TMathEngineArgType::Gpu;
		} else if( equal( rawArg, cuda ) ) {
			return TMathEngineArgType::Cuda;
		} else if( equal( rawArg, vulkan ) ) {
			return TMathEngineArgType::Vulkan;
		} else if( equal( rawArg, metal ) ) {
			return TMathEngineArgType::Metal;
		}
	}

	return TMathEngineArgType::Undefined;
}

#ifdef NEOML_USE_FINEOBJ

int getThreadCount( int argc, wchar_t* argv[] )
{
	const wchar_t* value = argValue( argc, argv, ThreadCount );
	int res = 0;
	if( value && FObj::Value( value, res ) ) {
		return res;
	}
	return 0;
}

#else // NEOML_USE_FINEOBJ

int getThreadCount( int argc, char* argv[] )
{
	const char* rawThreadCount = argValue( argc, argv, threadCount );

	if( rawThreadCount != nullptr ) {
		try {
			return std::stoi( rawThreadCount );
		} catch( std::exception& ) {
			return 0;
		}
	}

	return 0;
}

#endif // NEOML_USE_FINEOBJ

static std::string toString( TMathEngineType type )
{
	switch( type ) {
		case MET_Cpu:
			return "CPU";
		case MET_Cuda:
			return "CUDA";
		case MET_Vulkan:
			return "Vulkan";
		case MET_Metal:
			return "Metal";
		default:
			return "UNKNOWN";
	}
	return "UNKNOWN";
}

static IMathEngine* createMathEngine( TMathEngineArgType argType, int threadCount )
{
	switch( argType ) {
		case TMathEngineArgType::Cuda:
		case TMathEngineArgType::Vulkan:
		case TMathEngineArgType::Metal:
		{
			TMathEngineType meType = argType == TMathEngineArgType::Cuda ? MET_Cuda
				: ( argType == TMathEngineArgType::Vulkan ? MET_Vulkan : MET_Metal );
			std::unique_ptr<IGpuMathEngineManager> manager( CreateGpuMathEngineManager() );
			if( manager == nullptr ) {
				return nullptr;
			}
			for( int i = 0; i < manager->GetMathEngineCount(); ++i ) {
				CMathEngineInfo info;
				manager->GetMathEngineInfo( i, info );
				if( info.Type == meType ) {
					IMathEngine* mathEngine = manager->CreateMathEngine( i, 0 );
					if( mathEngine != nullptr ) {
						return mathEngine;
					}
				}
			}
			break;
		}
		case TMathEngineArgType::Gpu:
		{
			std::unique_ptr<IGpuMathEngineManager> manager( CreateGpuMathEngineManager() );
			if( manager == nullptr ) {
				return nullptr;
			}
			for( int i = 0; i < manager->GetMathEngineCount(); ++i ) {
				IMathEngine* mathEngine = manager->CreateMathEngine( i, 0 );
				if( mathEngine != nullptr ) {
					return mathEngine;
				}
			}
			break;
		}
		case TMathEngineArgType::Undefined:
			GTEST_LOG_( WARNING ) << "Unknown type of MathEngine!";
			// fall through
		case TMathEngineArgType::Cpu:
			return CreateCpuMathEngine( threadCount, 0 );
		default:
			return nullptr;
	}
	return nullptr;
}

int RunTests( int argc, char* argv[] ) 
{
	::testing::InitGoogleTest( &argc, argv );
	
	auto type = getMathEngineArgType( argc, argv );
	const int threadCount = getThreadCount( argc, argv );

	IMathEngine* mathEngine = createMathEngine( type, threadCount );

	if( mathEngine != nullptr ) {
		CMathEngineInfo info;
		mathEngine->GetMathEngineInfo( info );
		GTEST_LOG_( INFO ) << "Using " << toString( info.Type ) << " MathEngine: "
			<< info.Name << ", memory limit = " << info.AvailableMemory;
	} else {
		GTEST_LOG_( INFO ) << "Can't create MathEngine!";
		return 1;
	}

	setMathEngine( mathEngine );

	int result = RUN_ALL_TESTS();
	
	delete mathEngine;
	
	return result;
}

} // namespace NeoMLTest
