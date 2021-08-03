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
#include "TestFixture.h"

namespace NeoMLTest {

static IMathEngine* mathEngine = 0;

void SetMathEngine(IMathEngine* newMathEngine)
{
	mathEngine = newMathEngine;
}

IMathEngine& MathEngine()
{
	return *mathEngine;
}

//------------------------------------------------------------------------------------------------------------

TMathEngineArgType GetMathEngineArgType( int argc, char* argv[] )
{
	constexpr char arg[] = "--MathEngine=";
	constexpr size_t argSize = sizeof( arg ) / sizeof( arg[0] ) - 1;
	
	for( int i = 0; i < argc; ++i ) {
		if( std::search( argv[i], argv[i] + strlen( argv[i] ), arg, arg + argSize ) == argv[i] ) {
			const char* value = argv[i] + argSize;
			if( strcmp( value, "gpu" ) == 0 ) {
				return TMathEngineArgType::Gpu;
			} else if( strcmp( value, "cpu" ) == 0 ) {
				return TMathEngineArgType::Cpu;
			} else if( strcmp( value, "cuda" ) == 0 ) {
				return TMathEngineArgType::Cuda;
			} else if( strcmp( value, "vulkan" ) == 0 ) {
				return TMathEngineArgType::Vulkan;
			} else if( strcmp( value, "metal" ) == 0 ) {
				return TMathEngineArgType::Metal;
			} else {
				return TMathEngineArgType::Undefined;
			}
		}
	}
	return TMathEngineArgType::Undefined;
}

int GetThreadCount( int argc, char* argv[] )
{
	constexpr char arg[] = "--ThreadCount=";
	constexpr size_t argSize = sizeof( arg ) / sizeof( arg[0] ) - 1;

	for( int i = 0; i < argc; ++i ) {
		const size_t argLen = strlen( argv[i] );
		if( std::search( argv[i], argv[i] + argLen, arg, arg + argSize ) == argv[i] ) {
			try {
				return std::stoi( argv[i] + argSize );
			} catch( std::exception& ) {
				// ...
			}
			return 0;
		}
	}
	return 0;
}

static std::string ToString( TMathEngineType type )
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

int RunTests( int argc, char* argv[] ) 
{
	::testing::InitGoogleTest( &argc, argv );
	
	IMathEngine* mathEngine = nullptr;
	
	auto type = GetMathEngineArgType( argc, argv );
	if( type == TMathEngineArgType::Gpu ) {
		mathEngine = CreateGpuMathEngine( 0 );
		if( mathEngine != nullptr ) {
			CMathEngineInfo info;
			mathEngine->GetMathEngineInfo( info );
			GTEST_LOG_( INFO ) << "Using " << ToString( info.Type ) << " GPU MathEngine: "
				<< info.Name << ", memory limit = " << info.AvailableMemory;
		} else {
			GTEST_LOG_( INFO ) << "Can't create Gpu MathEngine!";
			return 1;
		}
	} else if( type == TMathEngineArgType::Cuda || type == TMathEngineArgType::Vulkan || type == TMathEngineArgType::Metal ) {
		std::unique_ptr<IGpuMathEngineManager> manager( CreateGpuMathEngineManager() );
		TMathEngineType requiredType = type == TMathEngineArgType::Cuda ? MET_Cuda
			: ( type == TMathEngineArgType::Vulkan ? MET_Vulkan : MET_Metal );

		for( int i = 0; i < manager->GetMathEngineCount(); ++i ) {
			CMathEngineInfo info;
			manager->GetMathEngineInfo( i, info );
			if( info.Type == requiredType ) {
				mathEngine = manager->CreateMathEngine( i, 0 );
				if( mathEngine != nullptr ) {
					GTEST_LOG_( INFO ) << "Using " << ToString( info.Type ) << " GPU MathEngine: "
						<< info.Name << ", memory limit = " << info.AvailableMemory;
					break;
				} else {
					GTEST_LOG_( INFO ) << "Can't create Gpu MathEngine!";
					return 1;
				}
			}
		}
	} else if( type == TMathEngineArgType::Undefined ) {
		GTEST_LOG_( INFO ) << "Unknown type of MathEngine in command line arguments!";
	}
	
	if( mathEngine == nullptr ) {
		const int threadCount = GetThreadCount( argc, argv );
		mathEngine = CreateCpuMathEngine( threadCount, 0 );
		GTEST_LOG_( INFO ) << "Using CPU MathEngine, threadCount = " << threadCount;
	}

	SetMathEngine( mathEngine );

	int result = RUN_ALL_TESTS();
	
	delete mathEngine;
	
	return result;
}

} // namespace NeoMLTest
