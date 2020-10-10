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

#include <TestFixture.h>
#include <algorithm>
#include <memory>

namespace NeoMLTest {

namespace {

	IMathEngine* mathEngine = nullptr;
	CString testDir;
	void* platformEnv = nullptr;
	int threadCount = 0;
	TMathEngineType type = MET_Undefined;

	template <typename T, std::size_t N>
	bool StartsWith( const T* str, const T(&prefix)[N] )
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
	bool Equal( const T* one, const T(&two)[N] )
	{
		return StartsWith( one, two ) && one[N] == '\0';
	}

	template <typename T, std::size_t N>
	const T* ArgValue( int argc, T* argv[], const T(&argument)[N] )
	{
		for( int i = 0; i < argc; ++i ) {
			if( StartsWith( argv[i], argument ) ) {
				return argv[i] + N;
			}
		}
		return nullptr;
	}

#ifdef NEOML_USE_FINEOBJ
    using CharType = wchar_t;
#else
	using CharType = char;
#endif

	static constexpr CharType TestDataPath[] = { '-','-','T','e','s','t','D','a','t','a','P','a','t','h','=' };
	static constexpr CharType MathEngine[] = { '-','-','M','a','t','h','E','n','g','i','n','e','=' };
	static constexpr CharType ThreadCount[] = { '-','-','T','h','r','e','a','d','C','o','u','n','t','=' };

	static constexpr CharType Cpu[] = { 'c', 'p', 'u' };
	static constexpr CharType Cuda[] = { 'c','u','d','a' };
	static constexpr CharType Vulkan[] = { 'v','u','l','k','a','n' };
	static constexpr CharType Metal[] = { 'm','e','t','a','l' };

	template <typename T>
	TMathEngineType GetMathEngineType( int argc, T* argv[] )
	{
		auto value = ArgValue( argc, argv, MathEngine );
		if( !value ) {
			return MET_Undefined;
		}

		if( Equal( value, Cpu ) ) {
			return MET_Cpu;
		} else if( Equal( value, Metal ) ) {
			return MET_Metal;
		} else if( Equal( value, Cuda ) ) {
			return MET_Cuda;
		} else if( Equal( value, Vulkan ) ) {
			return MET_Vulkan;
		}
		return MET_Undefined;
	}

	inline const char* toString( TMathEngineType type )
	{
		if( type == MET_Cpu ) {
			return "Cpu";
		} else if( type == MET_Cuda ) {
			return "Cuda";
		} else if( type == MET_Vulkan ) {
			return "Vulkan";
		} else if( type == MET_Metal ) {
			return "Metal";
		}
		return "";
	}

#ifdef NEOML_USE_FINEOBJ

	inline void InitTestDataPath( int argc, wchar_t* argv[] )
	{
		auto value = ArgValue( argc, argv, TestDataPath );
		if( value ) {
			testDir = CString( value, CP_UTF8 );
		}
	}

	inline int GetThreadCount( int argc, wchar_t* argv[] )
	{
		auto value = ArgValue( argc, argv, ThreadCount );
		int res = 0;
		if( value && FObj::Value( value, res ) ) {
			return res;
		}
		return 0;
	}

#else // NEOML_USE_FINEOBJ

	inline void InitTestDataPath( int argc, char* argv[] )
	{
		auto value = ArgValue( argc, argv, TestDataPath );
		if( value ) {
			testDir = FObj::CString( value );
		}
	}

	inline int GetThreadCount( int argc, char* argv[] )
	{
		auto value = ArgValue( argc, argv, ThreadCount );
		if( value ) {
			try {
				return std::stoi( value );
			} catch( std::exception& ) {
				return 0;
			}
		}
		return 0;
	}

#endif // NEOML_USE_FINEOBJ
}

IMathEngine* CreateMathEngine( TMathEngineType type, std::size_t memoryLimit, int threadCount )
{
	IMathEngine* result = nullptr;
	switch( type ) {
		case MET_Cuda:
		case MET_Vulkan:
		case MET_Metal: {
			std::unique_ptr<IGpuMathEngineManager> gpuManager( CreateGpuMathEngineManager() );
			CMathEngineInfo info;
			for( int i = 0; i < gpuManager->GetMathEngineCount(); ++i ) {
				gpuManager->GetMathEngineInfo( i, info );
				if( info.Type == type ) {
					result = gpuManager->CreateMathEngine( i, memoryLimit );
					break;
				}
			}
			if( result ) {
				GTEST_LOG_( INFO ) << "Create GPU " << toString( type ) << " MathEngine: " << info.Name;
			} else {
				GTEST_LOG_( ERROR ) << "Can't create GPU " << toString( type ) << " MathEngine!";
			}
			break;
		}
		case MET_Undefined:
			GTEST_LOG_( WARNING ) << "Unknown type of MathEngine!";
		case MET_Cpu: {
			result = CreateCpuMathEngine( threadCount, memoryLimit );
			GTEST_LOG_( INFO ) << "Create CPU MathEngine, threadCount = " << threadCount;
			break; 
		}
	}
	return result;
}

#ifdef NEOML_USE_FINEOBJ
int RunTests( int argc, wchar_t* argv[], void* platformEnv )
#else
int RunTests( int argc, char* argv[], void* platformEnv )
#endif
{
	NeoMLTest::InitTestDataPath( argc, argv );
	::testing::InitGoogleTest( &argc, argv );

	threadCount = NeoMLTest::GetThreadCount( argc, argv );

	type = GetMathEngineType( argc, argv );

	SetPlatformEnv( platformEnv );

	int result = RUN_ALL_TESTS();

	DeleteMathEngine();

	return result;
}

void SetPlatformEnv( void* _platformEnv )
{
	platformEnv = _platformEnv;
}

void* GetPlatformEnv()
{
	return platformEnv;
}

static inline bool isPathSeparator( char ch ) { return ch == '\\' || ch == '/'; }

static CString mergePathSimple( const CString& dir, const CString& relativePath )
{
	const char* dirPtr = dir;
	const size_t dirLen = strlen( dirPtr );
	const char* relativePathPtr = relativePath;
	const size_t relativePathLen = strlen( relativePathPtr );
	if( dirLen == 0 ) {
		return relativePath;
	}

	int separatorsCount = 0;
	if( isPathSeparator( dirPtr[dirLen - 1] ) ) {
		separatorsCount++;
	}
	if( relativePathLen > 0 && isPathSeparator( relativePathPtr[0] ) ) {
		separatorsCount++;
	}

	CString result;
	switch( separatorsCount ) {
		case 0:
			#if FINE_PLATFORM( FINE_WINDOWS )
				result = dir + "\\" + relativePath;
			#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_IOS ) || FINE_PLATFORM( FINE_DARWIN )
				result = dir + "/" + relativePath;
			#else
				#error Unknown platform
			#endif
			break;
		case 1:
			result = dir + relativePath;
			break;
		case 2:
			result = CString( dirPtr, static_cast<int>( dirLen - 1 ) ) + relativePath;
			break;
		default:
			NeoAssert( false );
	}
	return result;
}

//------------------------------------------------------------------------------------------------------------

CString GetTestDataFilePath( const CString& relativePath, const CString& fileName )
{
	return mergePathSimple( mergePathSimple( testDir, relativePath ), fileName );
}

IMathEngine& MathEngine()
{
	if( mathEngine == nullptr ) {
		mathEngine = CreateMathEngine( type, 0u, threadCount );
		NeoAssert( mathEngine != nullptr );
		SetMathEngineExceptionHandler( GetExceptionHandler() );
	}
	return *mathEngine;
}

void DeleteMathEngine()
{
	if( mathEngine ) {
		delete mathEngine;
		mathEngine = nullptr;
	}
}

TMathEngineType MathEngineType()
{
	return type;
}

} // namespace NeoMLTest
