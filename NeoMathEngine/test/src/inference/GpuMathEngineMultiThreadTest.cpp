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

#include <TestFixture.h>

#include <memory>
#include <thread>
#include <mutex>
#include <string>
#include <vector>

using namespace NeoML;
using namespace NeoMLTest;

class CGpuMathEngineMultiThreadTest : public CTestFixture {
};

static const size_t testCublasMemorySize = 512 * 1024 * 1024;

static std::mutex testLogMutex;
static std::vector<std::string> testLog;

static void clearLog()
{
	std::lock_guard<std::mutex> lock( testLogMutex );
	testLog.clear();
}

static bool isLogEmpty()
{
	std::lock_guard<std::mutex> lock( testLogMutex );
	return testLog.empty();
}

static void addToLog( const std::string& line )
{
	std::lock_guard<std::mutex> lock( testLogMutex );
	testLog.push_back( line );
}

static void printLog()
{
	std::lock_guard<std::mutex> lock( testLogMutex );
	for( const auto& line : testLog ) {
		std::cout << line << std::endl;
	}
}

typedef void( *TTestFunc ) ( IMathEngine& mathEngine, int runCount );

static void testCublas( IMathEngine& mathEngine, int runCount )
{
	try {
		CRandom random( 0x1984 );

		const int firstHeight = 2048;
		const int firstWidth = 5120;
		const int secondWidth = 3072;

		CFloatBlob first( mathEngine, 1, firstHeight, firstWidth, 1 );
		CFloatBlob second( mathEngine, 1, firstWidth, secondWidth, 1 );
		CFloatBlob result( mathEngine, 1, firstHeight, secondWidth, 1 );

		{
			// Filling with data
			CREATE_FILL_FLOAT_ARRAY( firstData, -1.f, 2.f, first.GetDataSize(), random );
			first.CopyFrom( firstData.data() );
			CREATE_FILL_FLOAT_ARRAY( secondData, -1.f, 2.f, second.GetDataSize(), random );
			second.CopyFrom( secondData.data() );
		}

		for( int run = 1; run <= runCount; ++run ) {
			mathEngine.MultiplyMatrixByMatrix( 1, first.GetData(), firstHeight, firstWidth,
				second.GetData(), secondWidth, result.GetData(), result.GetDataSize() );

			if( run % 10 == 0 ) {
				std::vector<float> resultData( result.GetDataSize() );
				result.CopyTo( resultData.data() );
			}
		}
	} catch( std::exception& ex ) {
		addToLog( ex.what() );
	}
}

static void testCusparse( IMathEngine& mathEngine, int runCount )
{
	try {
		CRandom random( 0x1984 );

		const int firstHeight = 10240;
		const int firstWidth = 5120;
		const int secondHeight = 3072;

		std::vector<int> rows, columns;
		std::vector<float> values;
		rows.push_back( 0 );
		for( int i = 0; i < firstHeight; i++ ) {
			int elementsInRow = 0;
			for( int j = 0; j < firstWidth; j++ ) {
				if( random.UniformInt( 0, 1 ) ) {
					float value = static_cast<float>( random.Uniform( -2., 1. ) );
					columns.push_back( j );
					values.push_back( value );
					elementsInRow++;
				}
			}
			rows.push_back( elementsInRow );
		}

		CFloatBlob second( mathEngine, 1, secondHeight, firstWidth, 1 );
		CFloatBlob result( mathEngine, 1, firstHeight, secondHeight, 1 );

		{
			// Filling with data
			CREATE_FILL_FLOAT_ARRAY( secondData, -1.f, 2.f, second.GetDataSize(), random );
			second.CopyFrom( secondData.data() );
		}

		for( int run = 1; run <= runCount; ++run ) {
			mathEngine.MultiplySparseMatrixByTransposedMatrix( firstHeight, firstWidth, secondHeight,
				GetSparseMatrix( MathEngine(), rows, columns, values ), second.GetData(), result.GetData() );

			if( run % 10 == 0 ) {
				std::vector<float> resultData( result.GetDataSize() );
				result.CopyTo( resultData.data() );
			}
		}
	} catch( std::exception& ex ) {
		addToLog( ex.what() );
	}
}

static void testKernel( IMathEngine& mathEngine, int runCount )
{
	try {
		CRandom random( 0x1984 );

		const int vectorSize = 16 * 1024 * 1024;

		CFloatBlob first( mathEngine, 1, 1, 1, vectorSize );
		CFloatBlob second( mathEngine, 1, 1, 1, vectorSize );
		CFloatBlob result( mathEngine, 1, 1, 1, vectorSize );

		{
			// Filling with data
			CREATE_FILL_FLOAT_ARRAY( firstData, -1.f, 2.f, first.GetDataSize(), random );
			first.CopyFrom( firstData.data() );
			CREATE_FILL_FLOAT_ARRAY( secondData, -1.f, 2.f, second.GetDataSize(), random );
			second.CopyFrom( secondData.data() );
		}

		for( int run = 1; run <= runCount; ++run ) {
			mathEngine.VectorEltwiseMultiply( first.GetData(), second.GetData(), result.GetData(),
				vectorSize );

			if( run % 10 == 0 ) {
				std::vector<float> resultData( result.GetDataSize() );
				result.CopyTo( resultData.data() );
			}
		}
	} catch( std::exception& ex ) {
		addToLog( ex.what() );
	}
}

static void testMemeFunc( IMathEngine& mathEngine, int runCount )
{
	try {
		CRandom random( 0x1984 );

		const int vectorSize = 16 * 1024 * 1024;

		CFloatBlob first( mathEngine, 1, 1, 1, vectorSize );
		CFloatBlob second( mathEngine, 1, 1, 1, vectorSize );
		CFloatBlob result( mathEngine, 1, 1, 1, vectorSize );

		{
			// Filling with data
			CREATE_FILL_FLOAT_ARRAY( firstData, -1.f, 2.f, first.GetDataSize(), random );
			first.CopyFrom( firstData.data() );
			CREATE_FILL_FLOAT_ARRAY( secondData, -1.f, 2.f, second.GetDataSize(), random );
			second.CopyFrom( secondData.data() );
		}

		for( int run = 1; run <= runCount; ++run ) {
			mathEngine.VectorCopy( run % 2 == 1 ? first.GetData() : second.GetData(), result.GetData(),
				result.GetDataSize() );

			if( run % 10 == 0 ) {
				std::vector<float> resultData( result.GetDataSize() );
				result.CopyTo( resultData.data() );
			}
		}
	} catch( std::exception& ex ) {
		addToLog( ex.what() );
	}
}

static bool multiThreadMultiMETest( TTestFunc func )
{
	clearLog();

	try {
		std::unique_ptr<IMathEngine> firstME( CreateGpuMathEngine( 0 ) );
		std::unique_ptr<IMathEngine> secondME( CreateGpuMathEngine( 0 ) );

		if( firstME == nullptr || secondME == nullptr ) {
			addToLog( "Failed to create 2 GPU math engines" );
		} else {
			std::thread firstThread( func, std::ref( *firstME ), 1000 );
			std::thread secondThread( func, std::ref( *secondME ), 1000 );
			firstThread.join();
			secondThread.join();
		}
	} catch( std::exception& ex ) {
		addToLog( ex.what() );
	}

	if( !isLogEmpty() ) {
		printLog();
		return false;
	}

	return true;
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_MultiThreadMultiMathEngineCublas )
{
	if( !multiThreadMultiMETest( testCublas ) ) {
		FAIL();
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_MultiThreadMultiMathEngineCusparse )
{
	if( !multiThreadMultiMETest( testCusparse ) ) {
		FAIL();
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_MultiThreadMultiMathEngineMemeFunc )
{
	if( !multiThreadMultiMETest( testMemeFunc ) ) {
		FAIL();
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_MultiThreadMultiMathEngineKernel )
{
	if( !multiThreadMultiMETest( testKernel ) ) {
		FAIL();
	}
}

static bool multiThreadSingleMETest( TTestFunc func )
{
	clearLog();

	try {
		std::unique_ptr<IMathEngine> firstME( CreateGpuMathEngine( 0 ) );

		if( firstME == nullptr ) {
			addToLog( "Failed to create math engine" );
		} else {
			std::thread firstThread( func, std::ref( *firstME ), 1000 );
			std::thread secondThread( func, std::ref( *firstME ), 1000 );
			firstThread.join();
			secondThread.join();
		}
	} catch( std::exception& ex ) {
		addToLog( ex.what() );
	}

	if( !isLogEmpty() ) {
		printLog();
		return false;
	}

	return true;
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_MultiThreadSingleMathEngineCublas )
{
	if( !multiThreadSingleMETest( testCublas ) ) {
		FAIL();
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_MultiThreadSingleMathEngineCusparse )
{
	if( !multiThreadSingleMETest( testCusparse ) ) {
		FAIL();
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_MultiThreadSingleMathEngineMemeFunc )
{
	if( !multiThreadSingleMETest( testMemeFunc ) ) {
		FAIL();
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_MultiThreadSingleMathEngineKernel )
{
	if( !multiThreadSingleMETest( testKernel ) ) {
		FAIL();
	}
}

static bool singleThreadMultiMathEngineTest( TTestFunc func )
{
	clearLog();

	try {
		std::unique_ptr<IMathEngine> firstME( CreateGpuMathEngine( 0 ) );
		std::unique_ptr<IMathEngine> secondME( CreateGpuMathEngine( 0 ) );

		if( firstME == nullptr || secondME == nullptr ) {
			addToLog( "Failed to create 2 GPU math engines" );
		}

		// Checking switching between two mathEngines
		for( int iter = 0; iter < 10; ++iter ) {
			func( *firstME, 100 );
			func( *secondME, 100 );
		}
	} catch( std::exception& ex ) {
		addToLog( ex.what() );
	}

	if( !isLogEmpty() ) {
		printLog();
		return false;
	}

	return true;
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_SingleThreadMultiMathEngineCublas )
{
	if( !singleThreadMultiMathEngineTest( testCublas ) ) {
		FAIL();
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_SingleThreadMultiMathEngineCusparse )
{
	if( !singleThreadMultiMathEngineTest( testCusparse ) ) {
		FAIL();
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_SingleThreadMultiMathEngineMemeFunc )
{
	if( !singleThreadMultiMathEngineTest( testMemeFunc ) ) {
		FAIL();
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_SingleThreadMultiMathEngineKernel )
{
	if( !singleThreadMultiMathEngineTest( testKernel ) ) {
		FAIL();
	}
}

static bool singleDeviceMultiThreadMultiMETest( TTestFunc func )
{
	clearLog();

	try {
		std::unique_ptr<IGpuMathEngineManager> gpuManager( CreateGpuMathEngineManager() );
		std::unique_ptr<IMathEngine> firstME( gpuManager->CreateMathEngine( 0, testCublasMemorySize ) );
		std::unique_ptr<IMathEngine> secondME( gpuManager->CreateMathEngine( 0, testCublasMemorySize ) );

		if( firstME == nullptr || secondME == nullptr ) {
			addToLog( "Failed to create 2 math engines on device #0" );
		} else {
			std::thread firstThread( func, std::ref( *firstME ), 1000 );
			std::thread secondThread( func, std::ref( *secondME ), 1000 );
			firstThread.join();
			secondThread.join();
		}
	} catch( std::exception& ex ) {
		addToLog( ex.what() );
	}

	if( !isLogEmpty() ) {
		printLog();
		return false;
	}

	return true;
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_SingleDeviceMultiThreadMultiMathEngineCublas )
{
	if( !singleDeviceMultiThreadMultiMETest( testCublas ) ) {
		FAIL();
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_SingleDeviceMultiThreadMultiMathEngineCusparse )
{
	if( !singleDeviceMultiThreadMultiMETest( testCusparse ) ) {
		FAIL();
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_SingleDeviceMultiThreadMultiMathEngineMemeFunc )
{
	if( !singleDeviceMultiThreadMultiMETest( testMemeFunc ) ) {
		FAIL();
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_SingleDeviceMultiThreadMultiMathEngineKernel )
{
	if( !singleDeviceMultiThreadMultiMETest( testKernel ) ) {
		FAIL();
	}
}
