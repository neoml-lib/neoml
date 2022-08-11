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

// Test log
// Used in order to avoid troubles with exception while using threads
// Empty log means test success

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

// Function to test
// Run runCount times some function of mathEngine
typedef void( *TTestedFunction ) ( IMathEngine& mathEngine, int runCount );

// Test mechanism
// Determines used number of devices/threads used in test
// Returns true if test was passed
typedef bool( *TTestMechanism ) ( TTestedFunction function, bool useSingleDevice );

// Forward-declarations (bodies are located at the end of the file

// TTestMechanism
// Creates 2 mathEngines and passes them to 2 separate threads for testing
static bool multiThreadMultiMETest( TTestedFunction func, bool useSingleDevice );
// Creates one mathEngine and passes it to 2 separate threads for testing
static bool multiThreadSingleMETest( TTestedFunction func, bool /* useSingleDevice */ );
// Creates 2 mathEngines and uses them in tests (no threads created)
static bool singleThreadMultiMathEngineTest( TTestedFunction func, bool useSingleDevice );
// Starts 2 threads. Each thread creates its mathEngine and tests it
static bool multiThreadCreateMETest( TTestedFunction func, bool useSingleDevice );

// TTestedFunction
static void testCublas( IMathEngine& mathEngine, int runCount ); // One cublas function
static void testCusparse( IMathEngine& mathEngine, int runCount ); // One cusparse function
static void testKernel( IMathEngine& mathEngine, int runCount ); // One custom CUDA kernel
static void testMemoryFunctions( IMathEngine& mathEngine, int runCount ); // copying to/from GPU (and between different memory regions on GPU)

class CMultiGpuMultiThreadTest : public CTestFixture, public ::testing::WithParamInterface<std::tuple<TTestMechanism, TTestedFunction, bool>> {
protected:
	void SetUp() override { clearLog(); }
};

std::string getTestMechanismName( const TTestMechanism& testMechanism )
{
	if( testMechanism == multiThreadMultiMETest ) {
		return "multiThreadMultiMETest";
	} else if( testMechanism == multiThreadCreateMETest ) {
		return "multiThreadCreateMETest";
	} else if( testMechanism == singleThreadMultiMathEngineTest ) {
		return "singleThreadMultiMathEngineTest";
	} else if( testMechanism == multiThreadCreateMETest ) {
		return "multiThreadCreateMETest";
	}
	return "UNKNOWN_TEST_MECH";
}

std::string getTestedFunctionName( const TTestedFunction& testedFunction )
{
	if( testedFunction == testCublas ) {
		return "testCublas";
	} else if( testedFunction == testCusparse ) {
		return "testCusparse";
	} else if( testedFunction == testKernel ) {
		return "testKernel";
	} else if( testedFunction == testMemoryFunctions ) {
		return "testMemoryFunctions";
	}
	return "UNKNOWN_TESTED_FUNCTION";
}

TEST_P( CMultiGpuMultiThreadTest, DISABLED_Run )
{
	TTestMechanism testMechanism = std::get<0>( GetParam() );
	TTestedFunction testedFunction = std::get<1>( GetParam() );
	bool useSingleDevice = std::get<2>( GetParam() );

	// There is no simple way to overload << operator when working with tuples
	GTEST_LOG_( INFO ) << getTestMechanismName( testMechanism ) << "(" << getTestedFunctionName( testedFunction )
		<< ", " << std::boolalpha << useSingleDevice << std::noboolalpha << ")" << std::endl;

	if( !testMechanism( testedFunction, useSingleDevice ) ) {
		FAIL();
	}
}

INSTANTIATE_TEST_CASE_P( CMultiGpuMultiThreadTestInstantiation0, CMultiGpuMultiThreadTest,
	::testing::Combine(
		::testing::Values( multiThreadMultiMETest, singleThreadMultiMathEngineTest, multiThreadCreateMETest ),
		::testing::Values( testCublas, testCusparse, testKernel, testMemoryFunctions ),
		::testing::Values( true, false )
	) );

// This test mechanism ignores last bool
INSTANTIATE_TEST_CASE_P( CMultiGpuMultiThreadTestInstantiation1, CMultiGpuMultiThreadTest,
	::testing::Combine(
		::testing::Values( multiThreadSingleMETest ),
		::testing::Values( testCublas, testCusparse, testKernel, testMemoryFunctions ),
		::testing::Values( true )
	) );

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

		const int firstHeight = 5120;
		const int firstWidth = 4096;
		const int secondHeight = 3072;

		std::vector<int> rows, columns;
		std::vector<float> values;
		rows.push_back( 0 );
		const int presetY = random.UniformInt( 0, firstHeight - 1 );
		const int presetX = random.UniformInt( 0, firstWidth - 1 );
		for( int i = 0; i < firstHeight; i++ ) {
			int elementsInRow = 0;
			for( int j = 0; j < firstWidth; j++ ) {
				if( ( i == presetY && j == presetX ) || random.UniformInt( 0, 2 ) != 0 ) {
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
			CSparseMatrix sparseMatrix( MathEngine(), rows, columns, values );
			mathEngine.MultiplySparseMatrixByTransposedMatrix( firstHeight, firstWidth, secondHeight,
				sparseMatrix.Desc(), second.GetData(), result.GetData() );

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

static void testMemoryFunctions( IMathEngine& mathEngine, int runCount )
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

static const size_t memoryRequiredForTests = 512 * 1024 * 1024;

static bool multiThreadMultiMETest( TTestedFunction func, bool useSingleDevice )
{
	try {
		std::unique_ptr<IMathEngine> firstME;
		std::unique_ptr<IMathEngine> secondME;

		if( useSingleDevice ) {
			std::unique_ptr<IGpuMathEngineManager> gpuManager( CreateGpuMathEngineManager() );
			firstME.reset( gpuManager->CreateMathEngine( 0, memoryRequiredForTests ) );
			secondME.reset( gpuManager->CreateMathEngine( 0, memoryRequiredForTests ) );
		} else {
			firstME.reset( CreateGpuMathEngine( 0 ) );
			secondME.reset( CreateGpuMathEngine( 0 ) );
		}

		if( firstME == nullptr || secondME == nullptr ) {
			addToLog( "Failed to create 2 GPU math engines" );
		} else {
			std::thread firstThread( func, std::ref( *firstME ), 200 );
			std::thread secondThread( func, std::ref( *secondME ), 200 );
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

static bool multiThreadSingleMETest( TTestedFunction func, bool /*useSingleDevice*/ )
{
	try {
		std::unique_ptr<IMathEngine> firstME( CreateGpuMathEngine( 0 ) );

		if( firstME == nullptr ) {
			addToLog( "Failed to create math engine" );
		} else {
			std::thread firstThread( func, std::ref( *firstME ), 200 );
			std::thread secondThread( func, std::ref( *firstME ), 200 );
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

static bool singleThreadMultiMathEngineTest( TTestedFunction func, bool useSingleDevice )
{
	try {
		std::unique_ptr<IMathEngine> firstME;
		std::unique_ptr<IMathEngine> secondME;

		if( useSingleDevice ) {
			std::unique_ptr<IGpuMathEngineManager> gpuManager( CreateGpuMathEngineManager() );
			firstME.reset( gpuManager->CreateMathEngine( 0, memoryRequiredForTests ) );
			secondME.reset( gpuManager->CreateMathEngine( 0, memoryRequiredForTests ) );
		} else {
			firstME.reset( CreateGpuMathEngine( 0 ) );
			secondME.reset( CreateGpuMathEngine( 0 ) );
		}

		if( firstME == nullptr || secondME == nullptr ) {
			addToLog( "Failed to create 2 GPU math engines" );
		} else {
			// Checking switching between two mathEngines
			for( int iter = 0; iter < 10; ++iter ) {
				func( *firstME, 10 );
				func( *secondME, 10 );
			}
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

static void createMathEngineAndRunTest( TTestedFunction func, bool useSingleDevice )
{
	try {
		std::unique_ptr<IMathEngine> mathEngine;

		if( useSingleDevice ) {
			std::unique_ptr<IGpuMathEngineManager> gpuManager( CreateGpuMathEngineManager() );
			// both threads will create GPU on device #0
			mathEngine.reset( gpuManager->CreateMathEngine( 0, memoryRequiredForTests ) );
		} else {
			mathEngine.reset( CreateGpuMathEngine( 0 ) );
		}

		if( mathEngine == nullptr ) {
			addToLog( "failed to create math engine" );
			return;
		}

		func( *mathEngine, 200 );
	} catch( std::exception& ex ) {
		addToLog( ex.what() );
	}
}

static bool multiThreadCreateMETest( TTestedFunction func, bool useSingleDevice )
{
	try {
		std::thread firstThread( createMathEngineAndRunTest, func, useSingleDevice );
		std::thread secondThread( createMathEngineAndRunTest, func, useSingleDevice );
		firstThread.join();
		secondThread.join();
	} catch( std::exception& ex ) {
		addToLog( ex.what() );
	}

	if( !isLogEmpty() ) {
		printLog();
		return false;
	}

	return true;
}
