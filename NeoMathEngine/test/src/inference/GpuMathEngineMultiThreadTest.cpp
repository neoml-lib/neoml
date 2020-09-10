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

using namespace NeoML;
using namespace NeoMLTest;

class CGpuMathEngineMultiThreadTest : public CTestFixture {
};

static const size_t testMathEngineMemorySize = 512 * 1024 * 1024;

static void testMathEngine( IMathEngine& mathEngine, int runCount )
{
	CRandom random( 0x1984 );
	
	const int firstHeight = 2048;
	const int firstWidth = 5120;
	const int secondWidth = 3072;
	
	CFloatBlob firstMat( mathEngine, 1, firstHeight, firstWidth, 1 );
	CFloatBlob secondMat( mathEngine, 1, firstWidth, secondWidth, 1 );
	CFloatBlob result( mathEngine, 1, firstHeight, secondWidth, 1 );

	{
		// Filling with data
		CREATE_FILL_FLOAT_ARRAY( firstData, -1.f, 2.f, firstHeight * firstWidth, random );
		firstMat.CopyFrom( firstData.data() );
		CREATE_FILL_FLOAT_ARRAY( secondData, -1.f, 2.f, firstWidth * secondWidth, random );
		secondMat.CopyFrom( secondData.data() );
	}

	for( int run = 1; run <= runCount; ++run ) {
		mathEngine.MultiplyMatrixByMatrix( 1, firstMat.GetData(), firstHeight, firstWidth,
			secondMat.GetData(), secondWidth, result.GetData(), result.GetDataSize() );

		if( run % 10 == 0 ) {
			std::vector<float> resultData( firstHeight * secondWidth );
			result.CopyTo( resultData.data() );
		}
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_MultiThreadMultiMathEngine )
{
	std::unique_ptr<IMathEngine> firstME( CreateGpuMathEngine( testMathEngineMemorySize ) );
	std::unique_ptr<IMathEngine> secondME( CreateGpuMathEngine( testMathEngineMemorySize ) );

	if( firstME == nullptr || secondME == nullptr ) {
		FAIL() << "Failed to create 2 GPU math engines";
	}

	std::thread firstThread( testMathEngine, std::ref( *firstME ), 1000 );
	std::thread secondThread( testMathEngine, std::ref( *secondME ), 1000 );
	firstThread.join();
	secondThread.join();
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_MultiThreadSingleMathEngine )
{
	std::unique_ptr<IMathEngine> firstME( CreateGpuMathEngine( 2 * testMathEngineMemorySize ) );

	if( firstME == nullptr ) {
		FAIL() << "Failed to create math engine";
	}

	std::thread firstThread( testMathEngine, std::ref( *firstME ), 1000 );
	std::thread secondThread( testMathEngine, std::ref( *firstME ), 1000 );
	firstThread.join();
	secondThread.join();
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_SingleThreadMultiMathEngine )
{
	std::unique_ptr<IMathEngine> firstME( CreateGpuMathEngine( testMathEngineMemorySize ) );
	std::unique_ptr<IMathEngine> secondME( CreateGpuMathEngine( testMathEngineMemorySize ) );

	if( firstME == nullptr || secondME == nullptr ) {
		FAIL() << "Failed to create 2 GPU math engines";
	}

	// Checking switching between two mathEngines
	for( int iter = 0; iter < 10; ++iter ) {
		testMathEngine( *firstME, 100 );
		testMathEngine( *secondME, 100 );
	}
}

TEST_F( CGpuMathEngineMultiThreadTest, DISABLED_SingleDeviceMultiThreadMultiMathEngine )
{
	std::unique_ptr<IGpuMathEngineManager> gpuManager( CreateGpuMathEngineManager() );
	std::unique_ptr<IMathEngine> firstME( gpuManager->CreateMathEngine( 0, testMathEngineMemorySize ) );
	std::unique_ptr<IMathEngine> secondME( gpuManager->CreateMathEngine( 0, testMathEngineMemorySize ) );

	if( firstME == nullptr || secondME == nullptr ) {
		FAIL() << "Failed to create 2 math engines on device #0";
	}

	std::thread firstThread( testMathEngine, std::ref( *firstME ), 1000 );
	std::thread secondThread( testMathEngine, std::ref( *secondME ), 1000 );
	firstThread.join();
	secondThread.join();
}
