/* Copyright © 2023-2024 ABBYY

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
#include <NeoMathEngine/PerformanceCounters.h>
#include <fstream>
#include <memory>

using namespace NeoML;
using namespace NeoMLTest;

namespace NeoMLTest {

static const char* vectorFunctionsNames[]{
	"VectorCopy",
	"VectorFill",
	"VectorAdd",
	"VectorAddVal",
	"VectorMultiply",
	"VectorEltwiseMultiply",
	"VectorEltwiseMultAdd",
	"VectorReLU(0)",
	"VectorReLU(Threshold)",
	"VectorHSwish"
};

//------------------------------------------------------------------------------------------------------------

class VectorBenchmarkParams final {
public:
	int Function = -2;
	int TestCount = -1;
	int VectorSize = -1;
	IPerformanceCounters* Counters = nullptr;
	std::ofstream FOut{};

	VectorBenchmarkParams( int function, int testCount, int vectorSize,
		const CInterval& valuesInterval, int seed );
	~VectorBenchmarkParams() { delete Counters; }

	void SetNextSeedForFunction( int function, int seed );

	CFloatWrapper& GetInputBuffer() { return *inputBuf; }
	CFloatWrapper& GetSecondBuffer() { return *secondBuf; }
	CFloatWrapper& GetResultBuffer() { return *resultBuf; }
	CFloatWrapper& GetZeroVal() { return *zeroBuf; }
	CFloatWrapper& GetMulVal() { return *mulBuf; }

private:
	const CInterval& valuesInterval;
	CRandom random;

	std::vector<float> input;
	std::vector<float> second;
	std::vector<float> result;

	std::unique_ptr<CFloatWrapper> inputBuf = nullptr;
	std::unique_ptr<CFloatWrapper> secondBuf = nullptr;
	std::unique_ptr<CFloatWrapper> resultBuf = nullptr;
	std::unique_ptr<CFloatWrapper> zeroBuf = nullptr;
	std::unique_ptr<CFloatWrapper> mulBuf = nullptr;
};

VectorBenchmarkParams::VectorBenchmarkParams( int function, int testCount, int vectorSize,
		const CInterval& valuesInterval, int seed  ) :
	Function( function ),
	TestCount( testCount ),
	VectorSize( vectorSize ),
	Counters( MathEngine().CreatePerformanceCounters() ),
	FOut( std::ofstream( "VectorBenchmarkTest.csv", std::ios::app ) ),
	valuesInterval( valuesInterval )
{
	FOut << "\n---------------------------" << std::endl;
	input.resize( vectorSize );
	second.resize( vectorSize );
	result.resize( vectorSize );
	float zero = 0; 
	zeroBuf.reset( new CFloatWrapper( MathEngine(), &zero, 1 ) );

	SetNextSeedForFunction( function, seed );
}

void VectorBenchmarkParams::SetNextSeedForFunction( int function, int seed )
{
	Function = function;
	random = CRandom( seed );
	for( int i = 0; i < VectorSize; ++i ) {
		input[i] = static_cast<float>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) );
		second[i] = static_cast<float>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) );
		result[i] = 0;
	}
	float multiplier = static_cast<float>( random.Uniform( 1, valuesInterval.End ) );
	mulBuf.reset( new CFloatWrapper( MathEngine(), &multiplier, 1 ) );
	CConstFloatHandle mulHandle = *mulBuf;
	ASSERT_EXPR( mulHandle.GetValueAt( 0 ) > 0 );

	inputBuf.reset( new CFloatWrapper( MathEngine(), input.data(), VectorSize ) );
	secondBuf.reset( new CFloatWrapper( MathEngine(), second.data(), VectorSize ) );
	resultBuf.reset( new CFloatWrapper( MathEngine(), result.data(), VectorSize ) );
}

//------------------------------------------------------------------------------------------------------------

static double vectorBenchmark( VectorBenchmarkParams& params )
{
	CFloatWrapper& input = params.GetInputBuffer();
	CFloatWrapper& second = params.GetSecondBuffer();
	CFloatWrapper& result = params.GetResultBuffer();
	CFloatWrapper& zeroVal = params.GetZeroVal();
	CFloatWrapper& mulVal = params.GetMulVal();
	const int vectorSize = params.VectorSize;

	if( params.Function == -1 ) { // warm-up
		MathEngine().VectorCopy( result, input, vectorSize );
		MathEngine().VectorFill( result, vectorSize, mulVal );
		MathEngine().VectorAdd( input, second, result, vectorSize );
		MathEngine().VectorAddValue( input, result, vectorSize, mulVal );
		MathEngine().VectorMultiply( input, second, vectorSize, mulVal );
		MathEngine().VectorEltwiseMultiply( input, second, result, vectorSize );
		MathEngine().VectorEltwiseMultiplyAdd( input, second, result, vectorSize );
		MathEngine().VectorReLU( input, result, vectorSize, zeroVal ); //Threshold == 0
		MathEngine().VectorReLU( input, result, vectorSize, mulVal ); //Threshold > 0
		MathEngine().VectorHSwish( input, result, vectorSize );
		return 0;
	}

	params.Counters->Synchronise();

	for( int i = 0; i < params.TestCount; ++i ) {
		switch( params.Function ) {
			case 0: MathEngine().VectorCopy( result, input, vectorSize ); break;
			case 1: MathEngine().VectorFill( result, vectorSize, mulVal ); break;
			case 2: MathEngine().VectorAdd( input, second, result, vectorSize ); break;
			case 3: MathEngine().VectorAddValue( input, result, vectorSize, mulVal ); break;
			case 4: MathEngine().VectorMultiply( input, second, vectorSize, mulVal ); break;
			case 5: MathEngine().VectorEltwiseMultiply( input, second, result, vectorSize ); break;
			case 6: MathEngine().VectorEltwiseMultiplyAdd( input, second, result, vectorSize ); break;
			case 7: MathEngine().VectorReLU( input, result, vectorSize, zeroVal ); break; //Threshold == 0
			case 8: MathEngine().VectorReLU( input, result, vectorSize, mulVal ); break; //Threshold > 0
			case 9: MathEngine().VectorHSwish( input, result, vectorSize ); break;
			default:
				ASSERT_EXPR( false );
		}
	}

	params.Counters->Synchronise();
	const double time = double( ( *params.Counters )[0].Value ) / 1000000 / params.TestCount; // average time in milliseconds
	params.FOut << time << ",";
	return time;
}

} // namespace NeoMLTest

//------------------------------------------------------------------------------------------------------------

class CMathEngineVectorBenchmarkTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineVectorBenchmarkTestInstantiation, CMathEngineVectorBenchmarkTest,
	::testing::Values(
		CTestParams(
			"TestCount = 10000;"
			"RepeatCount = 10;"
			"VectorSize = 16;"
			"VectorValues = (-128..128);"
		),
		CTestParams(
			"TestCount = 10000;"
			"RepeatCount = 10;"
			"VectorSize = 25;"
			"VectorValues = (-128..128);"
		),
		CTestParams(
			"TestCount = 10000;"
			"RepeatCount = 10;"
			"VectorSize = 32;"
			"VectorValues = (-128..128);"
		),
		CTestParams(
			"TestCount = 10000;"
			"RepeatCount = 10;"
			"VectorSize = 64;"
			"VectorValues = (-128..128);"
		),
		CTestParams(
			"TestCount = 10000;"
			"RepeatCount = 10;"
			"VectorSize = 69;"
			"VectorValues = (-128..128);"
		),
		CTestParams(
			"TestCount = 10000;"
			"RepeatCount = 10;"
			"VectorSize = 100000;"
			"VectorValues = (-50..50);"
		),
		CTestParams(
			"TestCount = 10000;"
			"RepeatCount = 10;"
			"VectorSize = 999989;"
			"VectorValues = (-10..10);"
		),
		CTestParams(
			"TestCount = 1000;"
			"RepeatCount = 10;"
			"VectorSize = 1179648;"
			"VectorValues = (-1..1);"
		)
	)
);

TEST_P( CMathEngineVectorBenchmarkTest, DISABLED_Random )
{
	CTestParams testParams = GetParam();
	const int testCount = testParams.GetValue<int>( "TestCount" );
	const int repeatCount = testParams.GetValue<int>( "RepeatCount" );
	const int vectorSize = testParams.GetValue<int>( "VectorSize" );
	const CInterval valuesInterval = testParams.GetInterval( "VectorValues" );

	VectorBenchmarkParams params( /*warm-up*/-1, testCount, vectorSize, valuesInterval, 282 );
	vectorBenchmark( params );

	for( int function = 0; function < 10; ++function ) {
		params.FOut << std::endl << vectorFunctionsNames[function] << ",";

		double timeSum = 0;
		for( int test = 0; test < repeatCount; ++test ) {
			const int seed = 282 + test * 10000 + test % 3;
			params.SetNextSeedForFunction( function, seed );
			timeSum += vectorBenchmark( params );
		}
		GTEST_LOG_( INFO ) << vectorFunctionsNames[function] << "\t" << timeSum;
	}
}
