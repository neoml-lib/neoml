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

static void vectorBenchmark( int function, int testCount, int vectorSize, const CInterval& vectorValuesInterval,
	int seed, IPerformanceCounters& counters, std::ofstream& fout )
{
	CRandom random( seed );
	CREATE_FILL_FLOAT_ARRAY( input, vectorValuesInterval.Begin, vectorValuesInterval.End, vectorSize, random )
	CREATE_FILL_FLOAT_ARRAY( second, vectorValuesInterval.Begin, vectorValuesInterval.End, vectorSize, random )
	std::vector<float> result( vectorSize, 0 );

	float zero = 0;
	float multiplier = static_cast<float>( random.Uniform( 1, vectorValuesInterval.End ) );
	CFloatWrapper zeroVal( MathEngine(), &zero, 1 );
	CFloatWrapper mulVal( MathEngine(), &multiplier, 1 );
	ASSERT_EXPR( ( ( CConstFloatHandle )mulVal ).GetValueAt( 0 ) > 0 );

	if( function == -1 ) { // warm-up
		return;
	}

	counters.Synchronise();

	for( int i = 0; i < testCount; ++i ) {
		switch( function ) {
			case 0:
				MathEngine().VectorCopy( CARRAY_FLOAT_WRAPPER( result ), CARRAY_FLOAT_WRAPPER( input ), vectorSize );
				break;
			case 1:
				MathEngine().VectorFill( CARRAY_FLOAT_WRAPPER( result ), vectorSize, mulVal );
				break;
			case 2:
				MathEngine().VectorAdd( CARRAY_FLOAT_WRAPPER( input ), CARRAY_FLOAT_WRAPPER( second ),
					CARRAY_FLOAT_WRAPPER( result ), vectorSize );
				break;
			case 3:
				MathEngine().VectorAddValue( CARRAY_FLOAT_WRAPPER( input ), CARRAY_FLOAT_WRAPPER( result ),
					vectorSize, mulVal );
				break;
			case 4:
				MathEngine().VectorMultiply( CARRAY_FLOAT_WRAPPER( input ),
					CARRAY_FLOAT_WRAPPER( second ), vectorSize, mulVal );
				break;
			case 5:
				MathEngine().VectorEltwiseMultiply( CARRAY_FLOAT_WRAPPER( input ),
					CARRAY_FLOAT_WRAPPER( second ), CARRAY_FLOAT_WRAPPER( result ), vectorSize );
				break;
			case 6:
				MathEngine().VectorEltwiseMultiplyAdd( CARRAY_FLOAT_WRAPPER( input ),
					CARRAY_FLOAT_WRAPPER( second ), CARRAY_FLOAT_WRAPPER( result ), vectorSize );
				break;
			case 7:
				MathEngine().VectorReLU( CARRAY_FLOAT_WRAPPER( input ), CARRAY_FLOAT_WRAPPER( result ),
					vectorSize, zeroVal ); //Threshold == 0
				break;
			case 8:
				MathEngine().VectorReLU( CARRAY_FLOAT_WRAPPER( input ), CARRAY_FLOAT_WRAPPER( result ),
					vectorSize, mulVal ); //Threshold > 0
				break;
			case 9:
				MathEngine().VectorHSwish( CARRAY_FLOAT_WRAPPER( input ), CARRAY_FLOAT_WRAPPER( result ),
					vectorSize );
				break;
			default:
				ASSERT_EXPR( false );
		}
	}

	counters.Synchronise();
	const double time = double( counters[0].Value ) / 1000000 / testCount; // average time in milliseconds

	GTEST_LOG_( INFO ) << vectorFunctionsNames[function] << ", " << time;
	fout << vectorFunctionsNames[function] << "," << time << "\n";
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
			"TestCount = 100000;"
			"RepeatCount = 10;"
			"VectorSize = 11796480;"
			"VectorValues = (-1..1);"
		)
	)
);

TEST_P( CMathEngineVectorBenchmarkTest, DISABLED_Random )
{
	CTestParams params = GetParam();

	const int testCount = params.GetValue<int>( "TestCount" );
	const int repeatCount = params.GetValue<int>( "RepeatCount" );
	const int vectorSize = params.GetValue<int>( "VectorSize" );
	const CInterval vectorValuesInterval = params.GetInterval( "VectorValues" );

	IPerformanceCounters* counters = MathEngine().CreatePerformanceCounters();
	std::ofstream fout( "VectorBenchmarkTest.csv", std::ios::app );
	fout << "---------------------------\n";

	vectorBenchmark( /*warm-up*/-1, testCount, vectorSize, vectorValuesInterval, 282, *counters, fout);

	for( int function = 0; function < 10; ++function ) {
		for( int test = 0; test < repeatCount; ++test ) {
			const int seed = 282 + test * 10000 + test % 3;
			vectorBenchmark( function, testCount, vectorSize, vectorValuesInterval, seed, *counters, fout );
		}
	}
	delete counters;
}
