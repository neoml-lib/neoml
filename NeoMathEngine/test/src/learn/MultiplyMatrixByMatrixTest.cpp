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
#include <MeTestCommon.h>

using namespace NeoML;
using namespace NeoMLTest;

static void multiplyMatrixByMatrixAndAddTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );

	const int firstHeight = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int firstWidth = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int secondWidth = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );

	CREATE_FILL_FLOAT_ARRAY( first, valuesInterval.Begin, valuesInterval.End, batchSize * firstHeight * firstWidth, random )
	CREATE_FILL_FLOAT_ARRAY( second, valuesInterval.Begin, valuesInterval.End, batchSize * firstWidth * secondWidth, random )

	std::vector<float> expected, get;
	expected.insert( expected.begin(), batchSize * firstHeight * secondWidth, 0.f );
	get.resize( batchSize * firstHeight * secondWidth );

	batchMultiplyMatrixByMatrixAndAddNaive( batchSize, first, second, firstHeight, firstWidth, secondWidth, expected );

	MathEngine().MultiplyMatrixByMatrix( batchSize, CARRAY_FLOAT_WRAPPER( first ), firstHeight, firstWidth,
		CARRAY_FLOAT_WRAPPER( second ), secondWidth, CARRAY_FLOAT_WRAPPER( get ), batchSize * firstHeight * secondWidth );

	for( int i = 0; i < batchSize * firstHeight * secondWidth; ++i ) {
		ASSERT_NEAR( expected[i], get[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMultiplyMatrixByMatrixTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMultiplyMatrixByMatrixTestInstantiation, CMultiplyMatrixByMatrixTest,
	::testing::Values(
		CTestParams(
			"Height = (1..50);"
			"Width = (1..50);"
			"BatchSize = (1..5);"
			"VectorSize = (1..20);"
			"Values = (-1..1);"
			"Channels = (1..5);"
			"TestCount = 100;"
		),
		CTestParams(
			"Height = (100..500);"
			"Width = (100..500);"
			"BatchSize = (1..5);"
			"VectorSize = (30..50);"
			"Values = (-1..1);"
			"Channels = (1..5);"
			"TestCount = 5;"
		)
	)
);

TEST_P( CMultiplyMatrixByMatrixTest, Random )
{
	RUN_TEST_IMPL( multiplyMatrixByMatrixAndAddTestImpl )
}
