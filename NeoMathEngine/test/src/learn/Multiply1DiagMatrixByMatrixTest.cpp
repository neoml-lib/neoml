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

using namespace NeoML;
using namespace NeoMLTest;

static void multiply1DiagMatrixByMatrixTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );

	const int firstSize = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int secondWidth = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );

	CREATE_FILL_FLOAT_ARRAY( firstData, valuesInterval.Begin, valuesInterval.End, firstSize, random )
	CREATE_FILL_FLOAT_ARRAY( secondData, valuesInterval.Begin, valuesInterval.End, batchSize * firstSize * secondWidth, random )

	std::vector<float> expected, get;
	expected.resize( batchSize * firstSize * secondWidth );
	get.resize( batchSize * firstSize * secondWidth );

	int index = 0;
	for( int b = 0; b < batchSize; ++b ) {
		for( int j = 0; j < firstSize; ++j ) {
			for( int i = 0; i < secondWidth; ++i, ++index ) {
				expected[index] = firstData[j] * secondData[index];
			}
		}
	}

	MathEngine().Multiply1DiagMatrixByMatrix( batchSize, CARRAY_FLOAT_WRAPPER( firstData ), firstSize,
		CARRAY_FLOAT_WRAPPER( secondData ), secondWidth, CARRAY_FLOAT_WRAPPER( get ), static_cast<int>( get.size() ) );

	for( int i = 0; i < batchSize * firstSize * secondWidth; ++i ) {
		ASSERT_NEAR( expected[i], get[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMultiply1DiagMatrixByMatrixTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMultiply1DiagMatrixByMatrixTestInstantiation, CMultiply1DiagMatrixByMatrixTest,
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

TEST_P( CMultiply1DiagMatrixByMatrixTest, Random )
{
	RUN_TEST_IMPL( multiply1DiagMatrixByMatrixTestImpl )
}
