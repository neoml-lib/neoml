/* Copyright © 2017-2024 ABBYY

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

static void multiplyVectorByTransposedLookupVectorAndAddToTableTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );

	int lookupMatrixSize = random.UniformInt( heightInterval.Begin, 3 * heightInterval.End );

	CREATE_FILL_FLOAT_ARRAY( matrixTable, valuesInterval.Begin, valuesInterval.End, lookupMatrixSize * width, random )
	std::vector<float> expectedTable;
	expectedTable = matrixTable;
	CREATE_FILL_INT_ARRAY( matrixIndices, 0, lookupMatrixSize - 1, batchSize * height, random )
	CREATE_FILL_FLOAT_ARRAY( first, valuesInterval.Begin, valuesInterval.End, batchSize * height, random )

	int lookupVectorSize = random.UniformInt( heightInterval.Begin, 3 * heightInterval.End );

	CREATE_FILL_FLOAT_ARRAY( secondTable, valuesInterval.Begin, valuesInterval.End, lookupVectorSize * width, random )

	std::vector<float> secondData;
	std::vector<int> secondIndices;
	secondData.resize( batchSize * width );
	secondIndices.resize( batchSize );
	for( int i = 0; i < batchSize; ++i ) {
		secondIndices[i] = random.UniformInt( 0, lookupVectorSize - 1 );
		for( int j = 0; j < width; ++j ) {
			secondData[i * width + j] = secondTable[secondIndices[i] * width + j];
		}
	}

	MathEngine().MultiplyVectorByTransposedLookupVectorAndAddToTable( batchSize,
		CARRAY_FLOAT_WRAPPER( matrixTable ), lookupMatrixSize, width, CARRAY_INT_WRAPPER( matrixIndices ),
		CARRAY_FLOAT_WRAPPER( first ), height,
		CLookupVector( CARRAY_FLOAT_WRAPPER( secondTable ), lookupVectorSize, width, CARRAY_INT_WRAPPER( secondIndices ) ) );

	std::vector<float> tempData;
	tempData.insert( tempData.begin(), batchSize * height * width, 0 );
	batchMultiplyMatrixByMatrixAndAddNaive( batchSize, first, secondData, height, 1, width, tempData );
	for( size_t i = 0; i < matrixIndices.size(); ++i ) {
		for( int j = 0; j < width; ++j ) {
			expectedTable[matrixIndices[i] * width + j] += tempData[i * width + j];
		}
	}

	for( size_t i = 0; i < matrixTable.size(); ++i ) {
		EXPECT_NEAR( expectedTable[i], matrixTable[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMultiplyVectorByTransposedLookupVectorAndAddToTableTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMultiplyVectorByTransposedLookupVectorAndAddToTableTestInstantiation, CMultiplyVectorByTransposedLookupVectorAndAddToTableTest,
	::testing::Values(
		CTestParams(
			"Height = (1..50);"
			"Width = (1..50);"
			"BatchSize = (1..5);"
			"Values = (-1..1);"
			"TestCount = 100;"
		),
		CTestParams(
			"Height = (100..500);"
			"Width = (100..500);"
			"BatchSize = (1..5);"
			"Values = (-1..1);"
			"TestCount = 5;"
		)
	)
);

TEST_P( CMultiplyVectorByTransposedLookupVectorAndAddToTableTest, Random )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}

	RUN_TEST_IMPL( multiplyVectorByTransposedLookupVectorAndAddToTableTestImpl )
}
