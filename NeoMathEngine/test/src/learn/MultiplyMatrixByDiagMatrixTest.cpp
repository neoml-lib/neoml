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

static void multiplyMatrixByDiagMatrixTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchInterval = params.GetInterval( "Batch" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int batch = random.UniformInt( batchInterval.Begin, batchInterval.End );
	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int matrixSize = height * width;
	const int dataSize = batch * matrixSize;

	CREATE_FILL_FLOAT_ARRAY( firstData, valuesInterval.Begin, valuesInterval.End, dataSize, random )
	CREATE_FILL_FLOAT_ARRAY( secondData, valuesInterval.Begin, valuesInterval.End, batch * width, random )

	std::vector<float> expected, actual;
	expected.resize( dataSize );
	actual.resize( dataSize );

	for( int firstMatrixOffset : { 0, matrixSize } ) {
		for( int secondMatrixOffset : { 0, width } ) {
			int index = 0;
			for( int b = 0; b < batch; ++b ) {
				for( int i = 0; i < height; ++i ) {
					for( int j = 0; j < width; ++j, ++index ) {
						expected[index] = firstData[b * firstMatrixOffset + i * width + j]
							* secondData[b * secondMatrixOffset + j];
					}
				}
			}

			MathEngine().MultiplyMatrixByDiagMatrix( batch, CARRAY_FLOAT_WRAPPER( firstData ), height, width,
				firstMatrixOffset, CARRAY_FLOAT_WRAPPER( secondData ), secondMatrixOffset,
				CARRAY_FLOAT_WRAPPER( actual ), dataSize );

			for( int i = 0; i < dataSize; ++i ) {
				ASSERT_NEAR( expected[i], actual[i], 1e-3 );
			}
		}
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMultiplyMatrixByDiagMatrixTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMultiplyMatrixByDiagMatrixTestInstantiation, CMultiplyMatrixByDiagMatrixTest,
	::testing::Values(
		CTestParams(
			"Batch = (1..1);"
			"Height = (10..10);"
			"Width = (10..10);"
			"Values = (-1..1);"
			"TestCount = 1;"
		),
		CTestParams(
			"Batch = (10..10);"
			"Height = (1..1);"
			"Width = (10..10);"
			"Values = (-1..1);"
			"TestCount = 1;"
		),
		CTestParams(
			"Batch = (10..10);"
			"Height = (10..10);"
			"Width = (1..1);"
			"Values = (-1..1);"
			"TestCount = 1;"
		),
		CTestParams(
			"Batch = (10..10);"
			"Height = (1..1);"
			"Width = (1..1);"
			"Values = (-1..1);"
			"TestCount = 1;"
		),
		CTestParams(
			"Batch = (1..1);"
			"Height = (10..10);"
			"Width = (1..1);"
			"Values = (-1..1);"
			"TestCount = 1;"
		),
		CTestParams(
			"Batch = (1..1);"
			"Height = (1..1);"
			"Width = (10..10);"
			"Values = (-1..1);"
			"TestCount = 1;"
		),
		CTestParams(
			"Batch = (1..10);"
			"Height = (1..50);"
			"Width = (1..50);"
			"Values = (-1..1);"
			"TestCount = 100;"
		),
		CTestParams(
			"Batch = (1..5);"
			"Height = (100..500);"
			"Width = (100..500);"
			"Values = (-1..1);"
			"TestCount = 5;"
		)
	)
);

TEST_P( CMultiplyMatrixByDiagMatrixTest, Random )
{
	RUN_TEST_IMPL( multiplyMatrixByDiagMatrixTestImpl )
}
