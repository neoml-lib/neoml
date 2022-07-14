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

void matrixSpreadRowsAddTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval resultHeightInterval = params.GetInterval( "ResultHeight" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int resultHeight = random.UniformInt( std::max( height, resultHeightInterval.Begin ), resultHeightInterval.End );

	ASSERT_TRUE( height <= resultHeight );

	CREATE_FILL_FLOAT_ARRAY( input, valuesInterval.Begin, valuesInterval.End, height * width, random )
	CREATE_FILL_FLOAT_ARRAY( actual, valuesInterval.Begin, valuesInterval.End, resultHeight * width, random )
	std::vector<float> expected = actual;

	std::vector<bool> usedRow;
	usedRow.insert( usedRow.begin(), resultHeight, false );
	std::vector<int> indices( height );

	for( int i = 0; i < height; ++i ) {
		int index = random.UniformInt( -1, resultHeight - 1 );
		if( index >= 0 ) {
			while( usedRow[index] ) {
				index = ( index + 1 ) % resultHeight;
			}
			usedRow[index] = true;
		}
		if( index >= 0 ) {
			for( int j = 0; j < width; ++j ) {
				expected[index * width + j] += input[i * width + j];
			}
		}
		indices[i] = index;
	}

	MathEngine().MatrixSpreadRowsAdd( CARRAY_FLOAT_WRAPPER( input ), height, width, CARRAY_FLOAT_WRAPPER( actual ),
		resultHeight, CARRAY_INT_WRAPPER( indices ) );

	for( int i = 0; i < resultHeight * width; ++i ) {
		ASSERT_EQ( expected[i], actual[i] );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMathEngineMatrixSpreadRowsAddTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineMatrixSpreadRowsAddTestInstantiation, CMathEngineMatrixSpreadRowsAddTest,
	::testing::Values(
		CTestParams(
			"Height = 37;"
			"Width = 12016;"
			"ResultHeight = 107;"
			"Values = (-10..10);"
			"TestCount = 1;"
		),
		CTestParams(
			"Height = (1..7);"
			"Width = (1..7);"
			"ResultHeight = (1..15);"
			"Values = (-10..10);"
			"TestCount = 1000"
		),
		CTestParams(
			"Height = (1..50);"
			"Width = (1..500);"
			"ResultHeight = (1..215);"
			"Values = (-10..10);"
			"TestCount = 100"
		)
	)
);

TEST_P( CMathEngineMatrixSpreadRowsAddTest, Random )
{
	RUN_TEST_IMPL( matrixSpreadRowsAddTestImpl )
}
