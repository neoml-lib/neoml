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

static void setVectorToMatrixElementsImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval matrixHeightInterval = params.GetInterval( "MatrixHeight" );
	const CInterval matrixWidthInterval = params.GetInterval( "MatrixWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int matrixWidth = random.UniformInt( matrixWidthInterval.Begin, matrixWidthInterval.End );
	const int matrixHeight = random.UniformInt( matrixHeightInterval.Begin, matrixHeightInterval.End );
	int vectorSize = random.UniformInt( 1, matrixWidth * matrixHeight );
	
	CREATE_FILL_INT_ARRAY( columnIndices, 0, matrixWidth - 1, vectorSize, random )
	CREATE_FILL_INT_ARRAY( rowIndices, 0, matrixHeight - 1, vectorSize, random )
	CREATE_FILL_FLOAT_ARRAY( vector, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
	CREATE_FILL_FLOAT_ARRAY( matrix, valuesInterval.Begin, valuesInterval.End, matrixHeight * matrixWidth, random )

	for( int i = 0; i < vectorSize; i++ ) {
		int j = i + 1;
		while( j < vectorSize ) {
			if ( columnIndices[i] == columnIndices[j] && rowIndices[i] == rowIndices[j] ) {
				columnIndices.erase( columnIndices.begin() + j );
				rowIndices.erase( rowIndices.begin() + j );
				vector.erase( vector.begin() + j );
				vectorSize--;
			} else {
				j++;
			}
		}
	}
	
	std::vector<float> expected;
	expected = matrix;
	for( int i = 0; i < vectorSize; i++ ) {
		expected[rowIndices[i] * matrixWidth + columnIndices[i]] = vector[i];
	}
	
	MathEngine().SetVectorToMatrixElements( CARRAY_FLOAT_WRAPPER( matrix ), matrixHeight, matrixWidth, CARRAY_INT_WRAPPER( rowIndices ), CARRAY_INT_WRAPPER( columnIndices ), CARRAY_FLOAT_WRAPPER( vector ), vectorSize );
	
	for( int i = 0; i < matrixHeight * matrixWidth; i++ ) {
		ASSERT_NEAR( expected[i], matrix[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineSetVectorToMatrixElementsTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineSetVectorToMatrixElementsTestInstantiation, CMathEngineSetVectorToMatrixElementsTest,
	::testing::Values(
		CTestParams(
			"MatrixHeight = (10..100);"
			"MatrixWidth = (10..100);"
			"Values = (-50..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CMathEngineSetVectorToMatrixElementsTest, Random )
{
	RUN_TEST_IMPL( setVectorToMatrixElementsImpl )
}
