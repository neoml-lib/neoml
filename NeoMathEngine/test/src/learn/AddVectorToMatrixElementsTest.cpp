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

static void addVectorToMatrixElementsTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );

	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );

	CREATE_FILL_FLOAT_ARRAY( vector1, valuesInterval.Begin, valuesInterval.End, height, random )
	CREATE_FILL_FLOAT_ARRAY( matrix, valuesInterval.Begin, valuesInterval.End, height * width, random )
	CREATE_FILL_INT_ARRAY( indices, 0, width - 1, height, random )
	std::vector<float> initMatrix;
	initMatrix = matrix;

	MathEngine().AddVectorToMatrixElements( CARRAY_FLOAT_WRAPPER( matrix ), height, width, CARRAY_INT_WRAPPER( indices ), CARRAY_FLOAT_WRAPPER( vector1 ) );

	for( int h = 0; h < height; ++h ) {
		initMatrix[h * width + indices[h]] += vector1[h];
	}
	for( size_t i = 0; i < matrix.size(); ++i ) {
		ASSERT_NEAR( initMatrix[i], matrix[i], 1e-3 );
	}
	
	matrix = initMatrix;
	CREATE_FILL_FLOAT_ARRAY( vector2, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
	CREATE_FILL_INT_ARRAY( rowIndices, 0, height - 1, vectorSize, random )
	CREATE_FILL_INT_ARRAY( columnIndices, 0, width - 1, vectorSize, random )
	MathEngine().AddVectorToMatrixElements( CARRAY_FLOAT_WRAPPER( matrix ), height, width, CARRAY_INT_WRAPPER( rowIndices ),
		CARRAY_INT_WRAPPER( columnIndices ), CARRAY_FLOAT_WRAPPER( vector2 ), vectorSize );

	for( int i = 0; i < vectorSize; ++i ) {
		initMatrix[rowIndices[i] * width + columnIndices[i]] += vector2[i];
	}
	for( size_t i = 0; i < matrix.size(); ++i ) {
		ASSERT_NEAR( initMatrix[i], matrix[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CAddVectorToMatrixElementsTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CAddVectorToMatrixElementsTestInstantiation, CAddVectorToMatrixElementsTest,
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

TEST_P( CAddVectorToMatrixElementsTest, Random )
{
	RUN_TEST_IMPL( addVectorToMatrixElementsTestImpl )
}
