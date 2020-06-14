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

static void addMatrixElementsToVectorTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int vectorSize = random.UniformInt( height, 2 * height );

	CREATE_FILL_FLOAT_ARRAY( vector, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
	CREATE_FILL_FLOAT_ARRAY( matrix, valuesInterval.Begin, valuesInterval.End, height * width, random )
	CREATE_FILL_INT_ARRAY( indices, 0, width - 1, height, random )
	std::vector<float> initVector;
	initVector = vector;

	MathEngine().AddMatrixElementsToVector( CARRAY_FLOAT_WRAPPER( matrix ), height, width, CARRAY_INT_WRAPPER( indices ), CARRAY_FLOAT_WRAPPER( vector ), vectorSize );

	for( int i = 0; i < height; ++i ) {
		ASSERT_NEAR( vector[i], initVector[i] + matrix[i * width + indices[i]], 1e-3 );
	}

	CREATE_FILL_INT_ARRAY( rowIndices, 0, height - 1, vectorSize, random )
	CREATE_FILL_INT_ARRAY( columnIndices, 0, width - 1, vectorSize, random )
	vector = initVector;

	MathEngine().AddMatrixElementsToVector( CARRAY_FLOAT_WRAPPER( matrix ), height, width, CARRAY_INT_WRAPPER( rowIndices ),
		CARRAY_INT_WRAPPER( columnIndices ), CARRAY_FLOAT_WRAPPER( vector ), vectorSize );

	for( int i = 0; i < vectorSize; ++i ) {
		ASSERT_NEAR( vector[i], initVector[i] + matrix[rowIndices[i] * width + columnIndices[i]], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CAddMatrixElementsToVectorTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CAddMatrixElementsToVectorTestInstantiation, CAddMatrixElementsToVectorTest,
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

TEST_P( CAddMatrixElementsToVectorTest, Random )
{
	RUN_TEST_IMPL( addMatrixElementsToVectorTestImpl )
}
