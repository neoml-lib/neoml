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

static void multiplyMatrixByTransposedMatrixAndAddNaive( const std::vector<float>& first, const std::vector<float>& second,
	int firstHeight, int firstWidth, int secondHeight, std::vector<float>& result )
{
	for( int i = 0; i < firstHeight; ++i ) {
		for( int j = 0; j < secondHeight; ++j ) {
			for( int k = 0; k < firstWidth; ++k ) {
				result[i * secondHeight + j] += first[i * firstWidth + k] * second[j * firstWidth + k];
			}
		}
	}
}

static void multiplyMatrixByTransposedMatrixTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int secondHeight = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int firstHeight = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int firstWidth = random.UniformInt( widthInterval.Begin, widthInterval.End );

	CREATE_FILL_FLOAT_ARRAY( a, valuesInterval.Begin, valuesInterval.End, firstHeight * firstWidth, random )
	CREATE_FILL_FLOAT_ARRAY( b, valuesInterval.Begin, valuesInterval.End, firstWidth * secondHeight, random )

	std::vector<float> exp;
	exp.insert( exp.begin(), firstHeight * secondHeight, 0.f );
	multiplyMatrixByTransposedMatrixAndAddNaive( a, b, firstHeight, firstWidth, secondHeight, exp );

	std::vector<float> result;
	result.resize( firstHeight * secondHeight );
	MathEngine().MultiplyMatrixByTransposedMatrix( CARRAY_FLOAT_WRAPPER( a ), firstHeight, firstWidth, firstWidth,
		CARRAY_FLOAT_WRAPPER( b ), secondHeight, firstWidth, CARRAY_FLOAT_WRAPPER( result ), secondHeight, firstHeight * secondHeight );

	for( int i = 0; i < firstHeight * secondHeight; ++i ) {
		ASSERT_NEAR( exp[i], result[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMultiplyMatrixByTransposedMatrixTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMultiplyMatrixByTransposedMatrixTestInstantiation, CMultiplyMatrixByTransposedMatrixTest,
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

TEST_P( CMultiplyMatrixByTransposedMatrixTest, Random )
{
	RUN_TEST_IMPL( multiplyMatrixByTransposedMatrixTestImpl )
}
