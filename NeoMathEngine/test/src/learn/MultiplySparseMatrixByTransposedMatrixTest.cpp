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

static void multiplySparseMatrixByTransposedMatrixAndAddNaive( int* firstRows, int* firstColumns, float* firstValues, float* second, float* result,
	int firstHeight, int firstWidth, int secondHeight )
{
	for( int row = 0; row < firstHeight; ++row ) {
		for( int ind = firstRows[row]; ind < firstRows[row + 1]; ++ind ) {
			for( int col = 0; col < secondHeight; ++col ) {
				result[row * secondHeight + col] += firstValues[ind] * second[col * firstWidth + firstColumns[ind]];
			}
		}
	}
}

static void multiplySparseMatrixByTransposedMatrixTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval secondHeightInterval = params.GetInterval( "SecondHeight" );
	const CInterval firstHeightInterval = params.GetInterval( "FirstHeight" );
	const CInterval firstWidthInterval = params.GetInterval( "FirstWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int firstHeight = random.UniformInt( firstHeightInterval.Begin, firstHeightInterval.End );
	const int firstWidth = random.UniformInt( firstWidthInterval.Begin, firstWidthInterval.End );
	const int secondHeight = random.UniformInt( secondHeightInterval.Begin, secondHeightInterval.End );

	std::vector<int> rows, columns;
	std::vector<float> values;
	rows.push_back( 0 );
	const int presetY = random.UniformInt( 0, firstHeight - 1 );
	const int presetX = random.UniformInt( 0, firstWidth - 1 );
	for( int i = 0; i < firstHeight; i++ ) {
		for( int j = 0; j < firstWidth; j++ ) {
			if( ( i == presetY && j == presetX ) || random.UniformInt( 0, 2 ) != 0 ) {
				float value = static_cast< float >( random.UniformInt( valuesInterval.Begin, valuesInterval.End ) );
				columns.push_back( j );
				values.push_back( value );
			}
		}
		rows.push_back( static_cast<int>( values.size() ) );
	}

	CREATE_FILL_FLOAT_ARRAY( second, valuesInterval.Begin, valuesInterval.End, firstWidth * secondHeight, random )

	std::vector<float> expected, actual( firstHeight * secondHeight );
	expected.insert( expected.begin(), firstHeight * secondHeight, 0.f );

	multiplySparseMatrixByTransposedMatrixAndAddNaive( rows.data(), columns.data(), values.data(), second.data(), expected.data(),
		firstHeight, firstWidth, secondHeight );

	CSparseMatrix sparseMatrix( MathEngine(), rows, columns, values );
	MathEngine().MultiplySparseMatrixByTransposedMatrix( firstHeight, firstWidth, secondHeight, sparseMatrix.Desc(),
		CARRAY_FLOAT_WRAPPER( second ), CARRAY_FLOAT_WRAPPER( actual ) );

	for( int i = 0; i < firstHeight * secondHeight; ++i ) {
		ASSERT_NEAR( expected[i], actual[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMultiplySparseMatrixByTransposedMatrixTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMultiplySparseMatrixByTransposedMatrixTestInstantiation, CMultiplySparseMatrixByTransposedMatrixTest,
	::testing::Values(
		CTestParams(
			"FirstHeight = (1..100);"
			"FirstWidth = (1..100);"
			"SecondHeight = (1..100);"
			"Values = (-10..10);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CMultiplySparseMatrixByTransposedMatrixTest, Random )
{
	RUN_TEST_IMPL( multiplySparseMatrixByTransposedMatrixTestImpl )
}
