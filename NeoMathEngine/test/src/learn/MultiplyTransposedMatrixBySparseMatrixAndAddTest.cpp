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

static void multiplyTransposedMatrixBySparseMatrixAndAddNaive( float* first, int* secondRows, int* secondColumns, float* secondValues, float* result,
	int firstHeight, int firstWidth, int secondWidth )
{
	for( int row = 0; row < firstHeight; ++row ) {
		for( int ind = secondRows[row]; ind < secondRows[row + 1]; ++ind ) {
			for( int col = 0; col < firstWidth; ++col ) {
				result[col * secondWidth + secondColumns[ind]] += first[row * firstWidth + col] * secondValues[ind];
			}
		}
	}
}

static void multiplyTransposedMatrixBySparseMatrixAndAddTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval secondWidthInterval = params.GetInterval( "SecondWidth" );
	const CInterval firstHeightInterval = params.GetInterval( "FirstHeight" );
	const CInterval firstWidthInterval = params.GetInterval( "FirstWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int firstHeight = random.UniformInt( firstHeightInterval.Begin, firstHeightInterval.End );
	const int firstWidth = random.UniformInt( firstWidthInterval.Begin, firstWidthInterval.End );
	const int secondWidth = random.UniformInt( secondWidthInterval.Begin, secondWidthInterval.End );

	std::vector<int> rows, columns;
	std::vector<float> values;
	rows.push_back( 0 );
	const int presetY = random.UniformInt( 0, firstHeight - 1 );
	const int presetX = random.UniformInt( 0, secondWidth - 1 );
	for( int i = 0; i < firstHeight; i++ ) {
		int elementsInRow = 0;
		for( int j = 0; j < secondWidth; j++ ) {
			if( ( i == presetY && j == presetX ) || random.UniformInt( 0, 2 ) != 0 ) {
				float value = static_cast< float >( random.UniformInt( valuesInterval.Begin, valuesInterval.End ) );
				columns.push_back( j );
				values.push_back( value );
				elementsInRow++;
			}
		}
		rows.push_back( elementsInRow );
	}

	CREATE_FILL_FLOAT_ARRAY( first, valuesInterval.Begin, valuesInterval.End, firstWidth * firstHeight, random )

	std::vector<float> expected, actual( firstWidth * secondWidth );
	expected.insert( expected.begin(), firstWidth * secondWidth, 0.f );

	multiplyTransposedMatrixBySparseMatrixAndAddNaive( first.data(), rows.data(), columns.data(), values.data(), expected.data(),
		firstHeight, firstWidth, secondWidth );

	CSparseMatrix sparseMatrix( MathEngine(), rows, columns, values );
	MathEngine().MultiplyTransposedMatrixBySparseMatrixAndAdd( firstHeight, firstWidth, secondWidth, CARRAY_FLOAT_WRAPPER( first ),
		sparseMatrix.Desc(), CARRAY_FLOAT_WRAPPER( actual ) );

	for( int i = 0; i < firstWidth * secondWidth; ++i ) {
		ASSERT_NEAR( expected[i], actual[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMultiplyTransposedMatrixBySparseMatrixAndAddTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMultiplyTransposedMatrixBySparseMatrixAndAddTestInstantiation, CMultiplyTransposedMatrixBySparseMatrixAndAddTest,
	::testing::Values(
		CTestParams(
			"FirstHeight = (1..100);"
			"FirstWidth = (1..100);"
			"SecondWidth = (1..100);"
			"Values = (-10..10);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CMultiplyTransposedMatrixBySparseMatrixAndAddTest, Random )
{
	RUN_TEST_IMPL( multiplyTransposedMatrixBySparseMatrixAndAddTestImpl )
}
