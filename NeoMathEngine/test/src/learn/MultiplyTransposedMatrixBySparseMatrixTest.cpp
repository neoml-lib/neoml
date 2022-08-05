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

static void multiplyTransposedMatrixBySparseMatrixNaive( float* first, int* secondRows, int* secondColumns, float* secondValues, float* result,
	int firstHeight, int firstWidth, int resultWidth )
{
	for( int row = 0; row < firstHeight; ++row ) {
		for( int ind = secondRows[row]; ind < secondRows[row + 1]; ++ind ) {
			for( int col = 0; col < firstWidth; ++col ) {
				result[col * resultWidth + secondColumns[ind]] += first[col] * secondValues[ind];
			}
		}
		first += firstWidth;
	}
}

static void multiplyTransposedMatrixByTransposedSparseMatrixNaive( float* first, int* secondRows, int* secondColumns, float* secondValues, float* result,
	int firstWidth, int resultWidth )
{
	for( int row = 0; row < resultWidth; ++row ) {
		for( int ind = secondRows[row]; ind < secondRows[row + 1]; ++ind ) {
			float* firstPtr = first + secondColumns[ind] * firstWidth;
			for( int col = 0; col < firstWidth; ++col ) {
				result[col * resultWidth + row] += firstPtr[col] * secondValues[ind];
			}
		}
	}
}

static void multiplyTransposedMatrixBySparseMatrixTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval resultWidthInterval = params.GetInterval( "ResultWidth" );
	const CInterval firstHeightInterval = params.GetInterval( "FirstHeight" );
	const CInterval firstWidthInterval = params.GetInterval( "FirstWidth" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int isSparseTransposed = random.UniformInt( 0, 2 );
	const int firstHeight = random.UniformInt( firstHeightInterval.Begin, firstHeightInterval.End );
	const int firstWidth = random.UniformInt( firstWidthInterval.Begin, firstWidthInterval.End );
	const int resultWidth = random.UniformInt( resultWidthInterval.Begin, resultWidthInterval.End );
	const int secondHeight = isSparseTransposed ? resultWidth : firstHeight;
	const int secondWidth = isSparseTransposed ? firstHeight : resultWidth;

	std::vector<int> rows, columns;
	std::vector<float> values;
	rows.push_back( 0 );
	const int presetY = random.UniformInt( 0, secondHeight - 1 );
	const int presetX = random.UniformInt( 0, secondWidth - 1 );
	for( int i = 0; i < secondHeight; i++ ) {
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

	std::vector<float> expected, actual( firstWidth * resultWidth );
	expected.insert( expected.begin(), firstWidth * resultWidth, 0.f );

	if( isSparseTransposed ) {
		multiplyTransposedMatrixByTransposedSparseMatrixNaive( first.data(), rows.data(), columns.data(), values.data(), expected.data(),
			firstWidth, resultWidth );

		CSparseMatrix sparseMatrix( MathEngine(), rows, columns, values );
		MathEngine().MultiplyTransposedMatrixBySparseMatrix( firstHeight, firstWidth, resultWidth, CARRAY_FLOAT_WRAPPER( first ),
			sparseMatrix.Desc(), CARRAY_FLOAT_WRAPPER( actual ), true );
	} else {
		multiplyTransposedMatrixBySparseMatrixNaive( first.data(), rows.data(), columns.data(), values.data(), expected.data(),
			firstHeight, firstWidth, resultWidth );

		CSparseMatrix sparseMatrix( MathEngine(), rows, columns, values );
		MathEngine().MultiplyTransposedMatrixBySparseMatrix( firstHeight, firstWidth, resultWidth, CARRAY_FLOAT_WRAPPER( first ),
			sparseMatrix.Desc(), CARRAY_FLOAT_WRAPPER( actual ), false );
	}

	for( int i = 0; i < firstWidth * resultWidth; ++i ) {
		ASSERT_NEAR( expected[i], actual[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMultiplyTransposedMatrixBySparseMatrixTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMultiplyTransposedMatrixBySparseMatrixTestInstantiation, CMultiplyTransposedMatrixBySparseMatrixTest,
	::testing::Values(
		CTestParams(
			"FirstHeight = (1..100);"
			"FirstWidth = (1..100);"
			"ResultWidth = (1..100);"
			"Values = (-10..10);"
			"TestCount = 200;"
		)
	)
);

TEST_P( CMultiplyTransposedMatrixBySparseMatrixTest, Random )
{
	if( MathEngine().GetType() != MET_Cpu ) {
		return;
	}
	RUN_TEST_IMPL( multiplyTransposedMatrixBySparseMatrixTestImpl )
}