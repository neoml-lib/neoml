/* Copyright © 2021 ABBYY Production LLC

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

#include <common.h>
#pragma hdrstop

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

class CFloatVectorTest : public CNeoMLTestFixture {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

CSparseFloatVector generateRandomVector( CRandom& rand, int maxLength = 100,
	float minValue = -100., float maxValue = 100. )
{
	CSparseFloatVector res;
	for( int i = 0; i < maxLength; ++i ) {
		int index = rand.UniformInt( 0, maxLength - 1 ) ;
		res.SetAt( index, static_cast<float>( rand.Uniform( minValue, maxValue ) ) );
	}
	return res;
}

TEST_F( CFloatVectorTest, DotProduct )
{
	const int maxLength = 100;
	const int numberOfTests = 10;
	const double results[numberOfTests] = {
		-15569.303377348762, 13946.586401587912, 2809.271572476262, -2051.5664889861182, -48118.667475214766,
		-15314.905784192648, -8661.4242571015129, -52983.736195432, -16993.9108481679, -13049.921836217276
	};

	CRandom rand( 0 );
	CSparseFloatVector s1 = generateRandomVector( rand, maxLength );
	CFloatVector s1Vec( maxLength, s1 );
	CFloatVectorDesc s1DenseDesc( s1Vec.GetDesc() );
	for( int i = 0; i < numberOfTests; ++i ) {
		CSparseFloatVector s2 = generateRandomVector( rand, maxLength );
		CFloatVector s2Vec( maxLength, s2.GetDesc() );
		CFloatVectorDesc s2DenseDesc( s2Vec.GetDesc() );

		ASSERT_DOUBLE_EQ( DotProduct( s1, s2 ), results[i] );
		ASSERT_DOUBLE_EQ( DotProduct( s1.GetDesc(), s2DenseDesc ), results[i] );
		ASSERT_DOUBLE_EQ( DotProduct( s1DenseDesc, s2.GetDesc() ), results[i] );
		ASSERT_DOUBLE_EQ( DotProduct( s1DenseDesc, s2DenseDesc ), results[i] );
		ASSERT_DOUBLE_EQ( DotProduct( s1Vec, s2 ), results[i] );
		ASSERT_DOUBLE_EQ( DotProduct( s1Vec, s2DenseDesc ), results[i] );
	}

	// test dense x sparse with no intersection
	CFloatVector minDense( 5, 2. );
	CSparseFloatVector minSparse;
	minSparse.SetAt( 5, 1 );
	minSparse.SetAt( 8, 1 );
	ASSERT_DOUBLE_EQ( DotProduct( minSparse, minDense ), 0 );

	// test empty (treated as dense with size = 0)
	ASSERT_DOUBLE_EQ( DotProduct( s1.GetDesc(), CFloatVectorDesc::Empty ), 0 );
	ASSERT_DOUBLE_EQ( DotProduct( CFloatVectorDesc::Empty, s1.GetDesc() ), 0 );
	ASSERT_DOUBLE_EQ( DotProduct( s1Vec, CFloatVectorDesc::Empty ), 0 );
}

TEST_F( CFloatVectorTest, MultiplyAndAdd )
{
	const int maxLength = 100;
	const int numberOfTests = 10;
	const double results[numberOfTests] = {
		-4586232.015529478, -14651445.851001369, -23598285.403770875, -12135458.076953262, -18803834.719856035,
		-14468612.137938973, 22336360.654346965, -15317589.717204597, -7453017.1095194202, 23853702.132571988
	};

	CRandom rand( 0 );
	CSparseFloatVector s1 = generateRandomVector( rand, maxLength );
	ASSERT_TRUE( s1.NumberOfElements() <= maxLength );
	CFloatVector s1Vec( maxLength, s1.GetDesc() );
	CFloatVectorDesc s1DenseDesc( s1Vec.GetDesc() );
	for( int i = 0; i < numberOfTests; ++i ) {
		CSparseFloatVector s2 = generateRandomVector( rand, maxLength );
		ASSERT_TRUE( s2.NumberOfElements() <= maxLength );
		CFloatVector s2Vec( maxLength, s2.GetDesc() );
		double factor = rand.Uniform( -100, 100 );

		for( int i = 0; i < maxLength; ++i ) {
			ASSERT_EQ( s2Vec[i], GetValue( s2.GetDesc(), i ) );
		}

		s2.MultiplyAndAdd( s1, factor );
		CFloatVector s2VecCopy( s2Vec );
		s2VecCopy.MultiplyAndAdd( s1, factor );

		for( int i = 0; i < maxLength; ++i ) {
			ASSERT_EQ( s2VecCopy[i], GetValue( s2.GetDesc(), i ) );
		}

		auto res1 = DotProduct( s2, s1 );
		auto res2 = DotProduct( s2VecCopy, s1 );
		ASSERT_DOUBLE_EQ( res1, res2 );
		ASSERT_DOUBLE_EQ( res1, results[i] );

		s2VecCopy = CFloatVector( s2Vec );
		s2VecCopy.MultiplyAndAdd( s1DenseDesc, factor );
		ASSERT_DOUBLE_EQ( DotProduct( s2VecCopy, s1 ), results[i] );

		s2VecCopy = CFloatVector( s2Vec );
		s2VecCopy.MultiplyAndAdd( s1Vec, factor );
		ASSERT_DOUBLE_EQ( DotProduct( s2VecCopy, s1 ), results[i] );
	}

	// test empty
	CFloatVector denseEmpty( 0 );
	denseEmpty.MultiplyAndAdd( CFloatVectorDesc::Empty, 4 );
	ASSERT_EQ( denseEmpty.Size(), 0 );
}

TEST_F( CFloatVectorTest, MultiplyAndAddExt )
{
	const int maxLength = 100;
	const int numberOfTests = 10;
	const double results[numberOfTests] = {
		8924832.7302359659, -7282416.5086089149, -476690.41539686103, -5745459.589456562, -5293083.5669151777,
		11644244.176419428, -19487143.552108571, 19096306.355894174, 21974893.570626672, 1738031.9983297046
	};

	CRandom rand( 0 );
	CSparseFloatVector s1 = generateRandomVector( rand, maxLength - 1 );
	ASSERT_TRUE( s1.NumberOfElements() < maxLength );
	CFloatVector s1Vec( maxLength - 1, s1.GetDesc() );
	CFloatVectorDesc s1DenseDesc( s1Vec.GetDesc() );
	for( int i = 0; i < numberOfTests; ++i ) {
		CSparseFloatVector s2 = generateRandomVector( rand, maxLength );
		ASSERT_TRUE( s2.NumberOfElements() <= maxLength );
		CFloatVector s2Vec( maxLength, s2.GetDesc() );
		double factor = rand.Uniform( -100, 100 );

		CFloatVector s2VecCopy( s2Vec );
		s2VecCopy.MultiplyAndAddExt( s1, factor );
		ASSERT_DOUBLE_EQ( DotProduct( s2VecCopy, s1 ), results[i] );

		s2VecCopy = CFloatVector( s2Vec );
		s2VecCopy.MultiplyAndAddExt( s1Vec, factor );
		ASSERT_DOUBLE_EQ( DotProduct( s2VecCopy, s1 ), results[i] );

		s2VecCopy = CFloatVector( s2Vec );
		s2VecCopy.MultiplyAndAddExt( s1DenseDesc, factor );
		ASSERT_DOUBLE_EQ( DotProduct( s2VecCopy, s1 ), results[i] );
	}

	// test empty
	CFloatVector denseOneElement( 1, 2 );
	denseOneElement.MultiplyAndAddExt( CFloatVectorDesc::Empty, 4 );
	ASSERT_EQ( denseOneElement.Size(), 1 );
	ASSERT_EQ( denseOneElement[0], 6 );
}

TEST_F( CFloatVectorTest, GetValue )
{
	const int maxLength = 100;
	CRandom rand( 0 );
	CSparseFloatVector s1 = generateRandomVector( rand, maxLength );
	CFloatVectorDesc s1Desc = s1.GetDesc();
	ASSERT_TRUE( s1.NumberOfElements() <= maxLength );
	CFloatVector s1Vec( maxLength, s1Desc );
	CFloatVectorDesc s1DenseDesc( s1Vec.GetDesc() );

	for( int i = -1; i < maxLength + 1; ++i ) {
		ASSERT_EQ( GetValue( s1Desc, i ), GetValue( s1DenseDesc, i ) );
	}
}

TEST_F( CFloatVectorTest, AddRowToSparseMatrix )
{
	const int maxLength = 100;
	const int rowsCount = 1000;
	CSparseFloatMatrix matrixFromDense;
	CSparseFloatMatrix matrixFromSparse;
	CRandom rand( 0 );
	for( int i = 0; i < rowsCount; ++i ) {
		CSparseFloatVector rowSparse = generateRandomVector( rand, maxLength );
		CFloatVector rowDense( maxLength, rowSparse.GetDesc() );
		CFloatVectorDesc denseDesc( rowDense.GetDesc() );

		matrixFromDense.AddRow( denseDesc );
		CFloatVectorDesc rowSparseGot = matrixFromDense.GetRow( i );
		ASSERT_EQ( rowSparse.GetDesc().Size, rowSparseGot.Size );
		for( int j = 0; j < rowSparseGot.Size; ++j ) {
			ASSERT_EQ( rowSparse.GetDesc().Indexes[j], rowSparseGot.Indexes[j] );
			ASSERT_EQ( rowSparse.GetDesc().Values[j], rowSparseGot.Values[j] );
		}

		matrixFromSparse.AddRow( rowSparse.GetDesc() );
		matrixFromSparse.GetRow( i, rowSparseGot );
		ASSERT_EQ( rowSparse.GetDesc().Size, rowSparseGot.Size );
		for( int j = 0; j < rowSparseGot.Size; ++j ) {
			ASSERT_EQ( rowSparse.GetDesc().Indexes[j], rowSparseGot.Indexes[j] );
			ASSERT_EQ( rowSparse.GetDesc().Values[j], rowSparseGot.Values[j] );
		}
	}
}

TEST_F( CFloatVectorTest, CreationSparseVectorFromDesc )
{
	const int maxLength = 100;
	const int vectorsCount = 1000;
	CRandom rand( 0 );
	for( int i = 0; i < vectorsCount; ++i ) {
		CSparseFloatVector rowSparse = generateRandomVector( rand, maxLength );
		CFloatVector rowDense( maxLength, rowSparse.GetDesc() );
		CFloatVectorDesc denseDesc( rowDense.GetDesc() );

		CSparseFloatVector sparseCopy( rowSparse.GetDesc() );
		CSparseFloatVector sparseFromDenseCopy( denseDesc );

		ASSERT_EQ( sparseCopy.NumberOfElements(), sparseFromDenseCopy.NumberOfElements() );
		for( int i = 0; i < sparseCopy.NumberOfElements(); ++i ) {
			ASSERT_EQ( sparseCopy.GetDesc().Indexes[i], sparseFromDenseCopy.GetDesc().Indexes[i] );
			ASSERT_EQ( sparseCopy.GetDesc().Values[i], sparseFromDenseCopy.GetDesc().Values[i] );
		}
	}
}

TEST_F( CFloatVectorTest, CreationSparseMatrixFromDesc )
{
	const int h = 5;
	const int w = 10;
	CSparseFloatMatrix sparseMatrix;

	CArray<float> values;
	values.SetSize( h * w );
	CArray<size_t> pointerB;
	pointerB.SetSize( h );
	CArray<size_t> pointerE;
	pointerE.SetSize( h );
	CRandom rand( 0 );
	for( int pos = 0, i = 0; i < h; ++i ) {
		CSparseFloatVector row = generateRandomVector( rand, w );
		pointerB[i] = pos;
		for( int j = 0; j < w; ++j, ++pos ) {
			NeoAssert( i*w + j == pos );
			values[pos] = GetValue( row.GetDesc(), j );
		}
		pointerE[i] = pos;
		NeoAssert( pointerB[i] + w == pointerE[i] );

		sparseMatrix.AddRow( row );
	}
	CFloatMatrixDesc orig = sparseMatrix.GetDesc();
	CSparseFloatMatrix sparseMatrixFromSparseDesc( orig );
	CFloatMatrixDesc fromSparse = sparseMatrixFromSparseDesc.GetDesc();

	// check if copied matrix equals to original
	ASSERT_EQ( fromSparse.Height, orig.Height );
	ASSERT_EQ( fromSparse.Width, orig.Width );
	ASSERT_EQ( ::memcmp( fromSparse.PointerB, orig.PointerB, fromSparse.Height * sizeof( size_t ) ), 0 );
	ASSERT_EQ( ::memcmp( fromSparse.PointerE, orig.PointerE, fromSparse.Height * sizeof( size_t ) ), 0 );

	const size_t elementsCount = fromSparse.PointerE[fromSparse.Height-1];
	ASSERT_EQ( ::memcmp( fromSparse.Columns, orig.Columns, elementsCount * sizeof( int ) ), 0 );
	ASSERT_EQ( ::memcmp( fromSparse.Values, orig.Values, elementsCount * sizeof( float ) ), 0 );

	// check if a matrix created from dense desc is equal to created from sparse one
	CFloatMatrixDesc denseDesc;
	denseDesc.Height = h;
	denseDesc.Width = w;
	denseDesc.Values = values.GetPtr();
	denseDesc.PointerB = pointerB.GetPtr();
	denseDesc.PointerE = pointerE.GetPtr();
	CSparseFloatMatrix sparseMatrixFromDenseDesc( denseDesc );
	CFloatMatrixDesc fromDense = sparseMatrixFromDenseDesc.GetDesc();

	ASSERT_EQ( fromSparse.Height, fromDense.Height );
	ASSERT_EQ( fromSparse.Width, fromDense.Width );
	ASSERT_EQ( ::memcmp( fromSparse.PointerB, fromDense.PointerB, fromSparse.Height * sizeof( size_t ) ), 0 );
	ASSERT_EQ( ::memcmp( fromSparse.PointerE, fromDense.PointerE, fromSparse.Height * sizeof( size_t ) ), 0 );
	ASSERT_EQ( ::memcmp( fromSparse.Columns, fromDense.Columns, elementsCount * sizeof( int ) ), 0 );
	ASSERT_EQ( ::memcmp( fromSparse.Values, fromDense.Values, elementsCount * sizeof( float ) ), 0 );

	// test empty
	CSparseFloatMatrix empty( CFloatMatrixDesc::Empty );
	ASSERT_EQ( empty.GetHeight(), 0 );
	ASSERT_EQ( empty.GetWidth(), 0 );

	// test creation from desc with skipped first row
	--denseDesc.Height;
	pointerB.DeleteAt( 0 );
	pointerE.DeleteAt( 0 );
	denseDesc.PointerB = pointerB.GetPtr();
	denseDesc.PointerE = pointerE.GetPtr();

	CSparseFloatMatrix sparseMatrixFromDenseDescSkippedFirst( denseDesc );
	fromDense = sparseMatrixFromDenseDescSkippedFirst.GetDesc();
	const size_t denseElementsCount = fromDense.PointerE[fromDense.Height-1];
	ASSERT_EQ( ::memcmp( fromSparse.Columns + fromSparse.PointerB[1], fromDense.Columns, denseElementsCount * sizeof( int ) ), 0 );
	ASSERT_EQ( ::memcmp( fromSparse.Values + fromSparse.PointerB[1], fromDense.Values, denseElementsCount * sizeof( float ) ), 0 );
	ASSERT_EQ( denseElementsCount + fromSparse.PointerE[0], elementsCount );

	// test the same but via GetRow
	for( int i = 0; i < fromDense.Height; ++i ) {
		auto fromDenseRow = fromDense.GetRow( i );
		auto fromSparseNextRow = fromSparse.GetRow( i+1 );
		ASSERT_EQ( fromSparseNextRow.Size, fromDenseRow.Size );
		ASSERT_EQ( ::memcmp( fromSparseNextRow.Indexes, fromDenseRow.Indexes, fromDenseRow.Size*sizeof( float ) ), 0 );
		ASSERT_EQ( ::memcmp( fromSparseNextRow.Values, fromDenseRow.Values, fromDenseRow.Size*sizeof( float ) ), 0 );

		// test GetRow from dense matrix desc
		fromDenseRow = denseDesc.GetRow( i );
		for( int i = 0; i < fromDenseRow.Size; ++i ) {
			ASSERT_EQ( GetValue( fromDenseRow, i ), GetValue( fromSparseNextRow, i ) );
		}
	}
}

TEST_F( CFloatVectorTest, Common )
{
	CSparseFloatVector sRandom;
	const int sRandomLen = 100;
	CArray<bool> sRandomArray;
	sRandomArray.InsertAt( false, 0, sRandomLen );

	CRandom rand( 0 );
	for( int i = 0; i < sRandomLen; i++ ) {
		int index = rand.UniformInt( 0, sRandomLen - 1 ) ;
		sRandom.SetAt( index, static_cast<float>( index ) );
		sRandomArray[index] = true;
	}

	for( int i = 0; i < sRandomArray.Size(); i++ ) {
		float value;
		bool res = sRandom.GetValue( i, value );
		ASSERT_EQ( sRandomArray[i], res );
		if( res ) {
			ASSERT_EQ( value, i );
		} else {
			ASSERT_EQ( value, 0 );
		}
	}

	CSparseFloatVector s1;
	s1.SetAt( 0, 1 );
	s1.SetAt( 2, 1 );
	s1.SetAt( 4, 1 );
	s1.SetAt( 6, 1 );
	s1.SetAt( 8, 1 );

	CSparseFloatVector s2;
	s2.SetAt( 1, 1 );
	s2.SetAt( 3, 1 );
	s2.SetAt( 5, 1 );
	s2.SetAt( 7, 1 );
	s2.SetAt( 9, 1 );

	CSparseFloatVector sRes;

	sRes += s1;
	ASSERT_EQ( sRes.NumberOfElements(), s1.NumberOfElements() );

	sRes += s2;
	ASSERT_EQ( sRes.NumberOfElements(), 10 );

	s1 -= s2;

	ASSERT_EQ( s1.NumberOfElements(), 10 );

	s2.MultiplyAndAdd( s1, 666 );

	ASSERT_EQ( s2.NumberOfElements(), 10 );

	CFloatVector s1Full( 10 );
	s1Full.SetAt( 1, 2.2f );
}

#if FINE_32_BIT
TEST_F( CFloatVectorTest, DISABLED_CreateHugeSparseMatrix )
#else
TEST_F( CFloatVectorTest, CreateHugeSparseMatrix )
#endif
{
	const int maxLength = 128;
	const int rowsCount = 17000000;
	try {
		CSparseFloatMatrix matrix( maxLength, rowsCount );
		for( int i = 0; i < rowsCount; ++i ) {
			CFloatVector row( maxLength, 1.0 );
			matrix.AddRow( row.GetDesc() );
		}
		GTEST_LOG_( INFO ) << rowsCount << " rows added";
		// test some random elements have been set correctly
		CRandom rand( 0 );
		const int elementsToTestCount = 1000;
		for( int i = 0; i < elementsToTestCount; ++i ) {
			const int r = rand.UniformInt( 0, rowsCount - 1 );
			const int c = rand.UniformInt( 0, maxLength - 1 );
			const size_t pos = static_cast< size_t >( r ) * maxLength + c;
			ASSERT_EQ( matrix.GetDesc().Columns[pos], c );
			ASSERT_DOUBLE_EQ( matrix.GetDesc().Values[pos], 1.0 );
		}
	} catch( CMemoryException* ex ) {
		delete ex;
	}
}

