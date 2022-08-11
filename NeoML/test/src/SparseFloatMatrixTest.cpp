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
#include <MlTestCommon.h>

using namespace NeoML;
using namespace NeoMLTest;

class CSparseFloatMatrixTest : public CNeoMLTestFixture {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

TEST_F( CSparseFloatMatrixTest, AddRow )
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

TEST_F( CSparseFloatMatrixTest, AddEmptyRow )
{
	CSparseFloatMatrix matrix;

	CSparseFloatVector emptyWithBuffer( 1 );
	matrix.AddRow( emptyWithBuffer );

	EXPECT_EQ( 1, matrix.GetHeight() );
	EXPECT_EQ( 0, matrix.GetWidth() );
}

TEST_F( CSparseFloatMatrixTest, CreationFromEmptyDesc )
{
	CSparseFloatMatrix empty( CFloatMatrixDesc::Empty );
	ASSERT_EQ( empty.GetHeight(), 0 );
	ASSERT_EQ( empty.GetWidth(), 0 );
}

TEST_F( CSparseFloatMatrixTest, Grow )
{
	const int h = 5;
	const int w = 10;
	CRandom rand( 0 );
	CSparseFloatMatrix matrix( w );
	for( int i = 0; i < h; ++i ) {
		matrix.AddRow( generateRandomVector( rand, w ) );
	}
	CFloatMatrixDesc descInitial = matrix.GetDesc();
	auto columnsInitialPtr = descInitial.Columns;
	auto valuesInitialPtr = descInitial.Values;
	auto bInitialPtr = descInitial.PointerB;
	auto eInitialPtr = descInitial.PointerE;

	// modifiable desc should be the same
	CFloatMatrixDesc* modifiableDesc = matrix.CopyOnWrite();
	ASSERT_EQ( descInitial.Columns, modifiableDesc->Columns );
	ASSERT_EQ( descInitial.PointerB, modifiableDesc->PointerB );

	// the matrix should be initially (32x512) allocated, so if we grow, buffers should stay unchanged
	matrix.GrowInElements( 512 );
	auto columnsGrownPtr = matrix.GetDesc().Columns;
	auto valuesGrownPtr = matrix.GetDesc().Values;
	matrix.GrowInRows( 32 );
	auto bGrownPtr = matrix.GetDesc().PointerB;
	auto eGrownPtr = matrix.GetDesc().PointerE;
	ASSERT_EQ( columnsInitialPtr, columnsGrownPtr );
	ASSERT_EQ( valuesInitialPtr, valuesGrownPtr );
	ASSERT_EQ( bInitialPtr, bGrownPtr );
	ASSERT_EQ( eInitialPtr, eGrownPtr );

	// now grow over and check that accessed buffers have changed
	matrix.GrowInElements( 513 );
	columnsGrownPtr = matrix.GetDesc().Columns;
	valuesGrownPtr = matrix.GetDesc().Values;
	bGrownPtr = matrix.GetDesc().PointerB;
	eGrownPtr = matrix.GetDesc().PointerE;
	ASSERT_NE( columnsInitialPtr, columnsGrownPtr );
	ASSERT_NE( valuesInitialPtr, valuesGrownPtr );
	ASSERT_EQ( bInitialPtr, bGrownPtr );
	ASSERT_EQ( eInitialPtr, eGrownPtr );

	matrix.GrowInRows( 33 );
	ASSERT_EQ( columnsGrownPtr, matrix.GetDesc().Columns );
	ASSERT_EQ( valuesGrownPtr, matrix.GetDesc().Values );
	ASSERT_NE( bGrownPtr, matrix.GetDesc().PointerB );
	ASSERT_NE( eGrownPtr, matrix.GetDesc().PointerE );
}

TEST_F( CSparseFloatMatrixTest, CopyOnWrite )
{
	const int h = 5;
	const int w = 10;
	CRandom rand( 0 );
	CSparseFloatMatrix matrix( w );
	for( int i = 0; i < h; ++i ) {
		matrix.AddRow( generateRandomVector( rand, w ) );
	}
	CFloatMatrixDesc* modifiableDesc = matrix.CopyOnWrite();
	CSparseFloatMatrix matrixCopy( matrix );
	CFloatMatrixDesc* modifiableDescCopy = matrixCopy.CopyOnWrite();

	// pointers should be different
	ASSERT_NE( modifiableDesc, modifiableDescCopy );
	ASSERT_NE( modifiableDescCopy->Columns, modifiableDesc->Columns );
	ASSERT_NE( modifiableDescCopy->Values, modifiableDesc->Values );
	ASSERT_NE( modifiableDescCopy->PointerB, modifiableDesc->PointerB );
	ASSERT_NE( modifiableDescCopy->PointerE, modifiableDesc->PointerE );

	// the desc data should be equal
	ASSERT_EQ( modifiableDesc->Height, modifiableDescCopy->Height );
	ASSERT_EQ( modifiableDesc->Width, modifiableDescCopy->Width );
	ASSERT_EQ( ::memcmp( modifiableDescCopy->PointerB, modifiableDesc->PointerB, h * sizeof( int ) ), 0 );
	ASSERT_EQ( ::memcmp( modifiableDescCopy->PointerE, modifiableDesc->PointerE, h * sizeof( int ) ), 0 );
	const int elementsCount = modifiableDescCopy->PointerE[h-1];
	ASSERT_EQ( ::memcmp( modifiableDescCopy->Columns, modifiableDesc->Columns, elementsCount * sizeof( int ) ), 0 );
	ASSERT_EQ( ::memcmp( modifiableDescCopy->Values, modifiableDesc->Values, elementsCount * sizeof( float ) ), 0 );
}

TEST_F( CSparseFloatMatrixTest, CreationFromSparseAndDenseDesc )
{
	const int h = 5;
	const int w = 10;
	CSparseFloatMatrix sparseMatrix;

	CArray<float> values;
	values.SetSize( h * w );
	CArray<int> pointerB;
	pointerB.SetSize( h );
	CArray<int> pointerE;
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
	ASSERT_EQ( ::memcmp( fromSparse.PointerB, orig.PointerB, fromSparse.Height * sizeof( int ) ), 0 );
	ASSERT_EQ( ::memcmp( fromSparse.PointerE, orig.PointerE, fromSparse.Height * sizeof( int ) ), 0 );

	const int elementsCount = fromSparse.PointerE[fromSparse.Height-1];
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
	ASSERT_EQ( ::memcmp( fromSparse.PointerB, fromDense.PointerB, fromSparse.Height * sizeof( int ) ), 0 );
	ASSERT_EQ( ::memcmp( fromSparse.PointerE, fromDense.PointerE, fromSparse.Height * sizeof( int ) ), 0 );
	ASSERT_EQ( ::memcmp( fromSparse.Columns, fromDense.Columns, elementsCount * sizeof( int ) ), 0 );
	ASSERT_EQ( ::memcmp( fromSparse.Values, fromDense.Values, elementsCount * sizeof( float ) ), 0 );

	// test creation from desc with skipped first row
	--denseDesc.Height;
	pointerB.DeleteAt( 0 );
	pointerE.DeleteAt( 0 );
	denseDesc.PointerB = pointerB.GetPtr();
	denseDesc.PointerE = pointerE.GetPtr();

	CSparseFloatMatrix sparseMatrixFromDenseDescSkippedFirst( denseDesc );
	fromDense = sparseMatrixFromDenseDescSkippedFirst.GetDesc();
	const int denseElementsCount = fromDense.PointerE[fromDense.Height-1];
	ASSERT_EQ( ::memcmp( fromSparse.Columns + fromSparse.PointerB[1], fromDense.Columns,
		denseElementsCount * sizeof( int ) ), 0 );
	ASSERT_EQ( ::memcmp( fromSparse.Values + fromSparse.PointerB[1], fromDense.Values,
		denseElementsCount * sizeof( float ) ), 0 );
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

// disable this test due to occasional error on linux build VM
TEST_F( CSparseFloatMatrixTest, DISABLED_CreateHuge )
{
	const int maxLength = 128;
	const int rowsCount = 17000000;
	try {
		CSparseFloatMatrix matrix( maxLength, rowsCount );
		for( int i = 0; i < rowsCount; ++i ) {
			CFloatVector row( maxLength, 1.0 );
			matrix.AddRow( row.GetDesc() );
			if( (i+1) % 1000000 == 0 ) {
				GTEST_LOG_( INFO ) << i+1 << " rows added";
			}
		}
		GTEST_LOG_( INFO ) << rowsCount << " rows added";
		// test some random elements have been set correctly
		CRandom rand( 0 );
		const int elementsToTestCount = 1000;
		for( int i = 0; i < elementsToTestCount; ++i ) {
			const int r = rand.UniformInt( 0, rowsCount - 1 );
			const int c = rand.UniformInt( 0, maxLength - 1 );
			const int pos = r * maxLength + c;
			ASSERT_EQ( matrix.GetDesc().Columns[pos], c );
			ASSERT_DOUBLE_EQ( matrix.GetDesc().Values[pos], 1.0 );
		}
	} catch( CMemoryException* ex ) {
		GTEST_LOG_( INFO ) << "CMemoryException* caught";
		delete ex;
	} catch( CMemoryException& ) {
		GTEST_LOG_( INFO ) << "CMemoryException caught";
	} catch( CInternalError* ex ) {
		GTEST_LOG_( INFO ) << "CInternalError* caught";
		delete ex;
	} catch( CInternalError& ) {
		GTEST_LOG_( INFO ) << "CInternalError caught";
	}
}

static void serializeSparseFloatMatrix( CSparseFloatMatrix& matrix, CBaseFile& file, CArchive::TDirection direction )
{
	file.Seek( 0, CBaseFile::begin );
	CArchive archive( &file, direction );
	matrix.Serialize( archive );
}

static void compareSparseFloatMatrices( const CSparseFloatMatrix& expected, const CSparseFloatMatrix& actual )
{
	ASSERT_EQ( expected.GetHeight(), actual.GetHeight() );
	ASSERT_EQ( expected.GetWidth(), actual.GetWidth() );
	for( int rowIndex = 0; rowIndex < expected.GetHeight(); ++rowIndex ) {
		CFloatVectorDesc expectedRow = expected.GetRow( rowIndex );
		CFloatVectorDesc actualRow = actual.GetRow( rowIndex );
		ASSERT_EQ( expectedRow.Size, actualRow.Size );
		for( int elemIndex = 0; elemIndex < expectedRow.Size; ++elemIndex ) {
			ASSERT_EQ( expectedRow.Indexes[elemIndex], actualRow.Indexes[elemIndex] );
			ASSERT_EQ( expectedRow.Values[elemIndex], actualRow.Values[elemIndex] );
		}
	}
}

static void testSparseFloatMatrixSerialization( CSparseFloatMatrix& original )
{
	CMemoryFile memoryFile;
	serializeSparseFloatMatrix( original, memoryFile, CArchive::store );

	CSparseFloatMatrix deserialized;
	serializeSparseFloatMatrix( deserialized, memoryFile, CArchive::load );
	compareSparseFloatMatrices( original, deserialized );

	serializeSparseFloatMatrix( deserialized, memoryFile, CArchive::store );
	serializeSparseFloatMatrix( deserialized, memoryFile, CArchive::load );
	compareSparseFloatMatrices( original, deserialized );
}

TEST_F( CSparseFloatMatrixTest, Serialization )
{
	const int h = 5;
	const int w = 10;
	CRandom rand( 0 );
	CSparseFloatMatrix original( w );
	for( int i = 0; i < h; ++i ) {
		original.AddRow( generateRandomVector( rand, w ) );
	}
	testSparseFloatMatrixSerialization( original );
}

// Case when body == nullptr
TEST_F( CSparseFloatMatrixTest, NullBodySerialization )
{
	CSparseFloatMatrix original;
	testSparseFloatMatrixSerialization( original );
}

// Case when body != nullptr but matrix doesn't have non-zero elements
TEST_F( CSparseFloatMatrixTest, ZeroElemsSerialization )
{
	CSparseFloatMatrix original( 3 );
	testSparseFloatMatrixSerialization( original );
}

