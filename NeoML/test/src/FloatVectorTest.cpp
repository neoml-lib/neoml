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
	CFloatVector s1Vec( maxLength, s1.GetDesc() );
	CSparseFloatVectorDesc s1DenseDesc;
	s1DenseDesc.Size = maxLength;
	s1DenseDesc.Values = s1Vec.CopyOnWrite();
	for( int i = 0; i < numberOfTests; ++i ) {
		CSparseFloatVector s2 = generateRandomVector( rand, maxLength );
		CFloatVector s2Vec( maxLength, s2.GetDesc() );
		CSparseFloatVectorDesc s2DenseDesc;
		s2DenseDesc.Size = maxLength;
		s2DenseDesc.Values = s2Vec.CopyOnWrite();

		ASSERT_DOUBLE_EQ( DotProduct( s1, s2 ), results[i] );
		ASSERT_DOUBLE_EQ( DotProduct( s1.GetDesc(), s2DenseDesc ), results[i] );
		ASSERT_DOUBLE_EQ( DotProduct( s1DenseDesc, s2.GetDesc() ), results[i] );
		ASSERT_DOUBLE_EQ( DotProduct( s1DenseDesc, s2DenseDesc ), results[i] );
		ASSERT_DOUBLE_EQ( DotProduct( s1Vec, s2 ), results[i] );
		ASSERT_DOUBLE_EQ( DotProduct( s1Vec, s2DenseDesc ), results[i] );
	}
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
	CSparseFloatVectorDesc s1DenseDesc;
	s1DenseDesc.Size = maxLength;
	s1DenseDesc.Values = s1Vec.CopyOnWrite();
	for( int i = 0; i < numberOfTests; ++i ) {
		CSparseFloatVector s2 = generateRandomVector( rand, maxLength );
		ASSERT_TRUE( s2.NumberOfElements() <= maxLength );
		CFloatVector s2Vec( maxLength, s2.GetDesc() );
		CFloatVector s2VecCopy( s2Vec );
		double factor = rand.Uniform( -100, 100 );

		for( int i = 0; i < maxLength; ++i ) {
			ASSERT_EQ( s2Vec[i], GetValue( s2.GetDesc(), i ) );
		}

		s2.MultiplyAndAdd( s1, factor );
		s2Vec.MultiplyAndAdd( s1, factor );

		for( int i = 0; i < maxLength; ++i ) {
			ASSERT_EQ( s2Vec[i], GetValue( s2.GetDesc(), i ) );
		}

		auto res1 = DotProduct( s2, s1 );
		auto res2 = DotProduct( s2Vec, s1 );
		ASSERT_DOUBLE_EQ( res1, res2 );
		ASSERT_DOUBLE_EQ( res1, results[i] );

		s2VecCopy.MultiplyAndAdd( s1DenseDesc, factor );
		auto res3 = DotProduct( s2VecCopy, s1 );
		ASSERT_DOUBLE_EQ( res3, results[i] );
	}
}

TEST_F( CFloatVectorTest, GetValue )
{
	const int maxLength = 100;
	CRandom rand( 0 );
	CSparseFloatVector s1 = generateRandomVector( rand, maxLength );
	CSparseFloatVectorDesc s1Desc = s1.GetDesc();
	ASSERT_TRUE( s1.NumberOfElements() <= maxLength );
	CFloatVector s1Vec( maxLength, s1Desc );
	CSparseFloatVectorDesc s1DenseDesc;
	s1DenseDesc.Size = maxLength;
	s1DenseDesc.Values = s1Vec.CopyOnWrite();

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

		CSparseFloatVectorDesc denseDesc;
		denseDesc.Size = maxLength;
		denseDesc.Values = rowDense.CopyOnWrite();

		matrixFromDense.AddRow( denseDesc );
		CSparseFloatVectorDesc rowSparseGot = matrixFromDense.GetRow( i );
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
		CSparseFloatVectorDesc denseDesc;
		denseDesc.Size = maxLength;
		denseDesc.Values = rowDense.CopyOnWrite();

		CSparseFloatVector sparseCopy( rowSparse.GetDesc() );
		CSparseFloatVector sparseFromDenseCopy( denseDesc );

		ASSERT_EQ( sparseCopy.NumberOfElements(), sparseFromDenseCopy.NumberOfElements() );
		for( int i = 0; i < sparseCopy.NumberOfElements(); ++i ) {
			ASSERT_EQ( sparseCopy.GetDesc().Indexes[i], sparseFromDenseCopy.GetDesc().Indexes[i] );
			ASSERT_EQ( sparseCopy.GetDesc().Values[i], sparseFromDenseCopy.GetDesc().Values[i] );
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

