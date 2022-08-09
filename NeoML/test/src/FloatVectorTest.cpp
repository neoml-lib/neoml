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

class CFloatVectorTest : public CNeoMLTestFixture {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

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

TEST_F( CFloatVectorTest, CreationFromDesc )
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

