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

#include <functional>

using namespace NeoML;
using namespace NeoMLTest;

struct CAutoDiffTestParam {

};

class CAutoDiffTest : public CNeoMLTestFixture, public ::testing::WithParamInterface<CAutoDiffTestParam> {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

TEST_F( CAutoDiffTest, TestConst )
{
	CGradientTape tape;

	CPtr<const CDnnBlob> const1 = Const( MathEngine(), 42.42, {100, 200, 1, 50} );

	CArray<float> const1Data;
	const1Data.SetSize( const1->GetDataSize() );
	const1->CopyTo( const1Data.GetPtr() );

	ASSERT_EQ( const1->GetChannelsCount(), 50 );
	ASSERT_EQ( const1->GetWidth(), 200 );
	ASSERT_EQ( const1->GetHeight(), 100 );
	for( int i = 0; i < const1Data.Size(); i++ ) {
		ASSERT_NEAR( 42.42, const1Data[i], 1e-4 );
	}

	const1 = Const( MathEngine(), const1Data.GetPtr(), {const1Data.Size()} );
	const1Data.SetSize( const1->GetDataSize() );
	const1->CopyTo( const1Data.GetPtr() );

	ASSERT_EQ( const1->GetChannelsCount(), 50 * 200 * 100 );
	for( int i = 0; i < const1Data.Size(); i++ ) {
		ASSERT_NEAR( 42.42, const1Data[i], 1e-4 );
	}
}

static void jacobianTestImpl( const CArray<float>& vectorA, const CArray<float>& vectorB, const CArray<float>& expectedLoss,
	const CArray<float>& expectedGradient,
	std::function<CPtr<const CDnnBlob>( CPtr<const CDnnBlob>&, CPtr<CDnnBlob>&, CPtr<CDnnBlob>& )> lossFunc,
	const std::initializer_list<int>& dimensionsX,
	const std::initializer_list<int>& dimensionsA,
	const std::initializer_list<int>& dimensionsB )
{
	CGradientTape tape;

	CArray<float> xData;
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensionsX ) );
	xData.InsertAt( 1.0, 0, xBlob->GetDataSize() );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	CPtr<CDnnBlob> a( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensionsA ) );
	a->CopyFrom( vectorA.GetPtr() );

	CPtr<CDnnBlob> b( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensionsB ) );
	b->CopyFrom( vectorB.GetPtr() );

	CPtr<const CDnnBlob> loss = lossFunc( x, a, b );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	for( int i = 0; i < expectedLoss.Size(); i++ ) {
		ASSERT_NEAR( expectedLoss[i], lossData[i], 1e-4 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	for( int i = 0; i < expectedGradient.Size(); i++ ) {
		ASSERT_NEAR( expectedGradient[i], gradData[i], 1e-4 );
	}
}

static void jacobianCommonTestImpl( const CArray<float>& vectorA, const CArray<float>& vectorB, const CArray<float>& expectedLoss,
	const CArray<float>& expectedGradient, const std::initializer_list<int>& dimensions,
	std::function<CPtr<const CDnnBlob>( CPtr<const CDnnBlob>&, CPtr<CDnnBlob>&, CPtr<CDnnBlob>& )> lossFunc )
{
	jacobianTestImpl( vectorA, vectorB, expectedLoss, expectedGradient, lossFunc,
		dimensions, dimensions, dimensions );
}

TEST_F( CAutoDiffTest, TestStack )
{
	jacobianCommonTestImpl(
		{ 0.28, 0.3, 0.2 , 0.73, 0.73, 0.72, 0.33, 0.11,
		  0.08, 0.49, 0.09, 0.76, 0.05, 0.65, 0.28, 0.97 },
		{ 0.1 , 0.08, 0.46, 0.26, 0.23, 0.08, 0.33, 0.34,
		  0.1 , 0.4, 0.37, 0.41, 0.32, 0.53, 0.43, 0.82 },
		{ 9.32, 9.24, 7.44, 14.06 },
		{ 1.38, 1.38, 1.66, 1.99, 1.96, 1.8, 1.66, 1.45,
		  1.18, 1.89, 1.46, 2.17, 1.37, 2.18, 1.71, 2.79 },
		{ 2, 1, 1, 4, 2 },
		[]( CPtr<const CDnnBlob>& x, CPtr<CDnnBlob>& a, CPtr<CDnnBlob>& b ) {
			CPtr<const CDnnBlob> sum = Add( Mul(x, a), Mul(b, x) );
			CObjectArray<CDnnBlob> arr;
			arr.Add( const_cast<CDnnBlob*>( x.Ptr() ) );
			arr.Add( a.Ptr() );
			arr.Add( b.Ptr() );
			arr.Add( const_cast<CDnnBlob*>( sum.Ptr() ) );
			return Sum( Stack( arr, 1 ), { 1, 3 } );
		}
	);
}

TEST_F( CAutoDiffTest, TestReshape )
{
	jacobianCommonTestImpl(
		{ 0.28, 0.3, 0.2 , 0.73, 0.73, 0.72, 0.33, 0.11,
		  0.08, 0.49, 0.09, 0.76, 0.05, 0.65, 0.28, 0.97 },
		{ 0.1 , 0.08, 0.46, 0.26, 0.23, 0.08, 0.33, 0.34,
		  0.1 , 0.4, 0.37, 0.41, 0.32, 0.53, 0.43, 0.82 },
		{ 0.5475, 0.95625 },
		{ 0.0475, 0.0475, 0.0825, 0.12375, 0.12, 0.1, 0.0825, 0.05625,
		  0.0225, 0.11125, 0.0575, 0.14625, 0.04625, 0.1475, 0.08875, 0.22375 },
		{ 2, 1, 1, 4, 2 },
		[]( CPtr<const CDnnBlob>& x, CPtr<CDnnBlob>& a, CPtr<CDnnBlob>& b ) {
			CPtr<CDnnBlob> sum = const_cast<CDnnBlob*>( Add( Mul(x, a), Mul(b, x) ).Ptr() );
			Reshape( sum, { 1, 1, 1, 8, 2, 1, 1 } );
			return Mean( sum, { 3 } );
		}
	);
}

TEST_F( CAutoDiffTest, TestBroadcast1 )
{
	jacobianTestImpl(
		{ 1 },
		{ 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 4, 4 },
		[]( CPtr<const CDnnBlob>& x, CPtr<CDnnBlob>& a, CPtr<CDnnBlob>& b ) {
			return Broadcast( x, {1, 2, 1, 2, 1, 2, 1} );
		},
		{1, 1, 1, 2, 1, 1, 1},
		{ 1 },
		{ 1 }
	);
}

TEST_F( CAutoDiffTest, TestBroadcast2 )
{
	jacobianTestImpl(
		{ 0.4, 0.5 },
		{ 0.72, 0.96, 0.49, 0.41, 0.39, 0.96, 0.95, 0.8,
		  0.57, 0.24, 0.32, 0.17, 0.93, 0.89, 0.18, 0.79 },
		{ 16.97 },
		{ 2.48, 1.7, 2.15, 2.55, 1.81, 1.49, 2.82, 1.97 },
		[]( CPtr<const CDnnBlob>& x, CPtr<CDnnBlob>& a, CPtr<CDnnBlob>& b ) {
			CPtr<const CDnnBlob> ax = Mul( x, a );
			CPtr<const CDnnBlob> bx = Mul( b, x );
			CPtr<const CDnnBlob> sum = Add( ax, bx );
			return Sum( sum, { 0, 3, 4 } );
		},
		{2, 1, 1, 4, 1},
		{2, 1, 1, 1, 1},
		{2, 1, 1, 4, 2}
	);
}

TEST_F( CAutoDiffTest, TestBroadcast3 )
{
	jacobianTestImpl(
		{ 0.52, 0.73, 0.76, 0.05 },
		{ 0.25, 0.08 },
		{ 7.12, 5.12, 5.76, 3.76 },
		{ 5.44, 5.44, 5.44, 5.44 },
		[]( CPtr<const CDnnBlob>& x, CPtr<CDnnBlob>& a, CPtr<CDnnBlob>& b ) {
			CPtr<const CDnnBlob> ax = Mul( x, a );
			CPtr<const CDnnBlob> bx = Mul( b, x );
			CPtr<const CDnnBlob> sum = Add( ax, bx );
			return Sum( sum, { 0, 1, 3 } );
		},
		{2, 1, 1, 2, 1},
		{1, 2, 1, 1, 2},
		{1, 1, 2, 1, 1}
	);
}

TEST_F( CAutoDiffTest, TestAdd1 )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 501, 1, 2, 3, 4, 4, 5, 505, 10, 10, 11, 490, 489, 488, 487, 491 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 1, 401, 2, 3, 4, 4, 5, 10, 405, 10, 11, 289, 390, 288, 391, 291 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> top4ax = TopK(ax, 4);
	CPtr<const CDnnBlob> top4bx = TopK(bx, 4);

	CPtr<const CDnnBlob> loss = Add(top4ax, top4bx);

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[4] = { 910, 902, 882, 880 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_EQ( lossRes[i], lossData[i] );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 501, 401, 0, 0, 0, 0, 0, 505, 405, 0, 0, 490, 390, 0, 391, 491 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_EQ( gradRes[i], gradData[i] );
	}
}

TEST_F( CAutoDiffTest, TestAdd2 )
{
	jacobianCommonTestImpl(
		{ 501, 1, 2, 3, 4, 4, 5, 505, 10, 10, 11, 490, 489, 488, 487, 491 },
		{ 1, 401, 2, 3, 4, 4, 5, 10, 405, 10, 11, 289, 390, 288, 391, 291 },
		{ 506, 902, 493, 493, 493, 492, 492, 21, 415, 20, 16, 293, 394, 291, 393, 292 },
		{ 502, 402, 4, 6, 8, 8, 10, 515, 415, 20, 22, 779, 879, 776, 878, 782 },
		{ 1, 1, 1, 1, 1, 1, 16 },
		[]( CPtr<const CDnnBlob>& x, CPtr<CDnnBlob>& a, CPtr<CDnnBlob>& b ) {
			CPtr<const CDnnBlob> ax = Mul(x, a);
			CPtr<const CDnnBlob> bx = Mul(b, x);

			CPtr<const CDnnBlob> top16ax = TopK(ax, 16);
			CPtr<const CDnnBlob> top16bx = bx;

			return Add(top16ax, top16bx);
		}
	);
}

TEST_F( CAutoDiffTest, TestSub1 )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.501, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.505,
								  0.010, 0.010, 0.011, 0.490, 0.489, 0.488, 0.487, 0.491 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.001, 0.401, 0.002, 0.003, 0.004, 0.004, 0.005, 0.010,
								  0.405, 0.010, 0.011, 0.289, 0.390, 0.288, 0.391, 0.291 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> top4ax = TopK(ax, 4);
	CPtr<const CDnnBlob> top4bx = TopK(bx, 4);

	CPtr<const CDnnBlob> loss = Sub(top4ax, top4bx);

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[4] = { 0.099999, 0.099999, 0.099999, 0.10000 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 0.501, -0.401, 0, 0,
								  0, 0, 0, 0.505,
								  -0.405, 0, 0, 0.49,
								  -0.39, 0, -0.391, 0.491 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
	}
}

TEST_F( CAutoDiffTest, TestSum1 )
{
	for( int axis : { -1, 6 } ) {
		CGradientTape tape;

		const int VectorSize = 16;

		CArray<float> xData;
		xData.InsertAt( 1.0, 0, VectorSize );
		CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
		xBlob->CopyFrom( xData.GetPtr() );
		CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

		float valuesA[VectorSize] = { 0.501, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.505,
			0.010, 0.010, 0.011, 0.490, 0.489, 0.488, 0.487, 0.491 };
		CPtr<CDnnBlob> a( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
		a->CopyFrom( valuesA );

		float valuesB[VectorSize] = { 0.001, 0.401, 0.002, 0.003, 0.004, 0.004, 0.005, 0.010,
			0.405, 0.010, 0.011, 0.289, 0.390, 0.288, 0.391, 0.291 };
		CPtr<CDnnBlob> b( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
		b->CopyFrom( valuesB );

		CPtr<const CDnnBlob> ax = Mul( x, a );
		CPtr<const CDnnBlob> bx = Mul( b, x );

		CPtr<const CDnnBlob> top4ax = TopK( ax, 4 );
		CPtr<const CDnnBlob> top4bx = TopK( bx, 4 );

		CArray<int> axes;
		if( axis != -1 ) {
			axes.Add( axis );
		}
		CPtr<const CDnnBlob> loss = Sum( Add( top4ax, top4bx ), axes );

		CArray<float> lossData;
		lossData.SetSize( loss->GetDataSize() );
		loss->CopyTo( lossData.GetPtr() );

		float lossRes[1] = { 3.574 };
		for( int i = 0; i < _countof( lossRes ); i++ ) {
			ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
		}

		CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

		CArray<float> gradData;
		gradData.SetSize( grad->GetDataSize() );
		grad->CopyTo( gradData.GetPtr() );

		float gradRes[VectorSize] = { 0.501, 0.401, 0, 0,
			0, 0, 0, 0.505,
			0.405, 0, 0, 0.49,
			0.39, 0, 0.391, 0.491 };
		for( int i = 0; i < _countof( gradRes ); i++ ) {
			ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
		}
	}
}

TEST_F( CAutoDiffTest, TestSum2 )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	auto dimensions = { 2, 1, 1, 4, 2 };
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.28, 0.3, 0.2 , 0.73, 0.73, 0.72, 0.33, 0.11,
								  0.08, 0.49, 0.09, 0.76, 0.05, 0.65, 0.28, 0.97 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.1 , 0.08, 0.46, 0.26, 0.23, 0.08, 0.33, 0.34,
								  0.1 , 0.4, 0.37, 0.41, 0.32, 0.53, 0.43, 0.82 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> loss = Sum( Add( ax, bx ), { 1, 3 } );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[4] = { 2.66, 2.62, 1.72, 5.03 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 0.38, 0.38, 0.66, 0.99,
								  0.96, 0.8, 0.66, 0.45,
								  0.18, 0.89, 0.46, 1.17,
								  0.37, 1.18, 0.71, 1.79 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
	}
}

TEST_F( CAutoDiffTest, TestSum3 )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	auto dimensions = { 2, 1, 1, 4, 2 };
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.28, 0.3, 0.2 , 0.73, 0.73, 0.72, 0.33, 0.11,
								  0.08, 0.49, 0.09, 0.76, 0.05, 0.65, 0.28, 0.97 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.1 , 0.08, 0.46, 0.26, 0.23, 0.08, 0.33, 0.34,
								  0.1 , 0.4, 0.37, 0.41, 0.32, 0.53, 0.43, 0.82 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> loss = TopK(Sum( Add( ax, bx ), { 0, 1, 4 } ), 2);

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[2] = { 3.61, 3.31 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 0, 0, 0, 0,
								  0.96, 0.8, 0.66, 0.45,
								  0, 0, 0, 0,
								  0.37, 1.18, 0.71, 1.79 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
	}
}

TEST_F( CAutoDiffTest, TestCumSum1 )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	auto dimensions = { 2, 1, 1, 4, 2 };
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.28, 0.3, 0.2 , 0.73, 0.73, 0.72, 0.33, 0.11,
								  0.08, 0.49, 0.09, 0.76, 0.05, 0.65, 0.28, 0.97 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.1 , 0.08, 0.46, 0.26, 0.23, 0.08, 0.33, 0.34,
								  0.1 , 0.4, 0.37, 0.41, 0.32, 0.53, 0.43, 0.82 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> loss = CumSum( Add( ax, bx ), 3 );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[16] = { 0.38, 0.38, 1.04, 1.37,
		                  2., 2.17, 2.66, 2.62,
	                      0.18, 0.89, 0.64, 2.06,
	                      1.01, 3.24, 1.72, 5.03 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 1.52, 1.52, 1.98, 2.97,
								  1.92, 1.6, 0.66, 0.45,
								  0.72, 3.56, 1.38, 3.51,
								  0.74, 2.36, 0.71, 1.79 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
	}
}

TEST_F( CAutoDiffTest, TestCumSum2 )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	auto dimensions = { 2, 1, 1, 4, 2 };
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.28, 0.3, 0.2 , 0.73, 0.73, 0.72, 0.33, 0.11,
								  0.08, 0.49, 0.09, 0.76, 0.05, 0.65, 0.28, 0.97 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.1 , 0.08, 0.46, 0.26, 0.23, 0.08, 0.33, 0.34,
								  0.1 , 0.4, 0.37, 0.41, 0.32, 0.53, 0.43, 0.82 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> loss = CumSum( Mean( Add( ax, bx ), { 4 } ), 3 );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[8] = { 0.38, 1.205, 2.085, 2.64,
		                 0.535, 1.35, 2.125, 3.375 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 0.76, 0.76, 0.99, 1.485,
								  0.96, 0.8, 0.33, 0.225,
								  0.36, 1.78, 0.69, 1.755,
								  0.37, 1.18, 0.355, 0.895 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
	}
}

TEST_F( CAutoDiffTest, TestMean1 )
{
	for( int axis : { -1, 6 } ) {
		CGradientTape tape;

		const int VectorSize = 16;

		CArray<float> xData;
		xData.InsertAt( 1.0, 0, VectorSize );
		CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
		xBlob->CopyFrom( xData.GetPtr() );
		CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

		float valuesA[VectorSize] = { 0.69, 0.83, 0.54, 0.25, 0.36, 0.73, 0.18, 0.42, 0.14, 0.09, 0.25,
			0.69, 0.57, 0.44, 0.39, 0.38 };
		CPtr<CDnnBlob> a( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
		a->CopyFrom( valuesA );

		float valuesB[VectorSize] = { 0.37, 0.44, 0.91, 0.68, 0.55, 0.1 , 0.4 , 0.46, 0.09, 0.31, 0.22,
			0.2 , 0.27, 0.14, 0.03, 0.75 };
		CPtr<CDnnBlob> b( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
		b->CopyFrom( valuesB );

		CPtr<const CDnnBlob> ax = Mul( x, a );
		CPtr<const CDnnBlob> bx = Mul( b, x );

		CPtr<const CDnnBlob> top4ax = TopK( ax, 4 );
		CPtr<const CDnnBlob> top4bx = TopK( bx, 4 );

		CArray<int> axes;
		if( axis != -1 ) {
			axes.Add( axis );
		}
		CPtr<const CDnnBlob> loss = Mean( Add( top4ax, top4bx ), axes );

		CArray<float> lossData;
		lossData.SetSize( loss->GetDataSize() );
		loss->CopyTo( lossData.GetPtr() );

		float lossRes[1] = { 1.4575 };
		for( int i = 0; i < _countof( lossRes ); i++ ) {
			ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
		}

		CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

		CArray<float> gradData;
		gradData.SetSize( grad->GetDataSize() );
		grad->CopyTo( gradData.GetPtr() );

		float gradRes[VectorSize] = { 0.1725, 0.2075, 0.2275, 0.17,
			0.1375, 0.1825, 0., 0.,
			0., 0., 0., 0.1725,
			0., 0., 0., 0.1875 };
		for( int i = 0; i < _countof( gradRes ); i++ ) {
			ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
		}
	}
}

TEST_F( CAutoDiffTest, TestMean2 )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	auto dimensions = { 2, 1, 1, 4, 2 };
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.28, 0.3, 0.2 , 0.73, 0.73, 0.72, 0.33, 0.11,
								  0.08, 0.49, 0.09, 0.76, 0.05, 0.65, 0.28, 0.97 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.1, 0.08, 0.46, 0.26, 0.23, 0.08, 0.33, 0.34,
								  0.1, 0.4, 0.37, 0.41, 0.32, 0.53, 0.43, 0.82 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> loss = TopK( Mean( Add( ax, bx ), { 2, 3 } ), 2 );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[2] = { 1.2575, 0.665 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 0.095, 0, 0.165, 0,
								  0.24, 0, 0.165, 0,
								  0, 0.2225, 0, 0.2925,
								  0, 0.295, 0, 0.4475 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
	}
}

TEST_F( CAutoDiffTest, TestMean3 )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	auto dimensions = { 2, 1, 1, 4, 2 };
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.28, 0.3, 0.2 , 0.73, 0.73, 0.72, 0.33, 0.11,
								  0.08, 0.49, 0.09, 0.76, 0.05, 0.65, 0.28, 0.97 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.1, 0.08, 0.46, 0.26, 0.23, 0.08, 0.33, 0.34,
								  0.1, 0.4, 0.37, 0.41, 0.32, 0.53, 0.43, 0.82 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateTensor( MathEngine(), CT_Float, dimensions ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Sum( Mul( x, a ), { 3 } );
	CPtr<const CDnnBlob> bx = Sum( Mul( b, x ), { 3 } );

	CPtr<const CDnnBlob> loss = Mean( Add( ax, bx ), { 2, 4 } );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[2] = { 2.64, 3.375 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 0.19, 0.19, 0.33, 0.495,
								  0.48, 0.4, 0.33, 0.225,
								  0.09, 0.445, 0.23, 0.585,
								  0.185, 0.59, 0.355, 0.895 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
	}
}

TEST_F( CAutoDiffTest, TestMean4 )
{
	jacobianCommonTestImpl(
		{ 0.28, 0.3, 0.2 , 0.73, 0.73, 0.72, 0.33, 0.11,
		  0.08, 0.49, 0.09, 0.76, 0.05, 0.65, 0.28, 0.97 },
		{ 0.1 , 0.08, 0.46, 0.26, 0.23, 0.08, 0.33, 0.34,
		  0.1 , 0.4, 0.37, 0.41, 0.32, 0.53, 0.43, 0.82 },
		{ 0.84375 },
		{ 0, 0, 0, 0, 0, 0, 0, 0,
		  0.0225, 0.11125, 0.0575, 0.14625, 0.04625, 0.1475, 0.08875, 0.22375 },
		{ 2, 1, 1, 4, 2 },
		[]( CPtr<const CDnnBlob>& x, CPtr<CDnnBlob>& a, CPtr<CDnnBlob>& b ) {
			CPtr<const CDnnBlob> ax = Mul(x, a);
			CPtr<const CDnnBlob> bx = Mul(b, x);
			return TopK(Mean( Add( ax, bx ), { 3, 4 } ), 1);
		}
	);
}

TEST_F( CAutoDiffTest, TestClip )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.501, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.505,
								  0.010, 0.010, 0.011, 0.490, 0.489, 0.488, 0.487, 0.491 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.001, 0.401, 0.002, 0.003, 0.004, 0.004, 0.005, 0.010,
								  0.405, 0.010, 0.011, 0.289, 0.390, 0.288, 0.391, 0.291 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> top4ax = TopK(ax, 4);
	CPtr<const CDnnBlob> top4bx = TopK(bx, 4);

	CPtr<const CDnnBlob> loss = Clip( Add(top4ax, top4bx), 0.8815, 0.903 );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[4] = { 0.903, 0.901999, 0.881999, 0.8815 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 0.501, 0.401, 0, 0,
								  0, 0, 0, 0,
								  0, 0, 0, 0,
								  0, 0, 0.391, 0.491 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
	}
}

TEST_F( CAutoDiffTest, TestMax )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.501, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.505,
								  0.010, 0.010, 0.011, 0.490, 0.489, 0.488, 0.487, 0.491 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.001, 0.401, 0.002, 0.003, 0.004, 0.004, 0.005, 0.010,
								  0.405, 0.010, 0.011, 0.289, 0.390, 0.288, 0.391, 0.291 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> top4ax = Max(ax, 0.3);
	CPtr<const CDnnBlob> top4bx = Max(bx, 0.2);

	CPtr<const CDnnBlob> loss = Add(top4ax, top4bx);

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[VectorSize] = { 0.701, 0.701, 0.5, 0.5,
								  0.5, 0.5, 0.5, 0.705,
								  0.705, 0.5, 0.5, 0.779,
								  0.878999, 0.776, 0.878, 0.782 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 0.501, 0.401, 0, 0,
								  0, 0, 0, 0.505,
								  0.405, 0, 0, 0.779,
								  0.878999, 0.776, 0.878, 0.782 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
	}
}

TEST_F( CAutoDiffTest, TestMult1 )
{
	jacobianCommonTestImpl(
		{ 501, 1, 2, 3, 4, 4, 5, 505, 10, 10, 11, 490, 489, 488, 487, 491 },
		{ 1, 401, 2, 3, 4, 4, 5, 10, 405, 10, 11, 289, 390, 288, 391, 291 },
		{ 505, 200901, 982, 1470, 1956, 1952, 2435, 110,
		  4050, 100, 55, 1156, 1560, 864, 782, 291 },
		{ 201406, 201192, 1764, 2334, 3112, 3512, 2490, 615,
		  8100, 200, 165, 2626, 3516, 2816, 3217, 1273 },
		{ 1, 1, 1, 1, 1, 1, 16 },
		[]( CPtr<const CDnnBlob>& x, CPtr<CDnnBlob>& a, CPtr<CDnnBlob>& b ) {
			CPtr<const CDnnBlob> ax = Mul(x, a);
			CPtr<const CDnnBlob> bx = Mul(b, x);

			CPtr<const CDnnBlob> top16ax = TopK(ax, 16);
			CPtr<const CDnnBlob> top16bx = bx;

			return Mul(top16ax, top16bx);
		}
	);
}

TEST_F( CAutoDiffTest, TestDiv1 )
{
	CGradientTape tape;

	const int VectorSize = 4;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.501, 0.001, 0.002, 0.003 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.001, 0.401, 0.002, 0.003 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = TopK( Mul(x, a), 4 );
	CPtr<const CDnnBlob> bx = TopK( Mul(b, x), 4 );

	CPtr<const CDnnBlob> loss = Div(ax, bx);

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[VectorSize] = { 1.2493765, 1., 1., 1. };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 0.2493765, -0.2493765, 0, 0 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
	}
}

TEST_F( CAutoDiffTest, TestDiv2 )
{
	jacobianCommonTestImpl(
		{ 0.501, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.505,
		  0.010, 0.010, 0.011, 0.490, 0.489, 0.488, 0.487, 0.491 },
		{ 0.001, 0.401, 0.002, 0.003, 0.004, 0.004, 0.005, 0.010,
		  0.405, 0.010, 0.011, 0.289, 0.390, 0.288, 0.391, 0.291 },
		{ 505, 1.2493765, 245.4999, 163.3333,
		  122.2499, 121.99999, 97.40000, 1.100000,
		  0.02469135, 1.000000, 0.45454543, 0.01384083,
		  0.01025641, 0.010416667, 0.00511509, 0.00343642 },
		{ -503.750549, -1.24594, -245.4948, -163.3229,
		  -122.2361, -121.9897, -96.94545, 503.89996,
		  0, 0, 0.6454546, 163.3195,
		  122.2397, 121.9895, 97.394882, 245.49655 },
		{ 1, 1, 1, 1, 1, 1, 16 },
		[]( CPtr<const CDnnBlob>& x, CPtr<CDnnBlob>& a, CPtr<CDnnBlob>& b ) {
			CPtr<const CDnnBlob> ax = Mul(x, a);
			CPtr<const CDnnBlob> bx = Mul(b, x);

			CPtr<const CDnnBlob> top16ax = TopK(ax, 16);
			CPtr<const CDnnBlob> top16bx = bx;

			return Div(top16ax, top16bx);
		}
	);
}

TEST_F( CAutoDiffTest, TestLog )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.501, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.505,
								  0.010, 0.010, 0.011, 0.490, 0.489, 0.488, 0.487, 0.491 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.001, 0.401, 0.002, 0.003, 0.004, 0.004, 0.005, 0.010,
								  0.405, 0.010, 0.011, 0.289, 0.390, 0.288, 0.391, 0.291 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> top4ax = TopK(ax, 4);
	CPtr<const CDnnBlob> top4bx = TopK(bx, 4);

	CPtr<const CDnnBlob> loss = Log( Add(top4ax, top4bx) );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[4] = { -0.09431072, -0.10314082, -0.12556326, -0.12783338 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-4 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 0.5554324, 0.44456762, 0, 0, 0, 0, 0, 0.55494505, 0.44505498, 0, 0, 0.5568182, 0.44318178, 0,
		0.44331068, 0.5566894 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-4 );
	}
}

TEST_F( CAutoDiffTest, TestExp )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.501, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.505,
								  0.010, 0.010, 0.011, 0.490, 0.489, 0.488, 0.487, 0.491 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.001, 0.401, 0.002, 0.003, 0.004, 0.004, 0.005, 0.010,
								  0.405, 0.010, 0.011, 0.289, 0.390, 0.288, 0.391, 0.291 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> top4ax = TopK(ax, 4);
	CPtr<const CDnnBlob> top4bx = TopK(bx, 4);

	CPtr<const CDnnBlob> loss = Exp( Add(top4ax, top4bx) );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[4] = { 2.484333, 2.464527, 2.415726, 2.410899 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-4 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 1.234728, 0.988275, 0, 0,
								  0, 0, 0, 1.254582,
								  1.006150, 0, 0, 1.181340,
								  0.940250, 0, 0.944548, 1.186121 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-4 );
	}
}

TEST_F( CAutoDiffTest, TestAbs )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { -0.501, -0.001, -0.002, -0.003, -0.004, -0.004, -0.005, -0.505,
								  -0.010, -0.010, -0.011, -0.490, -0.489, -0.488, -0.487, -0.491 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.001, 0.401, 0.002, 0.003, 0.004, 0.004, 0.005, 0.010,
								  0.405, 0.010, 0.011, 0.289, 0.390, 0.288, 0.391, 0.291 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> top4ax = TopK(ax, 4);
	CPtr<const CDnnBlob> top4bx = TopK(bx, 4);

	CPtr<const CDnnBlob> loss = Abs( Add(top4ax, top4bx) );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[4] = { 0.404, 0.399, 0.388, 0.385999 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-4 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 0., 0.4, -0.002, -0.003,
								  -0.004, 0., 0., 0.,
								  0.405, 0., 0., 0.,
								  0.39, 0, 0.391, 0. };

	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-4 );
	}
}

TEST_F( CAutoDiffTest, TestNeg )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.501, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.505,
								  0.010, 0.010, 0.011, 0.490, 0.489, 0.488, 0.487, 0.491 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.001, 0.401, 0.002, 0.003, 0.004, 0.004, 0.005, 0.010,
								  0.405, 0.010, 0.011, 0.289, 0.390, 0.288, 0.391, 0.291 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> top4ax = TopK(ax, 4);
	CPtr<const CDnnBlob> top4bx = TopK(bx, 4);

	CPtr<const CDnnBlob> loss = Neg( Add(top4ax, top4bx) );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[4] = { -0.9099999, -0.90199995, -0.88199997, -0.88 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-4 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { -0.501, -0.401, 0, 0, 0, 0, 0, -0.505, -0.405, 0, 0, -0.49, -0.39, 0, -0.391, -0.491 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-4 );
	}
}

TEST_F( CAutoDiffTest, TestTopK )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	for( int i = 0; i < VectorSize; i++ ) {
		xData.Add( (float)i );
	}
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
	xBlob->CopyFrom( xData.GetPtr() );

	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	CPtr<const CDnnBlob> loss = TopK( x, 4 );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[4] = { 15, 14, 13, 12 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_EQ( lossRes[i], lossData[i] );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_EQ( gradRes[i], gradData[i] );
	}
}

TEST_F( CAutoDiffTest, TestBinaryCrossEntropy )
{
	CGradientTape tape;

	const int VectorSize = 16;

	CArray<float> xData;
	xData.InsertAt( 1.0, 0, VectorSize );
	CPtr<CDnnBlob> xBlob( CDnnBlob::CreateVector( MathEngine(), CT_Float, xData.Size() ) );
	xBlob->CopyFrom( xData.GetPtr() );
	CPtr<const CDnnBlob> x = tape.Variable( *xBlob );

	float valuesA[VectorSize] = { 0.501, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.505,
		0.010, 0.010, 0.011, 0.490, 0.489, 0.488, 0.487, 0.491 };
	CPtr<CDnnBlob> a( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	a->CopyFrom( valuesA );

	float valuesB[VectorSize] = { 0.001, 0.401, 0.002, 0.003, 0.004, 0.004, 0.005, 0.010,
		0.405, 0.010, 0.011, 0.289, 0.390, 0.288, 0.391, 0.291 };
	CPtr<CDnnBlob> b( CDnnBlob::CreateVector( MathEngine(), CT_Float, VectorSize ) );
	b->CopyFrom( valuesB );

	CPtr<const CDnnBlob> ax = Mul(x, a);
	CPtr<const CDnnBlob> bx = Mul(b, x);

	CPtr<const CDnnBlob> top4ax = TopK(ax, 4);
	CPtr<const CDnnBlob> top4bx = TopK(bx, 4);

	CPtr<const CDnnBlob> loss = BinaryCrossEntropy( top4ax, top4bx, false );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[4] = { 0.7134542, 0.71354485, 0.7135042, 0.71347916 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-4 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[16] = { 0.2010512, -0.166944, 0., 0.,
						  0., 0., 0., 0.194260,
						  -0.168067, 0., 0., 0.219183,
						  -0.163934, 0., -0.164203, 0.217567};
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-4 );
	}
}
