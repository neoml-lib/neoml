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

	CPtr<const CDnnBlob> top16ax = TopK(ax, 16);
	CPtr<const CDnnBlob> top16bx = bx;

	CPtr<const CDnnBlob> loss = Add(top16ax, top16bx);

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[16] = { 506, 902, 493, 493, 493, 492, 492, 21, 415, 20, 16, 293, 394, 291, 393, 292 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_EQ( lossRes[i], lossData[i] );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 502, 402, 4, 6, 8, 8, 10, 515, 415, 20, 22, 779, 879, 776, 878, 782 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_EQ( gradRes[i], gradData[i] );
	}
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

TEST_F( CAutoDiffTest, TestSum )
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

	CPtr<const CDnnBlob> loss = Sum( Add(top4ax, top4bx) );

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[1] = { 3.574 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
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
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
	}
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

	CPtr<const CDnnBlob> top16ax = TopK(ax, 16);
	CPtr<const CDnnBlob> top16bx = bx;

	CPtr<const CDnnBlob> loss = Mul(top16ax, top16bx);

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[16] = { 505, 200901, 982, 1470, 1956, 1952, 2435, 110,
						  4050, 100, 55, 1156, 1560, 864, 782, 291 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_EQ( lossRes[i], lossData[i] );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { 201406, 201192, 1764, 2334, 3112, 3512, 2490, 615,
								 8100, 200, 165, 2626, 3516, 2816, 3217, 1273 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_EQ( gradRes[i], gradData[i] );
	}
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

	CPtr<const CDnnBlob> top16ax = TopK(ax, 16);
	CPtr<const CDnnBlob> top16bx = bx;

	CPtr<const CDnnBlob> loss = Div(top16ax, top16bx);

	CArray<float> lossData;
	lossData.SetSize( loss->GetDataSize() );
	loss->CopyTo( lossData.GetPtr() );

	float lossRes[16] = { 505, 1.2493765, 245.4999, 163.3333,
						  122.2499, 121.99999, 97.40000, 1.100000,
						  0.02469135, 1.000000, 0.45454543, 0.01384083,
						  0.01025641, 0.010416667, 0.00511509, 0.00343642 };
	for( int i = 0; i < _countof(lossRes); i++ ) {
		ASSERT_NEAR( lossRes[i], lossData[i], 1e-3 );
	}

	CPtr<const CDnnBlob> grad = tape.Gradient( *loss, *x );

	CArray<float> gradData;
	gradData.SetSize( grad->GetDataSize() );
	grad->CopyTo( gradData.GetPtr() );

	float gradRes[VectorSize] = { -503.750549, -1.24594, -245.4948, -163.3229,
								  -122.2361, -121.9897, -96.94545, 503.89996,
								  0, 0, 0.6454546, 163.3195,
								  122.2397, 121.9895, 97.394882, 245.49655 };
	for( int i = 0; i < _countof(gradRes); i++ ) {
		ASSERT_NEAR( gradRes[i], gradData[i], 1e-3 );
	}
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
