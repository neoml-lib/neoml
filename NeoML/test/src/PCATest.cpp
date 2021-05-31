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

CSparseFloatMatrix generateMatrix( int samples, int features, CArray<float>& values )
{
	CSparseFloatMatrix matrix( features, samples, samples * features );
	CFloatVectorDesc row;
	row.Size = features;

	int pos = 0;
	for( int i = 0; i < samples; i++, pos += features ) {
		row.Values = values.GetPtr() + pos;
		matrix.AddRow( row );
	}
	return matrix;
}

static void checkArraysEqual( const CArray<float>& expected, float* get )
{
	for( int i = 0; i < expected.Size(); i++ ) {
		ASSERT_NEAR( get[i], expected[i], 1e-4 );
	}
}

static void testExample( int samples, int features, float components,
	CArray<float> data, CArray<float> expectedSingularValues,
	CArray<float> expectedVariance, CArray<float> expectedVarianceRatio,
	float expectedNoiseVariance, CArray<float> expectedComponents,
	CArray<float> expectedTransform )
{
	CPca::CParams params;
	params.ComponentsType = CPca::TComponents::PCAC_Int;
	params.Components = components;

	const CSparseFloatMatrix& matrix = generateMatrix( samples, features, data );
	CPca pca( params );
	CFloatMatrixDesc transformed = pca.Transform( matrix.GetDesc() );
	ASSERT_EQ( samples, transformed.Height );
	ASSERT_EQ( components, transformed.Width );
	checkArraysEqual( expectedTransform, transformed.Values );

	CFloatMatrixDesc componentsMatrix = pca.GetComponents();
	ASSERT_EQ( components, componentsMatrix.Height );
	ASSERT_EQ( features, componentsMatrix.Width );
	checkArraysEqual( expectedComponents, componentsMatrix.Values );

	CArray<float> res;
	pca.GetSingularValues( res );
	ASSERT_EQ( components, res.Size() );
	checkArraysEqual( expectedSingularValues, res.GetPtr() );

	pca.GetExplainedVariance( res );
	ASSERT_EQ( components, res.Size() );
	checkArraysEqual( expectedVariance, res.GetPtr() );

	pca.GetExplainedVarianceRatio( res );
	ASSERT_EQ( components, res.Size() );
	checkArraysEqual( expectedVarianceRatio, res.GetPtr() );

	ASSERT_NEAR( expectedNoiseVariance, pca.GetNoiseVariance(), 1e-4 );
}

TEST( CPCATest, PCAExamplesTest )
{
	testExample( 4, 4, 2.,  { 2, 1, 3, 2, 2, 4, 4, 1, 2, 4, 1, 1, 4, 4, 3, 4 },
		{ 3.0407, 2.6677 }, { 3.0819, 2.3722 }, { 0.451, 0.3472 }, 0.6896,
		{ 0.5649,  0.2846,  0.1827,  0.7525, -0.0238, -0.8853,  0.3850,  0.2592 },
		{ -0.8772,  2.1003, -0.5931, -0.4300, -1.1413, -1.5853, 2.6118, -0.0849 } );

	testExample( 3, 5, 2.,  { 5, 4, 7, 5, 3, 5, 7, 8, 6, 7, 6, 4, 4, 6, 2 },
		{ 5.1339, 1.9084 }, { 13.1789, 1.8210 }, { 0.8786, 0.1214 }, 0,
		{ -0.1196,  0.4516,  0.5095,  0.0309,  0.7218, -0.2818, -0.4135,  0.7076, -0.4196, -0.2695 },
		{ -0.8146, 1.5285, 3.9683,  -0.5020, -3.1537, -1.0265 } );

	testExample( 5, 3, 3.,  { 4, 4, 8, 4, 5, 8, 5, 6, 8, 7, 5, 4, 2, 3, 3 },
		{ 5.2416, 3.8275, 1.0369 }, { 6.8686, 3.6624954, 0.2688 }, { 0.6359, 0.3391, 0.0248 }, 0,
		{ -0.2345, -0.3243, -0.9164, 0.8866,  0.3150, -0.3384, 0.3984, -0.8919,  0.2137 },
		{ -1.3611, -1.1528,  0.7605, -1.6854 , -0.8378, -0.1314, -2.2443,  0.3639, -0.6249, 1.2766,  3.1759,  0.2090, 4.0142, -1.5491, -0.2131 } );
}

TEST( CPCATest, PCAEllipseTest )
{
	CRandom rand( 42 );
	int samples = 1000;
	int features = 4;
	float components = 2;
	float a = 3;
	float b = 2;
	CArray<float> data;
	data.SetBufferSize( features );
	for( int i = 0; i < samples; i++ ) {
		float x = -a + 2 * i * a / samples;
		float y = sqrt( 1 - ( x * x ) / ( a * a ) ) * b;
		data.Add( { x, y, 0.f, 1.f } );
		data.Add( { x, -y, 0.f, 1.f } );
	}
	const CSparseFloatMatrix& matrix = generateMatrix( 2 * samples, features, data );
	CPca::CParams params;
	params.ComponentsType = CPca::TComponents::PCAC_Int;
	params.Components = components;
	CPca pca( params );
	pca.Train( matrix.GetDesc() );
	ASSERT_NEAR( 0, pca.GetNoiseVariance(), 1e-3 );
	CFloatMatrixDesc componentsMatrix = pca.GetComponents();
	ASSERT_EQ( components, componentsMatrix.Height );
	ASSERT_EQ( features, componentsMatrix.Width );

	CArray<float> expectedComponent;
	for( int row = 0; row < 2; row++ ) {
		CSparseFloatVector actualComponent( componentsMatrix.GetRow( row ) );
		expectedComponent.Empty();
		expectedComponent.Add( 0, features );
		expectedComponent[row] = 1.f;
		for( int i = 0; i < features; i++ ) {
			ASSERT_NEAR( expectedComponent[i], abs( actualComponent.GetValue( i ) ), 1e-4 );
		}
	}
}