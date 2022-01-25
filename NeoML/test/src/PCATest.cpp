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

static void checkArraysEqual( const CArray<float>& expected, const float* get )
{
	for( int i = 0; i < expected.Size(); i++ ) {
		ASSERT_NEAR( get[i], expected[i], 5e-3 );
	}
}

static void svdTestExample( int samples, int features, int components,
	CArray<float> data, CArray<float> expectedLeftVectors,
	CArray<float> expectedSingularValues, CArray<float> expectedRightVectors,
	bool returnLeftVectors, bool returnRightVectors, TSvd svdSolver )
{
	if( svdSolver == SVD_Sparse && ( returnLeftVectors == returnRightVectors ) ) {
		return;
	}
	const CSparseFloatMatrix& matrix = generateMatrix( samples, features, data );
	CArray<float> leftVectors;
	CArray<float> singularValues;
	CArray<float> rightVectors;
	SingularValueDecomposition( matrix.GetDesc(), svdSolver, leftVectors, singularValues, rightVectors,
		returnLeftVectors, returnRightVectors, components );
	int totalComponents = ( svdSolver == SVD_Sparse ) ? components : features;

	if( returnLeftVectors ) {
		for( int row = 0; row < samples; row++ ) {
			for( int col = 0; col < totalComponents; col++ ) {
				const float get = leftVectors[row * totalComponents + col];
				const float expected = expectedLeftVectors[row * features + col];
				ASSERT_NEAR( fabs( get ), fabs( expected ), 5e-3 );
			}
		}
	}
	expectedSingularValues.SetSize( totalComponents );
	checkArraysEqual( expectedSingularValues, singularValues.GetPtr() );
	if( returnRightVectors ) {
		for( int row = 0; row < totalComponents; row++ ) {
			for( int col = 0; col < features; col++ ) {
				const int index = row * features + col;
				ASSERT_NEAR( fabs( rightVectors[index] ), fabs( expectedRightVectors[index] ), 5e-3 );
			}
		}
	}
}

TEST( CSVDTest, SVDExampleTest )
{
	for( TSvd svdSolver : { SVD_Full, SVD_Sparse } ) {
		for( bool returnLeftVectors : { false, true } ) {
			for( bool returnRightVectors : { false, true } ) {
				svdTestExample( 4, 4, 2, { 2, 1, 3, 2, 2, 4, 4, 1, 2, 4, 1, 1, 4, 4, 3, 4 },
					{ -0.3495,  0.6025,  0.3083, -0.6479, -0.5253, -0.325,  0.7173,  0.3224,
					 -0.3879, -0.6554, -0.3305, -0.5575, -0.6719,  0.319, -0.5303,  0.4068 },
					{ 11.0117  ,  2.7114 ,  2.3155  ,  0.1736 },
					{ -0.4734, -0.6075, -0.5043, -0.3905, 0.19205657, -0.7533076 ,  0.29856473,  0.5536254,
					 -0.3158, -0.1149,  0.8087, -0.4828, -0.7995,  0.2241,  0.05091,  0.5549 },
					returnLeftVectors, returnRightVectors, svdSolver );
			}
		}
	}
}

static void pcaTestExample( int samples, int features, int components,
	CArray<float> data, CArray<float> expectedSingularValues,
	CArray<float> expectedVariance, CArray<float> expectedVarianceRatio,
	float expectedNoiseVariance, CArray<float> expectedComponents,
	CArray<float> expectedTransform, TSvd svdSolver )
{
	CPca::CParams params;
	params.ComponentsType = CPca::TComponents::PCAC_Int;
	params.Components = static_cast<float>( components );
	params.SvdSolver = svdSolver;

	const CSparseFloatMatrix& matrix = generateMatrix( samples, features, data );
	CPca pca( params );
	CFloatMatrixDesc transformed = pca.Transform( matrix.GetDesc() );
	ASSERT_EQ( samples, transformed.Height );
	ASSERT_EQ( components, transformed.Width );
	checkArraysEqual( expectedTransform, transformed.Values );

	if( svdSolver == SVD_Full ) {
		CFloatMatrixDesc componentsMatrix = pca.GetComponents();
		ASSERT_EQ( components, componentsMatrix.Height );
		ASSERT_EQ( features, componentsMatrix.Width );
		checkArraysEqual( expectedComponents, componentsMatrix.Values );
	}

	const CArray<float>& singular = pca.GetSingularValues();
	ASSERT_EQ( components, singular.Size() );
	checkArraysEqual( expectedSingularValues, singular.GetPtr() );
	
	const CArray<float>& variance = pca.GetExplainedVariance();
	ASSERT_EQ( components, variance.Size() );
	checkArraysEqual( expectedVariance, variance.GetPtr() );

	const CArray<float>& varianceRatio = pca.GetExplainedVarianceRatio();
	ASSERT_EQ( components, varianceRatio.Size() );
	checkArraysEqual( expectedVarianceRatio, varianceRatio.GetPtr() );

	ASSERT_NEAR( expectedNoiseVariance, pca.GetNoiseVariance(), 5e-3 );
}

TEST( CPCATest, PCAExamplesTest )
{
	for( TSvd svdSolver : { SVD_Full, SVD_Sparse } ) {
		pcaTestExample( 4, 4, 2, { 2, 1, 3, 2, 2, 4, 4, 1, 2, 4, 1, 1, 4, 4, 3, 4 },
			{ 3.0407, 2.6677 }, { 3.0819, 2.3722 }, { 0.451, 0.3472 }, 0.6896,
			{ 0.5649, 0.2846, 0.1827, 0.7525, -0.0238, -0.8853, 0.3850, 0.2592 },
			{ -0.8772, 2.1003, -0.5931, -0.4300, -1.1413, -1.5853, 2.6118, -0.0849 }, svdSolver );

		pcaTestExample( 3, 5, 2, { 5, 4, 7, 5, 3, 5, 7, 8, 6, 7, 6, 4, 4, 6, 2 },
			{ 5.1339, 1.9084 }, { 13.1789, 1.8210 }, { 0.8786, 0.1214 }, 0,
			{ -0.1196, 0.4516, 0.5095, 0.0309, 0.7218, -0.2818, -0.4135, 0.7076, -0.4196, -0.2695 },
			{ -0.8146, 1.5285, 3.9683, -0.5020, -3.1537, -1.0265 }, svdSolver );

		pcaTestExample( 5, 3, 3, { 4, 4, 8, 4, 5, 8, 5, 6, 8, 7, 5, 4, 2, 3, 3 },
			{ 5.2416, 3.8275, 1.0369 }, { 6.8686, 3.6624954, 0.2688 }, { 0.6359, 0.3391, 0.0248 }, 0,
			{ -0.2345, -0.3243, -0.9164, 0.8866, 0.3150, -0.3384, 0.3984, -0.8919, 0.2137 },
			{ -1.3611, -1.1528, 0.7605, -1.6854, -0.8378, -0.1314, -2.2443, 0.3639, -0.6249, 1.2766, 3.1759, 0.2090, 4.0142, -1.5491, -0.2131 },
			svdSolver );
	}
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
	params.SvdSolver = SVD_Full;
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
			ASSERT_NEAR( expectedComponent[i], abs( actualComponent.GetValue( i ) ), 5e-3 );
		}
	}
}