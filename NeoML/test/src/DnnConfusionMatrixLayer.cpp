/* Copyright © 2024 ABBYY

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
#include <DnnSimpleTest.h>

using namespace NeoML;
using namespace NeoMLTest;

TEST( ConfusionMatrixLayerTest, ConfusionMatrix )
{
	const int classCount = 7;
	const int initBatchSize = 5;
	const int newBatchSize = 3;

	CRandom random( 0x2345 );
	CDnn dnn( random, MathEngine() );

	CPtr<CSourceLayer> actual = Source( dnn, "actual" );
	CPtr<CSourceLayer> expected = Source( dnn, "expected" );
	CPtr<CConfusionMatrixLayer> quality = ConfusionMatrix()( "quality", { actual.Ptr(), expected.Ptr() } );
	CPtr<CSinkLayer> output = Sink( quality.Ptr(), "output" );

	CPtr<CDnnBlob> actualData = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, initBatchSize, classCount );
	CPtr<CDnnBlob> expectedData = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, initBatchSize, classCount );

	actual->SetBlob( actualData );
	expected->SetBlob( expectedData );

	CArray<float> expectedMatrix;
	expectedMatrix.SetSize( classCount * classCount );

	quality->SetReset( true );
	for( int i = 0; i < expectedMatrix.Size(); ++i ) {
		expectedMatrix[i] = 0;
	}

	for( int epoch = 0; epoch < 5; ++epoch ) {
		float* actualBuff = actualData->GetBuffer<float>( 0, actualData->GetDataSize(), false );
		float* expectedBuff = expectedData->GetBuffer<float>( 0, expectedData->GetDataSize(), false );
		for( int batch = 0; batch < actualData->GetObjectCount(); ++batch ) {
			const int actualClass = random.UniformInt( 0, classCount - 1 );
			const int expectedClass = random.UniformInt( 0, classCount - 1 );
			for( int i = 0; i < classCount; ++i ) {
				actualBuff[batch * classCount + i] = actualClass == i ? 1.f : 0.f;
				expectedBuff[batch * classCount + i] = expectedClass == i ? 1.f : 0.f;
			}
			expectedMatrix[expectedClass * classCount + actualClass] += 1;
		}
		expectedData->ReleaseBuffer( expectedBuff, true );
		actualData->ReleaseBuffer( actualBuff, true );

		dnn.RunOnce();
		quality->SetReset( false );

		CPtr<CDnnBlob> actualMatrixData = output->GetBlob();
		EXPECT_EQ( expectedMatrix.Size(), actualMatrixData->GetDataSize() );

		float* actualMatrix = actualMatrixData->GetBuffer<float>( 0, actualMatrixData->GetDataSize(), true );
		for( int i = 0; i < expectedMatrix.Size(); ++i ) {
			EXPECT_FLOAT_EQ( expectedMatrix[i], actualMatrix[i] );
		}
		actualMatrixData->ReleaseBuffer( actualMatrix, false );
	}

	// Checking reset...
	quality->SetReset( true );
	for( int i = 0; i < expectedMatrix.Size(); ++i ) {
		expectedMatrix[i] = 0;
	}
	for( int epoch = 0; epoch < 3; ++epoch ) {
		float* actualBuff = actualData->GetBuffer<float>( 0, actualData->GetDataSize(), false );
		float* expectedBuff = expectedData->GetBuffer<float>( 0, expectedData->GetDataSize(), false );
		for( int batch = 0; batch < actualData->GetObjectCount(); ++batch ) {
			const int actualClass = random.UniformInt( 0, classCount - 1 );
			const int expectedClass = random.UniformInt( 0, classCount - 1 );
			for( int i = 0; i < classCount; ++i ) {
				actualBuff[batch * classCount + i] = actualClass == i ? 1.f : 0.f;
				expectedBuff[batch * classCount + i] = expectedClass == i ? 1.f : 0.f;
			}
			expectedMatrix[expectedClass * classCount + actualClass] += 1;
		}
		expectedData->ReleaseBuffer( expectedBuff, true );
		actualData->ReleaseBuffer( actualBuff, true );

		dnn.RunOnce();
		quality->SetReset( false );

		CPtr<CDnnBlob> actualMatrixData = output->GetBlob();
		EXPECT_EQ( expectedMatrix.Size(), actualMatrixData->GetDataSize() );

		float* actualMatrix = actualMatrixData->GetBuffer<float>( 0, actualMatrixData->GetDataSize(), true );
		for( int i = 0; i < expectedMatrix.Size(); ++i ) {
			EXPECT_FLOAT_EQ( expectedMatrix[i], actualMatrix[i] );
		}
		actualMatrixData->ReleaseBuffer( actualMatrix, false );
	}

	actualData = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, newBatchSize, classCount );
	expectedData = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, newBatchSize, classCount );

	actual->SetBlob( actualData );
	expected->SetBlob( expectedData );

	// Checking batchSize change (it should not reset matrix!)
	for( int epoch = 0; epoch < 7; ++epoch ) {
		float* actualBuff = actualData->GetBuffer<float>( 0, actualData->GetDataSize(), false );
		float* expectedBuff = expectedData->GetBuffer<float>( 0, expectedData->GetDataSize(), false );
		for( int batch = 0; batch < actualData->GetObjectCount(); ++batch ) {
			const int actualClass = random.UniformInt( 0, classCount - 1 );
			const int expectedClass = random.UniformInt( 0, classCount - 1 );
			for( int i = 0; i < classCount; ++i ) {
				actualBuff[batch * classCount + i] = actualClass == i ? 1.f : 0.f;
				expectedBuff[batch * classCount + i] = expectedClass == i ? 1.f : 0.f;
			}
			expectedMatrix[expectedClass * classCount + actualClass] += 1;
		}
		expectedData->ReleaseBuffer( expectedBuff, true );
		actualData->ReleaseBuffer( actualBuff, true );

		dnn.RunOnce();
		quality->SetReset( false );

		CPtr<CDnnBlob> actualMatrixData = output->GetBlob();
		EXPECT_EQ( expectedMatrix.Size(), actualMatrixData->GetDataSize() );

		float* actualMatrix = actualMatrixData->GetBuffer<float>( 0, actualMatrixData->GetDataSize(), true );
		for( int i = 0; i < expectedMatrix.Size(); ++i ) {
			EXPECT_FLOAT_EQ( expectedMatrix[i], actualMatrix[i] );
		}
		actualMatrixData->ReleaseBuffer( actualMatrix, false );
	}
}
