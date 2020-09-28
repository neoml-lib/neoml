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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/AccuracyLayer.h>
#include <float.h>

namespace NeoML {

CAccuracyLayer::CAccuracyLayer( IMathEngine& mathEngine ) :
	CQualityControlLayer( mathEngine, "CCnnAccuracyLayer" ),
	iterationsCount( 0 ),
	collectedAccuracy( 0. )
{
}

static const int AccuracyLayerVersion = 2000;

void CAccuracyLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( AccuracyLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CQualityControlLayer::Serialize( archive );
}

void CAccuracyLayer::OnReset()
{
	iterationsCount = 0;
	collectedAccuracy = 0;
}

void CAccuracyLayer::Reshape()
{
	CQualityControlLayer::Reshape();
	NeoAssert( inputDescs[0].Height() == 1
		&& inputDescs[0].Width() == 1
		&& inputDescs[0].Depth() == 1 );

	outputDescs[0] = CBlobDesc( CT_Float );

	iterationsCount = 0;
	collectedAccuracy = 0;
}

void CAccuracyLayer::RunOnceAfterReset()
{
	CPtr<CDnnBlob> inputBlob = inputBlobs[0];
	CPtr<CDnnBlob> expectedLabelsBlob = inputBlobs[1];
	CFastArray<float, 1> inputBuffer;
	CFastArray<float, 1> expectedBuffer;

	const int dataSize = inputBlob->GetDataSize();
	const int objectCount = inputBlob->GetObjectCount();
	const int objectSize = inputBlob->GetObjectSize();

	inputBuffer.SetSize( dataSize );
	expectedBuffer.SetSize( dataSize );
	inputBlob->CopyTo( inputBuffer.GetPtr(), dataSize );
	expectedLabelsBlob->CopyTo( expectedBuffer.GetPtr(), dataSize );
	int correctlyClassifiedCount = 0;
	for( int i = 0; i < inputBlob->GetBatchWidth(); i++ ) {
		for( int j = 0; j < inputBlob->GetBatchLength(); j++ ) {
			const int sampleId = inputBlob->GetBatchWidth() * j + i;
			if( objectSize >= 2 ) {
				int expectedClass = 0;
				float maxValue = -FLT_MAX;
				for( int classWeightId = 0; classWeightId < objectSize; classWeightId++ ) {
					float currentValue = inputBuffer[sampleId * objectSize + classWeightId];
					if( maxValue < currentValue ) {
						maxValue = currentValue;
						expectedClass = classWeightId;
					}
				}
				if( expectedBuffer[sampleId * objectSize + expectedClass] > 0.f ) {
					correctlyClassifiedCount += 1;
				}
			} else {
				NeoAssert( objectSize == 1 );
				// The input blob has one channel
				// That means a positive value corresponds to one class and a negative to the other
				// The input blob with the correct labels should only contain +1 and -1 values
				const float predictedValue = inputBuffer[sampleId];
				const float expectedClass = expectedBuffer[sampleId];
				if( ( predictedValue >= 0 && expectedClass > 0 ) || ( predictedValue < 0 && expectedClass < 0 ) ) {
					correctlyClassifiedCount += 1;
				}
			}
		}
	}
	collectedAccuracy += static_cast<double>( correctlyClassifiedCount ) / objectCount;
	outputBlobs[0]->GetData().SetValue( static_cast<float>( collectedAccuracy ) / ++iterationsCount );
}

//---------------------------------------------------------------------------------------------------------------------

CConfusionMatrixLayer::CConfusionMatrixLayer( IMathEngine& mathEngine ) :
	CQualityControlLayer( mathEngine, "CCnnConfusionMatrixLayer" )
{
}

static const int ConfusionMatrixLayerVersion = 2000;

void CConfusionMatrixLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ConfusionMatrixLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CQualityControlLayer::Serialize( archive );
}

void CConfusionMatrixLayer::Reshape()
{
	CheckInputs();
	NeoAssert( inputDescs.Size() == 2 );
	// For classifying a sigmoid a special implementation is needed
	NeoAssert( inputDescs[0].Channels() >= 2 );
	NeoAssert( inputDescs[0].Height() == 1 );
	NeoAssert( inputDescs[0].Width() == 1 );
	NeoAssert( inputDescs[0].ObjectCount() == inputDescs[1].ObjectCount() );
	NeoAssert( inputDescs[0].ObjectSize() >= 1 );
	NeoAssert( inputDescs[0].ObjectSize() == inputDescs[1].ObjectSize() );

	const int classCount = inputDescs[0].Channels();
	if( confusionMatrix.SizeX() != classCount ) {
		confusionMatrix.Reset();
		confusionMatrix.SetSize( classCount, classCount );
		confusionMatrix.Set( 0.f );
	}

	outputDescs[0] = CBlobDesc( CT_Float );
	outputDescs[0].SetDimSize( BD_Height, classCount );
	outputDescs[0].SetDimSize( BD_Width, classCount );
}

void CConfusionMatrixLayer::RunOnceAfterReset()
{
	CPtr<CDnnBlob> inputBlob = inputBlobs[0];
	CPtr<CDnnBlob> expectedLabelsBlob = inputBlobs[1];
	CFastArray<float, 1> predictedClassBuffer;
	CFastArray<float, 1> expectedClassBuffer;

	int dataSize = inputBlob->GetDataSize();
	int objectCount = inputBlob->GetObjectCount();
	int objectSize = inputBlob->GetObjectSize();

	predictedClassBuffer.SetSize( dataSize );
	expectedClassBuffer.SetSize( dataSize );

	inputBlob->CopyTo( predictedClassBuffer.GetPtr(), dataSize );
	expectedLabelsBlob->CopyTo( expectedClassBuffer.GetPtr(), dataSize );

	for( int sampleId = 0; sampleId < objectCount; sampleId++ ) {
		// Class labels
		int expectedClass = NotFound;
		int predictedClass = NotFound;
		// Maximums
		float maxValueForPredictedClass = -FLT_MAX;
		float maxValueForExpectedClass = -FLT_MAX;

		for( int classWeightId = 0; classWeightId < objectSize; classWeightId++ ) {
			const float currentPredictedClassWeight = predictedClassBuffer[sampleId * objectSize + classWeightId];
			const float currentExpectedClassWeight = expectedClassBuffer[sampleId * objectSize + classWeightId];
			if( maxValueForPredictedClass < currentPredictedClassWeight ) {
				maxValueForPredictedClass = currentPredictedClassWeight;
				predictedClass = classWeightId;
			}
			if( maxValueForExpectedClass < currentExpectedClassWeight ) {
				maxValueForExpectedClass = currentExpectedClassWeight;
				expectedClass = classWeightId;
			}
		}
		if( maxValueForExpectedClass < 0.f ) {
			continue;
		}
		NeoAssert( expectedClass != NotFound && predictedClass != NotFound );
		// Add a new entry
		confusionMatrix( expectedClass, predictedClass ) += 1;
	}
	// Even though we know the CVariableMatrix stores data in columns, best make another copy
	CFastArray<float, 1> outputData;
	outputData.SetSize( confusionMatrix.SizeX() * confusionMatrix.SizeY() );
	NeoAssert( outputBlobs[0]->GetDataSize() == outputData.Size() );
	// Write data in rows
	float* ptr = outputData.GetPtr();
	for( int i = 0; i < confusionMatrix.SizeY(); i++ ) {
		for( int j = 0; j < confusionMatrix.SizeX(); j++, ptr++ ) {
			*ptr = confusionMatrix( i, j );
		}
	}
	// Copy into the output
	outputBlobs[0]->CopyFrom( outputData.GetPtr() );
}

} // namespace NeoML
