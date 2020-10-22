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

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

static void maxOverTimeBackwardNaive( int inputBatchLength, int batchWidth, int objectSize, int strideLength, int filterLength,
	const float *outputDiff, const int *maxIndices, float *inputDiff )
{
	const int resultBatchLength = ( inputBatchLength - filterLength ) / strideLength + 1;
	for( int l = 0; l < resultBatchLength; ++l ) {
		for( int w = 0; w < batchWidth; ++w ) {
			for( int i = 0; i < objectSize; ++i ) {
				const int ind = l * batchWidth * objectSize + w * objectSize + i;
				const int maxIndex = maxIndices[ind];
				const float diff = outputDiff[ind];
				inputDiff[maxIndex * batchWidth * objectSize + w * objectSize + i] += diff;
			}
		}
	}
}

static void maxOverTimePoolingBackwardImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval objectSizeInterval = params.GetInterval( "ObjectSize" );
	const CInterval strideLengthInterval = params.GetInterval( "StrideLength" );
	const CInterval filterLengthInterval = params.GetInterval( "FilterLength" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int inputBatchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int objectSize = random.UniformInt( objectSizeInterval.Begin, objectSizeInterval.End );
	const int strideLength = random.UniformInt( strideLengthInterval.Begin, strideLengthInterval.End );
	const int filterLength = random.UniformInt( filterLengthInterval.Begin, filterLengthInterval.End );

	const int inputSize = objectSize * batchWidth * inputBatchLength;

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, inputSize, random )
	CFloatBlob inputBlob( MathEngine(), inputBatchLength, batchWidth, 1, 1, 1, 1, objectSize );
	inputBlob.CopyFrom( inputData.data() );
	const int resultBatchLength = ( inputBatchLength - filterLength ) / strideLength + 1;
	const int outputSize = objectSize * batchWidth * resultBatchLength;

	CFloatBlob outputBlob( MathEngine(), resultBatchLength, batchWidth, 1, 1, 1, 1, objectSize );
	CIntBlob indexBlob( MathEngine(), resultBatchLength, batchWidth, 1, 1, 1, 1, objectSize );
	CIntHandle indexBlobPtr = indexBlob.GetData();

	CREATE_FILL_FLOAT_ARRAY( outputDiffData, valuesInterval.Begin, valuesInterval.End, outputSize, random )
	CFloatBlob outputDiffBlob( MathEngine(), resultBatchLength, batchWidth, 1, 1, 1, 1, objectSize );
	outputDiffBlob.CopyFrom( outputDiffData.data() );
	CFloatBlob inputDiffBlob( MathEngine(), inputBatchLength, batchWidth, 1, 1, 1, 1, objectSize );

	CMaxOverTimePoolingDesc *desc = MathEngine().InitMaxOverTimePooling( inputBlob.GetDesc(), filterLength, strideLength, outputBlob.GetDesc() );
	MathEngine().BlobMaxOverTimePooling( *desc, inputBlob.GetData(), &indexBlobPtr, outputBlob.GetData() );
	MathEngine().BlobMaxOverTimePoolingBackward( *desc, outputDiffBlob.GetData(), indexBlobPtr, inputDiffBlob.GetData() );
	delete desc;

	std::vector<int> maxIndices;
	maxIndices.resize( outputSize );
	indexBlob.CopyTo( maxIndices.data() );

	std::vector<float> actualDiff, expectedDiff;
	actualDiff.resize( inputSize );
	inputDiffBlob.CopyTo( actualDiff.data() );
	expectedDiff.insert( expectedDiff.begin(), inputSize, 0 );

	maxOverTimeBackwardNaive( inputBatchLength, batchWidth, objectSize, strideLength, filterLength,
		outputDiffData.data(), maxIndices.data(), expectedDiff.data() );

	for( int i = 0; i < inputSize; i++ ) {
		ASSERT_TRUE( FloatEq( expectedDiff[i], actualDiff[i] ) );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMathEngineBlobMaxOverTimePoolingBackwardTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineBlobMaxOverTimePoolingBackwardTestInstantiation, CMathEngineBlobMaxOverTimePoolingBackwardTest,
	::testing::Values(
		CTestParams(
			"BatchLength = (10..100);"
			"BatchWidth = (1..5);"
			"ObjectSize = (1..100);"
			"StrideLength = (1..10);"
			"FilterLength = (1..10);"
			"Values = (-50..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CMathEngineBlobMaxOverTimePoolingBackwardTest, Random )
{
	RUN_TEST_IMPL( maxOverTimePoolingBackwardImpl )
}
