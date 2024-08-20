/* Copyright © 2017-2024 ABBYY

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

static void maxOverTimeBackwardNaive( int sourceBatchLength, int batchWidth, int objectSize, int strideLength, int filterLength,
	const float* resultDiff, const int* maxIndices, float* sourceDiff )
{
	const int resultBatchLength = ( sourceBatchLength - filterLength ) / strideLength + 1;
	for( int l = 0; l < resultBatchLength; ++l ) {
		for( int w = 0; w < batchWidth; ++w ) {
			for( int i = 0; i < objectSize; ++i ) {
				const int ind = l * batchWidth * objectSize + w * objectSize + i;
				const int maxIndex = maxIndices[ind];
				const float diff = resultDiff[ind];
				sourceDiff[maxIndex * batchWidth * objectSize + w * objectSize + i] += diff;
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

	const int sourceBatchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int objectSize = random.UniformInt( objectSizeInterval.Begin, objectSizeInterval.End );
	const int strideLength = random.UniformInt( strideLengthInterval.Begin, strideLengthInterval.End );
	const int filterLength = random.UniformInt( filterLengthInterval.Begin, filterLengthInterval.End );

	const int sourceSize = objectSize * batchWidth * sourceBatchLength;

	CREATE_FILL_FLOAT_ARRAY( sourceData, valuesInterval.Begin, valuesInterval.End, sourceSize, random )
	CFloatBlob sourceBlob( MathEngine(), sourceBatchLength, batchWidth, 1, 1, 1, 1, objectSize );
	sourceBlob.CopyFrom( sourceData.data() );
	const int resultBatchLength = ( sourceBatchLength - filterLength ) / strideLength + 1;
	const int resultSize = objectSize * batchWidth * resultBatchLength;

	CFloatBlob resultBlob( MathEngine(), resultBatchLength, batchWidth, 1, 1, 1, 1, objectSize );
	CIntBlob indexBlob( MathEngine(), resultBatchLength, batchWidth, 1, 1, 1, 1, objectSize );
	CIntHandle indexBlobPtr = indexBlob.GetData();

	CREATE_FILL_FLOAT_ARRAY( resultDiffData, valuesInterval.Begin, valuesInterval.End, resultSize, random )
	CFloatBlob resultDiffBlob( MathEngine(), resultBatchLength, batchWidth, 1, 1, 1, 1, objectSize );
	resultDiffBlob.CopyFrom( resultDiffData.data() );
	CFloatBlob sourceDiffBlob( MathEngine(), sourceBatchLength, batchWidth, 1, 1, 1, 1, objectSize );

	CMaxOverTimePoolingDesc* desc = MathEngine().InitMaxOverTimePooling( sourceBlob.GetDesc(), filterLength, strideLength, resultBlob.GetDesc() );
	MathEngine().BlobMaxOverTimePooling( *desc, sourceBlob.GetData(), &indexBlobPtr, resultBlob.GetData() );
	MathEngine().BlobMaxOverTimePoolingBackward( *desc, resultDiffBlob.GetData(), indexBlobPtr, sourceDiffBlob.GetData() );
	delete desc;

	std::vector<int> maxIndices;
	maxIndices.resize( resultSize );
	indexBlob.CopyTo( maxIndices.data() );

	std::vector<float> actualDiff, expectedDiff;
	actualDiff.resize( sourceSize );
	sourceDiffBlob.CopyTo( actualDiff.data() );
	expectedDiff.insert( expectedDiff.begin(), sourceSize, 0 );

	maxOverTimeBackwardNaive( sourceBatchLength, batchWidth, objectSize, strideLength, filterLength,
		resultDiffData.data(), maxIndices.data(), expectedDiff.data() );

	for( int i = 0; i < sourceSize; i++ ) {
		EXPECT_TRUE( FloatEq( expectedDiff[i], actualDiff[i] ) );
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
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}

	RUN_TEST_IMPL( maxOverTimePoolingBackwardImpl )
}
