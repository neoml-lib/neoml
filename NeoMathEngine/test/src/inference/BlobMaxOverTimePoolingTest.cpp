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

static void maxOverTimeNaive( const float *source, int inputBatchLength, int batchWidth, int objectSize, int strideLength, int filterLength, float *result, int *maxIndices ) 
{
	const int resultBatchLength = ( inputBatchLength - filterLength ) / strideLength + 1;

	int seqElemSize = objectSize * batchWidth;

	const float* sourceStart = source;
	float* resultStart = result;
	int* indexStart = maxIndices;

	int indexValueStart = 0;
	for(int l = 0; l < resultBatchLength; ++l) {
		const float* sourceData = sourceStart;

		for( int i = 0; i < seqElemSize; i++ ) {
			resultStart[i] = sourceData[i];
			if( indexStart != NULL ) {
				indexStart[i] = indexValueStart;
			}
		}
		sourceData += seqElemSize;

		for(int n = 1; n < filterLength; ++n) {
			float* resultData = resultStart;
			int* indexData = indexStart;
			int indexValue = indexValueStart + n;

			for(int i = 0; i < seqElemSize; ++i) {
				if(*sourceData > *resultData) {
					*resultData = *sourceData;
					if( indexData != NULL ) {
						*indexData = indexValue;
					}
				}
				++sourceData;
				++resultData;
				if( indexData != NULL ) {
					++indexData;
				}
			}
		}
		sourceStart += strideLength * seqElemSize;
		resultStart += seqElemSize;
		if( indexStart != NULL ) {
			indexStart += seqElemSize;
		}

		indexValueStart += strideLength;
	}
}

static void maxOverTimePoolingImpl( const CTestParams& params, int seed )
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

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, objectSize * batchWidth * inputBatchLength, random )
	CFloatBlob inputBlob( MathEngine(), inputBatchLength, batchWidth, 1, 1, 1, 1, objectSize );
	inputBlob.CopyFrom( inputData.data() );
	const int resultBatchLength = ( inputBatchLength - filterLength ) / strideLength + 1;
	
	std::vector<float> expected;
	expected.resize( objectSize * batchWidth * resultBatchLength );
	maxOverTimeNaive( inputData.data(), inputBatchLength, batchWidth, objectSize, strideLength, filterLength, expected.data(), nullptr);

	CFloatBlob resultBlob( MathEngine(), resultBatchLength, batchWidth, 1, 1, 1, 1, objectSize );

	CMaxOverTimePoolingDesc *desc = MathEngine().InitMaxOverTimePooling( inputBlob.GetDesc(), filterLength, strideLength, resultBlob.GetDesc() );
	MathEngine().BlobMaxOverTimePooling( *desc, inputBlob.GetData(), nullptr, resultBlob.GetData() );
	delete desc;

	std::vector<float> resultData;
	resultData.resize( objectSize * batchWidth * resultBatchLength );
	resultBlob.CopyTo( resultData.data() );

	for( size_t i = 0; i < resultData.size(); i++ ) {
		ASSERT_NEAR( expected[i], resultData[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineMaxOverTimePoolingTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineMaxOverTimePoolingTestInstantiation, CMathEngineMaxOverTimePoolingTest,
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

TEST_P(CMathEngineMaxOverTimePoolingTest, Random)
{
	RUN_TEST_IMPL( maxOverTimePoolingImpl )
}
