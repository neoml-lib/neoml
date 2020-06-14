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

static void globalMaxOverTimePoolingNaive( const float *source, int batchLength, int batchWidth, int objectSize, float *result, int *maxIndices ) 
{
	
	for( int j = 0; j < objectSize * batchWidth; j++ ) {
		float maxVal = -FLT_MAX;
		int maxInd = 0;
		for( int i = 0; i < batchLength; i++ ) {
			if( source[i * objectSize * batchWidth + j] > maxVal ) {
				maxVal = source[i * objectSize * batchWidth + j];
				maxInd = i;
			}
		}
		result[j] = maxVal;
		if( maxIndices != nullptr ) {
			maxIndices[j] = maxInd;
		}
	}
}

static void globalMaxOverTimePoolingTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchLengthInterval = params.GetInterval( "BatchLength" );
	const CInterval batchWidthInterval = params.GetInterval( "BatchWidth" );
	const CInterval objectSizeInterval = params.GetInterval( "ObjectSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	
	const int batchLength = random.UniformInt( batchLengthInterval.Begin, batchLengthInterval.End );
	const int batchWidth = random.UniformInt( batchWidthInterval.Begin, batchWidthInterval.End );
	const int objectSize = random.UniformInt( objectSizeInterval.Begin, objectSizeInterval.End );

	CREATE_FILL_FLOAT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, objectSize * batchWidth * batchLength, random )
	CFloatBlob inputBlob( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, objectSize );
	inputBlob.CopyFrom( inputData.data() );

	std::vector<float> expected;
	expected.resize( objectSize * batchWidth );

	globalMaxOverTimePoolingNaive( inputData.data(), batchLength, batchWidth, objectSize, expected.data(), nullptr);

	CFloatBlob resultBlob( MathEngine(), 1, batchWidth, 1, 1, 1, 1, objectSize );

	CGlobalMaxOverTimePoolingDesc *desc = MathEngine().InitGlobalMaxOverTimePooling( inputBlob.GetDesc(), resultBlob.GetDesc() );
	MathEngine().BlobGlobalMaxOverTimePooling( *desc, inputBlob.GetData(), nullptr, resultBlob.GetData() );
	delete desc;

	std::vector<float> resultData;
	resultData.resize( objectSize * batchWidth );
	resultBlob.CopyTo( resultData.data() );

	for( size_t i = 0; i < resultData.size(); i++ ) {
		ASSERT_NEAR( expected[i], resultData[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineGlobalMaxOverTimePoolingTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineGlobalMaxOverTimePoolingTestInstantiation, CMathEngineGlobalMaxOverTimePoolingTest,
	::testing::Values(
		CTestParams(
			"BatchLength = (10..30);"
			"BatchWidth = (5..50);"
			"ObjectSize = (1..100);"
			"Values = (-50..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineGlobalMaxOverTimePoolingTest, Random)
{
	RUN_TEST_IMPL( globalMaxOverTimePoolingTestImpl )
}
