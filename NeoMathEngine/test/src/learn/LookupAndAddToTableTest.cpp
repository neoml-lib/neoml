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

static void lookupAndAddToTableTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int batchLength = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int batchWidth = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	const int vectorCount = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	const int objectSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	
	CREATE_FILL_FLOAT_ARRAY( outDiffBuffer, valuesInterval.Begin, valuesInterval.End, batchLength * batchWidth * vectorSize, random )
	CFloatBlob outDiff( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, vectorSize );
	outDiff.CopyFrom( outDiffBuffer.data() );

	std::vector<float> expected;
	expected.insert( expected.begin(), vectorCount * vectorSize, 0.f );

	std::vector<int> inputBuffer;
	inputBuffer.resize( objectSize * batchLength * batchWidth );
	for( int i = 0; i < batchLength * batchWidth; ++i ) {
		for( int j = 0; j < objectSize; ++j ) {
			const int index = random.UniformInt( -1, vectorCount - 1 );
			inputBuffer[i * objectSize + j] = index;
			if( index >= 0 ) {
				for( int k = 0; k < vectorSize; ++k ) {
					expected[index * vectorSize + k] += outDiffBuffer[i * vectorSize + k];
				}
			}
		}
	}
	CIntBlob input( MathEngine(), batchLength, batchWidth, 1, 1, 1, 1, objectSize );
	input.CopyFrom( inputBuffer.data() );

	CFloatBlob paramDiffBlob( MathEngine(), vectorCount, vectorSize, 1, 1, 1, 1, 1 );
	MathEngine().LookupAndAddToTable( input.GetData(), batchLength * batchWidth, objectSize,
		outDiff.GetData(), vectorSize, paramDiffBlob.GetData(), vectorCount );
	std::vector<float> paramDiff;
	paramDiff.resize( paramDiffBlob.GetDataSize() );
	paramDiffBlob.CopyTo( paramDiff.data() );
	ASSERT_EQ( expected.size(), paramDiff.size() );
	for( size_t i = 0; i < paramDiff.size(); ++i ) {
		ASSERT_NEAR( expected[i], paramDiff[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CLookupAndAddToTableTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CLookupAndAddToTableTestInstantiation, CLookupAndAddToTableTest,
	::testing::Values(
		CTestParams(
			"Height = (1..50);"
			"Width = (1..50);"
			"BatchSize = (1..5);"
			"VectorSize = (1..20);"
			"Values = (-1..1);"
			"Channels = (1..5);"
			"TestCount = 100;"
		),
		CTestParams(
			"Height = (100..500);"
			"Width = (100..500);"
			"BatchSize = (1..5);"
			"VectorSize = (30..50);"
			"Values = (-1..1);"
			"Channels = (1..5);"
			"TestCount = 5;"
		)
	)
);

TEST_P( CLookupAndAddToTableTest, Random )
{
	RUN_TEST_IMPL( lookupAndAddToTableTestImpl )
}
