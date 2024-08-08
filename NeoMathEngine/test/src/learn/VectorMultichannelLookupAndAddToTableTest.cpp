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

template <typename T>
static void multichannelLookupAndAddToTableImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval vectorCountInterval = params.GetInterval( "VectorCount" );
	const CInterval lookupCountInterval = params.GetInterval( "LookupCount" );
	const CInterval channelCountInterval = params.GetInterval( "ChannelCount" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int lookupCount = random.UniformInt( lookupCountInterval.Begin, lookupCountInterval.End );
	const int channelCount = random.UniformInt( lookupCount, channelCountInterval.End );
	float mult = static_cast<float>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) );

	std::vector<CLookupDimension> lookupDimensions;
	lookupDimensions.resize( lookupCount );
	std::vector<std::vector<float>> lookupData;
	lookupData.resize( lookupCount );
	std::vector<CFloatHandleVar*> lookupHandleVars;
	std::vector<CFloatHandle> lookupHandles;
	for( int i = 0; i < lookupCount; i++ ) {
		lookupDimensions[i].VectorCount = random.UniformInt( vectorCountInterval.Begin, vectorCountInterval.End );
		lookupDimensions[i].VectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
		lookupData[i].resize( lookupDimensions[i].VectorCount * lookupDimensions[i].VectorSize );
		for( size_t j = 0; j < lookupData[i].size(); j++ ) {
			lookupData[i][j] = float( j ); (void)static_cast<float>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) );
		}
		lookupHandleVars.push_back( new CFloatHandleVar( MathEngine(), lookupData[i].size() ) );
		MathEngine().DataExchangeTyped( lookupHandleVars[i]->GetHandle(), lookupData[i].data(), lookupData[i].size() );
		lookupHandles.push_back( CFloatHandle( lookupHandleVars[i]->GetHandle() ) );
	}

	std::vector<T> inputData;
	inputData.resize( batchSize * channelCount );
	for( int i = 0; i < batchSize; i++ ) {
		for( int j = 0; j < channelCount; j++ ) {
			if( j < lookupCount ) {
				inputData[i * channelCount + j] = static_cast<T>( random.Uniform( 0, lookupDimensions[j].VectorCount - 1 ) );
			}
			else {
				inputData[i * channelCount + j] = static_cast<T>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) );
			}
		}
	}

	int remained = channelCount - lookupCount;
	int resultChannelCount = remained;
	for( int i = 0; i < lookupCount; i++ ) {
		resultChannelCount += lookupDimensions[i].VectorSize;
	}

	CMemoryHandleStackVar<T> inputHandle( MathEngine(), inputData.size() );
	MathEngine().DataExchangeTyped( inputHandle.GetHandle(), inputData.data(), inputData.size() );
	CREATE_FILL_FLOAT_ARRAY( matrix, valuesInterval.Begin, valuesInterval.End, batchSize * resultChannelCount, random )

	MathEngine().VectorMultichannelLookupAndAddToTable( batchSize, channelCount, inputHandle.GetHandle(),
		lookupHandles.data(), lookupDimensions.data(), lookupCount, FLOAT_WRAPPER( &mult ), CARRAY_FLOAT_WRAPPER( matrix ), resultChannelCount );

	int inputDataIndex = 0;
	int expectedIndex = 0;

	for( int i = 0; i < batchSize; ++i ) {
		for( int j = 0; j < lookupCount; ++j ) {
			int index = ( int )inputData[inputDataIndex++];
			EXPECT_TRUE( 0 <= index && index < lookupDimensions[j].VectorCount );
			int vectorSize = lookupDimensions[j].VectorSize;
			for( int c = 0; c < vectorSize; c++ ) {
				lookupData[j][index * vectorSize + c] += mult * matrix[expectedIndex + c];
			}
			expectedIndex += vectorSize;
		}
		if( remained > 0 ) {
			expectedIndex += remained;
			inputDataIndex += remained;
		}
	}

	for( int i = 0; i < lookupCount; i++ ) {
		std::vector<float> getData;
		getData.resize( lookupData[i].size() );
		MathEngine().DataExchangeTyped<float>( getData.data(), lookupHandles[i], lookupData[i].size() );
		for( size_t j = 0; j < lookupData[i].size(); j++ ) {
			EXPECT_NEAR( getData[j], lookupData[i][j], 1e-3 );
		}
	}

	for( size_t i = 0; i < lookupHandleVars.size(); i++ ) {
		delete lookupHandleVars[i];
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineMultichannelLookupAndAddToTableTest : public CTestFixtureWithParams {
};

TEST_P( CMathEngineMultichannelLookupAndAddToTableTest, Random )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}
	RUN_TEST_IMPL( multichannelLookupAndAddToTableImpl<float> )
	RUN_TEST_IMPL( multichannelLookupAndAddToTableImpl<int> )
}

INSTANTIATE_TEST_CASE_P( CMathEngineMultichannelLookupAndAddToTableTestInstantiation, CMathEngineMultichannelLookupAndAddToTableTest,
	::testing::Values(
		CTestParams(
			"BatchSize = (1..5);"
			"LookupCount = (10..50);"
			"ChannelCount = (10..50);"
			"VectorCount = (10..20);"
			"VectorSize = (10..20);"
			"Values = (-50..50);"
			"TestCount = 100;"
		)
	)
);
