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

#include <type_traits>
#include <numeric>

using namespace NeoML;
using namespace NeoMLTest;

template <typename TIndex, typename TLookup>
static void multichannelLookupAndCopyImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval vectorCountInterval = params.GetInterval( "VectorCount" );
	const CInterval lookupCountInterval = params.GetInterval( "LookupCount" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int lookupCount = random.UniformInt( lookupCountInterval.Begin, lookupCountInterval.End );

	std::vector<CLookupDimension> lookupDimensions;
	lookupDimensions.resize( lookupCount );
	std::vector<std::vector<TLookup>> lookupData;
	lookupData.resize( lookupCount );
	std::vector<CMemoryHandleVar<TLookup>*> lookupHandleVars;
	std::vector<CTypedMemoryHandle<const TLookup>> lookupHandles;
	for( int i = 0; i < lookupCount; i++ ) {
		lookupDimensions[i].VectorCount = random.UniformInt( vectorCountInterval.Begin, vectorCountInterval.End );
		lookupDimensions[i].VectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
		lookupData[i].resize( lookupDimensions[i].VectorCount * lookupDimensions[i].VectorSize );
		for( size_t j = 0; j < lookupData[i].size(); j++ ) {
			lookupData[i][j] = static_cast<TLookup>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) );
		}
		lookupHandleVars.push_back( new CMemoryHandleVar<TLookup>( MathEngine(), lookupData[i].size() ) );
		MathEngine().DataExchangeTyped( lookupHandleVars[i]->GetHandle(), lookupData[i].data(), lookupData[i].size() );
		lookupHandles.push_back( CTypedMemoryHandle<const TLookup>( lookupHandleVars[i]->GetHandle() ) );
	}

	// If input data and lookup data are of the same type then number of channels may be larger than number of lookup tables
	// (these additional values will be forwarded as is)
	const int channelCount = std::is_same<TIndex, TLookup>::value ? lookupCount + random.UniformInt( 0, 3 ) : lookupCount;

	std::vector<TIndex> inputData;
	inputData.resize( batchSize * channelCount );
	for( int i = 0; i < batchSize; i++ ) {
		for( int j = 0; j < channelCount; j++ ) {
			if( j < lookupCount ) {
				inputData[i * channelCount + j] = static_cast<TIndex>( random.Uniform( 0, lookupDimensions[j].VectorCount - 1 ) );
			} else {
				inputData[i * channelCount + j] = static_cast<TIndex>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) );
			}
		}
	}

	const int resultChannelCount = std::accumulate( lookupDimensions.begin(), lookupDimensions.end(), channelCount - lookupCount,
		[] ( const int& sum, const CLookupDimension& dim ) { return sum + dim.VectorSize; } );

	std::vector<TLookup> expected;
	expected.resize( batchSize * resultChannelCount );
	int inputDataIndex = 0;
	int expectedIndex = 0;

	for( int i = 0; i < batchSize; ++i ) {
		for( int j = 0; j < lookupCount; ++j ) {
			int index = (int)inputData[inputDataIndex++];
			ASSERT_TRUE(0 <= index && index < lookupDimensions[j].VectorCount);
			int vectorSize = lookupDimensions[j].VectorSize;
			for( int c = 0; c < vectorSize; c++ ) {
				expected[expectedIndex + c] = lookupData[j][index * vectorSize + c];
			}
			expectedIndex += vectorSize;
		}
		int remained = channelCount - lookupCount;
		if(remained > 0) {
			for( int c = 0; c < remained; c++ ) {
				expected[expectedIndex + c] = static_cast<TLookup>( inputData[inputDataIndex + c] );
			}
			expectedIndex += remained;
			inputDataIndex += remained;
		}
	}

	CMemoryHandleStackVar<TIndex> inputHandle( MathEngine(), inputData.size() );
	MathEngine().DataExchangeTyped( inputHandle.GetHandle(), inputData.data(), inputData.size() );

	std::vector<TLookup> result;
	result.resize( batchSize * resultChannelCount);
	MathEngine().VectorMultichannelLookupAndCopy( batchSize, channelCount, inputHandle.GetHandle(), 
		lookupHandles.data(), lookupDimensions.data(), lookupCount, CARRAY_WRAPPER( TLookup, result ), resultChannelCount);

	for( size_t i = 0; i < lookupHandleVars.size(); i++ ) {
		delete lookupHandleVars[i];
	}

	for( size_t i = 0; i < result.size(); i++ ) {
		ASSERT_NEAR( static_cast<float>( expected[i] ), static_cast<float>( result[i] ), 1e-6 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineMultichannelLookupAndCopyTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineMultichannelLookupAndCopyTestInstantiation, CMathEngineMultichannelLookupAndCopyTest,
	::testing::Values(
		CTestParams(
			"BatchSize = (1..5);"
			"LookupCount = (10..100);"
			"VectorCount = (10..15);"
			"VectorSize = (20..20);"

			"Values = (-50..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineMultichannelLookupAndCopyTest, Random)
{
	RUN_TEST_IMPL((multichannelLookupAndCopyImpl<float, float>))
	RUN_TEST_IMPL((multichannelLookupAndCopyImpl<int, float>))
	RUN_TEST_IMPL((multichannelLookupAndCopyImpl<int, int>))
}
