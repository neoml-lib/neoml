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
#include <MeTestCommon.h>

using namespace NeoML;
using namespace NeoMLTest;

static void addHeightIndexImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const auto size = static_cast<size_t>( height ) * width * channels * batchSize;

	CREATE_FILL_INT_ARRAY( inputData, valuesInterval.Begin, valuesInterval.End, size, random )
	CIntBlob inputBlob( MathEngine(), batchSize, height, width, channels );
	inputBlob.CopyFrom( inputData.data() );
	CIntBlob outputBlob( MathEngine(), batchSize, height, width, channels );
	std::vector<int> outputData, getData;
	outputData.resize( size );
	getData.resize( size );

	MathEngine().AddHeightIndex( inputBlob.GetDesc(), inputBlob.GetData(), /*isForward*/true, outputBlob.GetData() );

	addIndexNaive( inputData.data(), batchSize, height, width, channels, outputData.data(), /*isHeight*/true );
	outputBlob.CopyTo( getData.data() );

	for( size_t i = 0; i < size; ++i ) {
		ASSERT_EQ( outputData[i], getData[i] );
	}

	MathEngine().AddHeightIndex( inputBlob.GetDesc(), outputBlob.GetData(), /*isForward*/false, inputBlob.GetData() );

	inputBlob.CopyTo( getData.data() );
	for( size_t i = 0; i < size; ++i ) {
		ASSERT_EQ( inputData[i], getData[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CAddHeightIndexTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CAddHeightIndexTestInstantiation, CAddHeightIndexTest,
	::testing::Values(
		CTestParams(
			"Height = (1..20);"
			"Width = (1..20);"
			"Channels = (1..5);"
			"BatchSize = (1..5);"
			"Values = (-100..100);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CAddHeightIndexTest, Random )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		GTEST_LOG_(INFO) << "Skipped rest of test for MathEngine type=" << int(met) << " because no implementation.\n";
		return;
	}

	RUN_TEST_IMPL( addHeightIndexImpl )
}
