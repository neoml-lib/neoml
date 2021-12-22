/* Copyright © 2017-2021 ABBYY Production LLC

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

static void bertConvNaive( const std::vector<float>& data, const std::vector<float>& kernel,
	int seqLen, int batchSize, int numHeads, int headSize, int kernelSize, std::vector<float>& output )
{
	assert( static_cast<int>( data.size() ) == seqLen * batchSize * numHeads * headSize );
	assert( static_cast<int>( kernel.size() ) == seqLen * batchSize * numHeads * kernelSize );

	output.resize( seqLen * batchSize * numHeads * headSize );

	const int pad = ( kernelSize - 1 ) / 2;
	const int dataSeqStep = batchSize * numHeads * headSize;

	int outOffset = 0;
	int kernelOffset = 0;
	for( int seq = 0; seq < seqLen; ++seq ) {
		for( int b = 0; b < batchSize * numHeads; ++b ) {
			for( int h = 0; h < headSize; ++h ) {
				output[outOffset] = 0.f;
				int dataOffset = h + b * headSize + ( seq - pad ) * dataSeqStep;
				for( int k = 0; k < kernelSize; ++k ) {
					if( dataOffset >= 0 && dataOffset < static_cast<int>( data.size() ) ) {
						output[outOffset] += data[dataOffset] * kernel[kernelOffset + k];
					}
					dataOffset += dataSeqStep;
				}
				outOffset++;
			}
			kernelOffset += kernelSize;
		}
	}
}

static void bertConvTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval seqLenInterval = params.GetInterval( "SeqLen" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval numHeadInterval = params.GetInterval( "NumHeads" );
	const CInterval headSizeInterval = params.GetInterval( "HeadSize" );
	const CInterval kernelSizeInterval = params.GetInterval( "KernelSize" );

	const int seqLen = random.UniformInt( seqLenInterval.Begin, seqLenInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int numHeads = random.UniformInt( numHeadInterval.Begin, numHeadInterval.End );
	const int headSize = random.UniformInt( headSizeInterval.Begin, headSizeInterval.End );
	const int kernelSize = random.UniformInt( kernelSizeInterval.Begin, kernelSizeInterval.End );

	CFloatBlob dataBlob( MathEngine(), seqLen, batchSize, 1, 1, 1, 1, numHeads * headSize );
	CREATE_FILL_FLOAT_ARRAY( dataArr, -1.f, 1.f, dataBlob.GetDataSize(), random )
	dataBlob.CopyFrom( dataArr.data() );

	CFloatBlob kernelBlob( MathEngine(), seqLen, batchSize * numHeads, 1, kernelSize, 1, 1, 1 );
	CREATE_FILL_FLOAT_ARRAY( kernelArr, -1.f, 1.f, kernelBlob.GetDataSize(), random )
	kernelBlob.CopyFrom( kernelArr.data() );

	CFloatBlob outputBlob( MathEngine(), seqLen, batchSize * numHeads, 1, headSize, 1, 1, 1 );
	MathEngine().BertConv( dataBlob.GetData(), kernelBlob.GetData(), seqLen, batchSize, numHeads, headSize, kernelSize,
		outputBlob.GetData() );
	std::vector<float> outputArr( outputBlob.GetDataSize() );
	outputBlob.CopyTo( outputArr.data() );

	std::vector<float> expectedArr;
	bertConvNaive( dataArr, kernelArr, seqLen, batchSize, numHeads, headSize, kernelSize, expectedArr );

	ASSERT_EQ( expectedArr.size(), outputArr.size() );
	for( size_t i = 0; i < expectedArr.size(); ++i ) {
		ASSERT_NEAR( expectedArr[i], outputArr[i], 1e-3f );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBertConvTest : public CTestFixtureWithParams {
};

TEST_P(CMathEngineBertConvTest, Random)
{
	RUN_TEST_IMPL( bertConvTestImpl )
}

INSTANTIATE_TEST_CASE_P( CMathEngineBertConvTestInstantiation, CMathEngineBertConvTest,
	::testing::Values(
		CTestParams(
			"SeqLen = (10..100);"
			"BatchSize = (1..5);"
			"NumHeads = (1..100);"
			"HeadSize = (1..10);"
			"KernelSize = (1..10);"
			"TestCount = 100;"
		)
	)
);
