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

static void bertConvBackwardNaive( const std::vector<float>& data, const std::vector<float>& kernel,
	const std::vector<float>& outputDiff, int seqLen, int batchSize, int numHeads, int headSize, int kernelSize,
	std::vector<float>& dataDiff, std::vector<float>& kernelDiff )
{
	assert( static_cast<int>( outputDiff.size() ) == seqLen * batchSize * numHeads * headSize );
	assert( static_cast<int>( data.size() ) == seqLen * batchSize * numHeads * headSize );
	assert( static_cast<int>( kernel.size() ) == seqLen * batchSize * numHeads * kernelSize );

	dataDiff.resize( 0 );
	dataDiff.resize( static_cast<size_t>( seqLen ) * batchSize * numHeads * headSize, 0 );
	kernelDiff.resize( 0 );
	kernelDiff.resize( static_cast<size_t>( seqLen ) * batchSize * numHeads * kernelSize, 0 );

	const int pad = ( kernelSize - 1 ) / 2;
	const int dataSeqStep = batchSize * numHeads * headSize;

	int outOffset = 0;
	int kernelOffset = 0;
	for( int seq = 0; seq < seqLen; ++seq ) {
		for( int b = 0; b < batchSize * numHeads; ++b ) {
			for( int h = 0; h < headSize; ++h ) {
				int dataOffset = h + b * headSize + ( seq - pad ) * dataSeqStep;
				for( int k = 0; k < kernelSize; ++k ) {
					if( dataOffset >= 0 && dataOffset < static_cast<int>( dataDiff.size() ) ) {
						dataDiff[dataOffset] += outputDiff[outOffset] * kernel[kernelOffset + k];
						kernelDiff[kernelOffset + k] += data[dataOffset] * outputDiff[outOffset];
					}
					dataOffset += dataSeqStep;
				}
				outOffset++;
			}
			kernelOffset += kernelSize;
		}
	}
}

static void bertConvBackwardTestImpl( const CTestParams& params, int seed )
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

	CFloatBlob outputDiffBlob( MathEngine(), seqLen, batchSize * numHeads, 1, headSize, 1, 1, 1 );
	CREATE_FILL_FLOAT_ARRAY( outputDiffArr, -1.f, 1.f, outputDiffBlob.GetDataSize(), random )
	outputDiffBlob.CopyFrom( outputDiffArr.data() );

	CFloatBlob dataDiffBlob( MathEngine(), seqLen, batchSize, 1, 1, 1, 1, numHeads * headSize );
	MathEngine().VectorFill( dataDiffBlob.GetData(), 0.f, dataDiffBlob.GetDataSize() );
	CFloatBlob kernelDiffBlob( MathEngine(), seqLen, batchSize * numHeads, 1, kernelSize, 1, 1, 1 );
	MathEngine().VectorFill( kernelDiffBlob.GetData(), 0.f, kernelDiffBlob.GetDataSize() );

	MathEngine().BertConvBackward( dataBlob.GetData(), kernelBlob.GetData(), outputDiffBlob.GetData(),
		seqLen, batchSize, numHeads, headSize, kernelSize, dataDiffBlob.GetData(), kernelDiffBlob.GetData() );

	std::vector<float> dataDiffArr( dataDiffBlob.GetDataSize() );
	dataDiffBlob.CopyTo( dataDiffArr.data() );
	std::vector<float> kernelDiffArr( kernelDiffBlob.GetDataSize() );
	kernelDiffBlob.CopyTo( kernelDiffArr.data() );

	std::vector<float> expectedDataDiffArr;
	std::vector<float> expectedKernelDiffArr;
	bertConvBackwardNaive( dataArr, kernelArr, outputDiffArr, seqLen, batchSize, numHeads, headSize, kernelSize,
		expectedDataDiffArr, expectedKernelDiffArr );

	ASSERT_EQ(expectedDataDiffArr.size(), dataDiffArr.size());
	for( size_t i = 0; i < expectedDataDiffArr.size(); ++i ) {
		if( ::fabsf( expectedDataDiffArr[i] - dataDiffArr[i] ) >= 1e-3f ) {
			::printf( "hrere\n" );
		}
		ASSERT_NEAR( expectedDataDiffArr[i], dataDiffArr[i], 1e-3f );
	}

	ASSERT_EQ( expectedKernelDiffArr.size(), kernelDiffArr.size() );
	for( size_t i = 0; i < expectedKernelDiffArr.size(); ++i ) {
		if( ::fabsf( expectedKernelDiffArr[i] - kernelDiffArr[i] ) >= 1e-3f ) {
			::printf( "hrere\n" );
		}
		ASSERT_NEAR( expectedKernelDiffArr[i], kernelDiffArr[i], 1e-3f );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBertConvBackwardTest : public CTestFixtureWithParams {
};

TEST_P( CMathEngineBertConvBackwardTest, Random )
{
	RUN_TEST_IMPL( bertConvBackwardTestImpl )
}

INSTANTIATE_TEST_CASE_P( CMathEngineBertConvBackwardTestInstantiation, CMathEngineBertConvBackwardTest,
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
