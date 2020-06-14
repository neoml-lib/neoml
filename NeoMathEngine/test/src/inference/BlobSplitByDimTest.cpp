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

static void blobSplitByDimNaive( int dimNum, const CBlobDesc& from, const std::vector<float>& fromData, const CBlobDesc* to, std::vector<std::vector<float>>& toData, int toCount )
{
	int s[CBlobDesc::MaxDimensions];
	from.GetDimSizes( s );
	int objectCount = 1;
	for(int z  = 0; z < dimNum; z++) {
		objectCount *= s[z];
	}
	int objectSize = from.BlobSize() / objectCount;
	for(int x = 0; x < objectCount; x++) {
		const float *input = fromData.data() + x * objectSize;
		for( int i = 0; i < toCount; ++i ) {
			int toLimits[CBlobDesc::MaxDimensions];
			to[i].GetDimSizes( toLimits );
			int toObjectSize = 1;
			for(int z = dimNum; z < CBlobDesc::MaxDimensions; z++) {
				toObjectSize *= toLimits[z];
			}
			for( int c = 0; c < toObjectSize; c++ ) {
				toData[i][x * toObjectSize + c] = input[c];
			}
			input += toObjectSize;
		}
	}
}

static void blobSplitByDimTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval dimSizeInterval = params.GetInterval( "DimSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	CBlobDesc sourceDesc( CT_Float );
	for( int dim = 0; dim < CBlobDesc::MaxDimensions; dim++ ) {
		sourceDesc.SetDimSize( dim, random.UniformInt( dimSizeInterval.Begin, dimSizeInterval.End ) );
	}

	CREATE_FILL_FLOAT_ARRAY(sourceData, valuesInterval.Begin, valuesInterval.End, sourceDesc.BlobSize(), random)

	CFloatBlob sourceBlob( MathEngine(), sourceDesc.BatchLength(), sourceDesc.BatchWidth(), sourceDesc.ListSize(), sourceDesc.Height(),
		sourceDesc.Width(), sourceDesc.Depth(), sourceDesc.Channels() );
	sourceBlob.CopyFrom( sourceData.data() );

	const int splitDim = random.UniformInt( 1, CBlobDesc::MaxDimensions - 1 );
	const int toCount = random.UniformInt( 1, sourceDesc.DimSize(splitDim) );

	std::vector<int> splitDimSizes;
	splitDimSizes.insert( splitDimSizes.begin(), toCount, 1 );
	for( int i = 0; i < sourceDesc.DimSize( splitDim ) - toCount; i++ ) {
		int currentBlock = random.UniformInt( 0, toCount - 1 );
		splitDimSizes[currentBlock]++;
	}

	std::vector<CBlobDesc> toDescs;
	toDescs.insert( toDescs.begin(), toCount, sourceDesc );
	for( int i = 0; i < toCount; i++ ) {
		toDescs[i].SetDimSize( splitDim, splitDimSizes[i] );
	}

	std::vector<std::vector<float>> expected;
	expected.resize( toCount );
	for( int i = 0; i < toCount; i++ ) {
		expected[i].resize( toDescs[i].BlobSize() );
	}

	blobSplitByDimNaive( splitDim, sourceDesc, sourceData, toDescs.data(), expected, toCount );

	std::vector<CFloatHandleVar*> toHandleVars;
	std::vector<CFloatHandle> toHandles;
	for( int i = 0; i < toCount; i++ ) {
		toHandleVars.push_back( new CFloatHandleVar( MathEngine(), toDescs[i].BlobSize() ) );
		toHandles.push_back( toHandleVars[i]->GetHandle() );
	}

	MathEngine().BlobSplitByDim( static_cast<TBlobDim>(splitDim), sourceDesc, sourceBlob.GetData(), toDescs.data(), toHandles.data(), toCount );

	std::vector<std::vector<float>> result;
	result.resize( toCount );
	for( int i = 0; i < toCount; i++ ) {
		result[i].resize( toDescs[i].BlobSize() );
		MathEngine().DataExchangeTyped( result[i].data(), CConstFloatHandle( toHandles[i] ), result[i].size() );
	}

	for(size_t i = 0; i < toHandleVars.size(); i++) {
		delete toHandleVars[i];
	}

	for( int i = 0; i < toCount; i++ ) {
		for( size_t j = 0; j < expected[i].size(); j++ ) {
			ASSERT_NEAR( expected[i][j], result[i][j], 1e-3 );
		}
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBlobSplitByDimTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P(CMathEngineBlobSplitByDimTestInstantiation, CMathEngineBlobSplitByDimTest,
	::testing::Values(
		CTestParams(
			"DimSize = (1..7);"
			"Values = (-50..50);"
			"MergeFromCount = (1..5);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineBlobSplitByDimTest, Random)
{
	RUN_TEST_IMPL(blobSplitByDimTestImpl)
}
