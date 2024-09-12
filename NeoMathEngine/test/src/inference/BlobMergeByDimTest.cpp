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

static void blobMergeByDimNaive( int dimNum, const CBlobDesc* from, const std::vector<std::vector<float>>& fromData, int fromCount,
	const CBlobDesc& to, std::vector<float>& toData )
{
	int s[CBlobDesc::MaxDimensions];
	to.GetDimSizes( s );
	int objectCount = 1;
	for(int z  = 0; z < dimNum; z++) {
		objectCount *= s[z];
	}
	int objectSize = to.BlobSize() / objectCount;
	for(int x = 0; x < objectCount; x++) {
		float *output = toData.data() + x * objectSize;
		for( int i = 0; i < fromCount; ++i ) {
			int fromLimits[CBlobDesc::MaxDimensions];
			from[i].GetDimSizes( fromLimits );
			int fromObjectSize = 1;
			for(int z = dimNum; z < CBlobDesc::MaxDimensions; z++) {
				fromObjectSize *= fromLimits[z];
			}
			for( int c = 0; c < fromObjectSize; c++ ) {
				output[c] = fromData[i][x * fromObjectSize + c];
			}
			output += fromObjectSize;
		}
	}
}

static void blobMergeByDimTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval dimSizeInterval = params.GetInterval( "DimSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );
	const CInterval fromCountInterval = params.GetInterval( "MergeFromCount" );

	const int mergeDim = random.UniformInt( 1, CBlobDesc::MaxDimensions - 1 );
	const int fromCount = random.UniformInt( fromCountInterval.Begin, fromCountInterval.End );

	CBlobDesc sourceInitalDesc( CT_Float );
	for( int dim = 0; dim < CBlobDesc::MaxDimensions; dim++ ) {
		sourceInitalDesc.SetDimSize( dim, random.UniformInt( dimSizeInterval.Begin, dimSizeInterval.End ) );
	}

	std::vector<CBlobDesc> sourceDescs;
	sourceDescs.insert( sourceDescs.begin(), fromCount, sourceInitalDesc );
	int mergeDimSum = 0;
	for( int i = 0; i < fromCount; i++ ) {
		sourceDescs[i].SetDimSize( mergeDim, random.UniformInt( dimSizeInterval.Begin, dimSizeInterval.End ) );
		mergeDimSum += sourceDescs[i].DimSize( mergeDim );
	}

	std::vector<std::vector<float>> sourceData;
	sourceData.resize( fromCount );
	for( int i = 0; i < fromCount; i++ ) {
		sourceData[i].resize( sourceDescs[i].BlobSize() );
		for( size_t j = 0; j < sourceData[i].size(); j++ ) {
			sourceData[i][j] = static_cast<float>( random.Uniform( valuesInterval.Begin, valuesInterval.End ) );
		}
	}

	std::vector<CFloatHandleVar*> fromHandleVars;
	std::vector<CFloatHandle> fromHandles;
	for( int i = 0; i < fromCount; i++ ) {
		fromHandleVars.push_back( new CFloatHandleVar( MathEngine(), sourceDescs[i].BlobSize() ) );
		fromHandles.push_back( fromHandleVars[i]->GetHandle() );
		MathEngine().DataExchangeTyped( fromHandles[i], sourceData[i].data(), sourceData[i].size() );
	}

	CBlobDesc resultDesc = sourceInitalDesc;
	resultDesc.SetDimSize( mergeDim, mergeDimSum );

	std::vector<float> expected;
	expected.resize( resultDesc.BlobSize() );
	blobMergeByDimNaive( mergeDim, sourceDescs.data(), sourceData, fromCount, resultDesc, expected );

	CFloatHandleStackVar resultHandle( MathEngine(), resultDesc.BlobSize() );
	MathEngine().BlobMergeByDim( static_cast<TBlobDim>( mergeDim ), sourceDescs.data(), fromHandles.data(), fromCount, resultDesc, resultHandle.GetHandle() );
	
	for(size_t i = 0; i < fromHandleVars.size(); i++) {
		delete fromHandleVars[i];
	}

	std::vector<float> result;
	result.resize( resultDesc.BlobSize() );
	MathEngine().DataExchangeTyped( result.data(), CConstFloatHandle( resultHandle.GetHandle() ), result.size() );

	for( size_t i = 0; i < result.size(); i++ ) {
		ASSERT_NEAR( expected[i], result[i], 1e-3 );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineBlobMergeByDimTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P(CMathEngineBlobMergeByDimTestInstantiation, CMathEngineBlobMergeByDimTest,
	::testing::Values(
		CTestParams(
			"DimSize = (1..7);"
			"Values = (-50..50);"
			"MergeFromCount = (1..5);"
			"TestCount = 100;"
		)
	)
);

TEST_P(CMathEngineBlobMergeByDimTest, Random)
{
	RUN_TEST_IMPL(blobMergeByDimTestImpl)
}
