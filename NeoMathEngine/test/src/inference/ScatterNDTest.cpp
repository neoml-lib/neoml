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
#include <numeric>
#include <algorithm>

using namespace NeoML;
using namespace NeoMLTest;

template<class T>
static void scatterNDTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval indexDimsInterval = params.GetInterval( "IndexDims" );
	const int indexDims = random.UniformInt( indexDimsInterval.Begin, indexDimsInterval.End );

	CBlobDesc dataDesc;
	const CInterval dimSizeInterval = params.GetInterval( "DimSize" );
	int objectSize = 1;
	for( TBlobDim dim = BD_BatchLength; dim != BD_Count; ++dim ) {
		dataDesc.SetDimSize( dim, random.UniformInt( dimSizeInterval.Begin, dimSizeInterval.End ) );
		if( static_cast<int>( dim ) >= indexDims ) {
			objectSize *= dataDesc.DimSize( dim );
		}
	}

	const int objectCount = dataDesc.BlobSize() / objectSize;
	const CInterval updateCountInterval = params.GetInterval( "UpdateCount" );
	const int updateCount = std::min( objectCount,
		random.UniformInt( updateCountInterval.Begin, updateCountInterval.End ) );

	const CInterval valuesInterval = params.GetInterval( "Values" );

	CREATE_FILL_ARRAY( T, data, valuesInterval.Begin, valuesInterval.End, dataDesc.BlobSize(), random )
	CREATE_FILL_ARRAY( T, updates, valuesInterval.Begin, valuesInterval.End, updateCount * objectSize, random )
	std::vector<int> indices( updateCount * indexDims );
	std::vector<T> expected = data;

	std::vector<int> perm( updateCount );
	std::iota( perm.begin(), perm.end(), 0 );
	for( int updateIndex = 0; updateIndex < updateCount; ++updateIndex ) {
		std::swap( perm[updateIndex], perm[random.UniformInt( updateIndex, updateCount - 1 )] );
		std::copy_n( updates.data() + updateIndex * objectSize, objectSize,
			expected.data() + perm[updateIndex] * objectSize );
		for( int i = indexDims - 1; i >= 0; --i ) {
			indices[updateIndex * indexDims + i] = perm[updateIndex] % dataDesc.DimSize( i );
			perm[updateIndex] /= dataDesc.DimSize( i );
		}
	}

	MathEngine().ScatterND( CARRAY_INT_WRAPPER( indices ), CARRAY_WRAPPER( T, updates ), CARRAY_WRAPPER( T, data ),
		dataDesc, updateCount, indexDims );

	for( size_t i = 0; i < data.size(); ++i ) {
		ASSERT_EQ( expected[i], data[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CMathEngineScatterNDTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineScatterNDTestInstantiation, CMathEngineScatterNDTest,
	::testing::Values(
		CTestParams(
			"UpdateCount = (1..10);"
			"IndexDims = (1..7);"
			"DimSize = (1..3);"
			"Values = (-10..10);"
			"TestCount = 100;"
		),
		CTestParams(
			"UpdateCount = (1..1000);"
			"IndexDims = (1..7);"
			"DimSize = (1..7);"
			"Values = (-10..10);"
			"TestCount = 1000;"
		)
	)
);

TEST_P( CMathEngineScatterNDTest, Random )
{
	RUN_TEST_IMPL( scatterNDTestImpl<int> );
	RUN_TEST_IMPL( scatterNDTestImpl<float> );
}
