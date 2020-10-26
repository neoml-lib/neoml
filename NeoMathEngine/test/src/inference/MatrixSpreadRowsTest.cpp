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

template<class T>
static T matrixSpreadRowsTestRandomFunc( CRandom& )
{
	assert( false );
}

template<>
int matrixSpreadRowsTestRandomFunc<int>( CRandom& random )
{
	return random.UniformInt( -1000, 1000 );
}

template<>
float matrixSpreadRowsTestRandomFunc<float>( CRandom& random )
{
	return static_cast<float>( random.Uniform( -5., 15. ) );
}

template <class T>
void matrixSpreadRowsTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval resultHeightInterval = params.GetInterval( "ResultHeight" );

	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );
	const int resultHeight = random.UniformInt( std::max( height, resultHeightInterval.Begin ), resultHeightInterval.End );

	ASSERT_TRUE( height <= resultHeight );

	CBlob<T> input( MathEngine(), 1, height, width, 1 );
	CBlob<T> output( MathEngine(), 1, resultHeight, width, 1 );

	CIntBlob indices( MathEngine(), 1, 1, 1, 1, height );

	T defaultValue = matrixSpreadRowsTestRandomFunc<T>( random );

	std::vector<T> inputBuff;
	inputBuff.resize( input.GetDataSize() );
	std::vector<int> indicesBuff;
	indicesBuff.resize( height );
	std::vector<T> expected;
	expected.insert( expected.begin(), output.GetDataSize(), defaultValue );
	std::vector<T> actual;
	actual.resize( output.GetDataSize() );

	std::vector<bool> usedRow;
	usedRow.insert( usedRow.begin(), resultHeight, false );

	for( int i = 0; i < height; ++i ) {
		int index = random.UniformInt( -1, resultHeight - 1 );
		if( index >= 0 ) {
			while( usedRow[index] ) {
				index = ( index + 1 ) % resultHeight;
			}
			usedRow[index] = true;
		}
		for( int j = 0; j < width; ++j ) {
			inputBuff[i * width + j] = matrixSpreadRowsTestRandomFunc<T>( random );
			if( index >= 0 ) {
				expected[index * width + j] = inputBuff[i * width + j];
			}
		}
		indicesBuff[i] = index;
	}

	CMemoryHandleStackVar<T> defaultValueHandle(MathEngine());
	defaultValueHandle.SetValue( defaultValue );

	indices.CopyFrom( indicesBuff.data() );
	input.CopyFrom( inputBuff.data() );
	MathEngine().MatrixSpreadRows( input.GetData(), height, width, output.GetData(),
		resultHeight, indices.GetData(), defaultValueHandle );
	output.CopyTo( actual.data() );

	for( size_t i = 0; i < expected.size(); ++i ) {
		ASSERT_EQ( expected[i], actual[i] );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMathEngineMatrixSpreadRowsTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMathEngineMatrixSpreadRowsTestInstantiation, CMathEngineMatrixSpreadRowsTest,
	::testing::Values(
		CTestParams(
			"Height = 37;"
			"Width = 12016;"
			"ResultHeight = 107;"
			"TestCount = 1"
		),
		CTestParams(
			"Height = (1..7);"
			"Width = (1..7);"
			"ResultHeight = (1..15);"
			"TestCount = 1000"
		),
		CTestParams(
			"Height = (1..50);"
			"Width = (1..500);"
			"ResultHeight = (1..215);"
			"TestCount = 100"
		)
	)
);

TEST_P(CMathEngineMatrixSpreadRowsTest, Random)
{
	RUN_TEST_IMPL(matrixSpreadRowsTestImpl<int>)
	RUN_TEST_IMPL(matrixSpreadRowsTestImpl<float>)
}
