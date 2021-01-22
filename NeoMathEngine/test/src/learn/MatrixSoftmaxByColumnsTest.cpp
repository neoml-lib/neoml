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

static void softmaxImpl( const std::vector<float>& input, int height, int width, bool byRow, std::vector<float>& output )
{
	std::vector<float> maxima;
	maxima.resize( byRow ? height : width );
	output = input;

	for( int i = 0; i < height; ++i ) {
		for( int j = 0; j < width; ++j ) {
			if( byRow ) {
				maxima[i] = j == 0 ? input[i * width + j] : std::max( maxima[i], input[i * width + j] );
			}
			else {
				maxima[j] = i == 0 ? input[i * width + j] : std::max( maxima[j], input[i * width + j] );
			}
		}
	}

	for( int i = 0; i < height; ++i ) {
		for( int j = 0; j < width; ++j ) {
			output[i * width + j] -= byRow ? maxima[i] : maxima[j];
			output[i * width + j] = expf( std::min( FLT_MAX_LOG, std::max( FLT_MIN_LOG, output[i * width + j] ) ) );
		}
	}

	std::vector<float> sums;
	sums.resize( byRow ? height : width );

	for( int i = 0; i < height; ++i ) {
		for( int j = 0; j < width; ++j ) {
			if( byRow ) {
				sums[i] = j == 0 ? output[i * width + j] : sums[i] + output[i * width + j];
			}
			else {
				sums[j] = i == 0 ? output[i * width + j] : sums[j] + output[i * width + j];
			}
		}
	}

	for( int i = 0; i < height; ++i ) {
		for( int j = 0; j < width; ++j ) {
			output[i * width + j] /= byRow ? sums[i] : sums[j];
		}
	}
}

static void matrixSoftmaxByColumnsTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval heightInterval = params.GetInterval( "Height" );
	const CInterval widthInterval = params.GetInterval( "Width" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int height = random.UniformInt( heightInterval.Begin, heightInterval.End );
	const int width = random.UniformInt( widthInterval.Begin, widthInterval.End );

	CREATE_FILL_FLOAT_ARRAY( matrix, valuesInterval.Begin, valuesInterval.End, height * width, random )
	std::vector<float> get, expected;
	get.insert( get.begin(), height * width, 0 );
	expected.insert( expected.begin(), height * width, 0 );

	MathEngine().MatrixSoftmaxByColumns( CARRAY_FLOAT_WRAPPER( matrix ), height, width, CARRAY_FLOAT_WRAPPER( get ) );

	softmaxImpl( matrix, height, width, false, expected );

	for( int i = 0; i < height; ++i ) {
		ASSERT_NEAR( expected[i], get[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMatrixSoftmaxByColumnsTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CMatrixSoftmaxByColumnsTestInstantiation, CMatrixSoftmaxByColumnsTest,
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

TEST_P( CMatrixSoftmaxByColumnsTest, Random )
{
	RUN_TEST_IMPL( matrixSoftmaxByColumnsTestImpl )
}
