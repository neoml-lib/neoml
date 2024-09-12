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

template<class T>
static void transposeMatrixTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval sizeInterval = params.GetInterval( "VectorSize" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int height = random.UniformInt( sizeInterval.Begin, sizeInterval.End );
	const int width = random.UniformInt( sizeInterval.Begin, sizeInterval.End );
	const int medium = random.UniformInt( sizeInterval.Begin, sizeInterval.End );
	const int channels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );

	const int matrixSize = batchSize * height * medium * width * channels;
	std::vector<T> matrix, matrixTransposed;
	matrix.resize( matrixSize );
	matrixTransposed.resize( matrixSize );
	for( int i = 0; i < matrixSize; ++i ) {
		matrix[i] = static_cast<T>( random.UniformInt( valuesInterval.Begin, valuesInterval.End ) );
	}

	for( int b = 0; b < batchSize; ++b ) {
		for( int j = 0; j < height; ++j ) {
			for( int m = 0; m < medium; ++m ) {
				for( int i = 0; i < width; ++i ) {
					for( int c = 0; c < channels; ++c ) {
						matrixTransposed[( ( ( b * width + i ) * medium + m ) * height + j ) * channels + c]=
							matrix[( ( ( b * height + j ) * medium + m ) * width + i ) * channels + c];
					}
				}
			}
		}
	}

	CMemoryHandleStackVar<T> from( MathEngine(), matrixSize );
	MathEngine().DataExchangeTyped<T>( from, matrix.data(), matrixSize );
	CMemoryHandleStackVar<T> result( MathEngine(), matrixSize );
	MathEngine().TransposeMatrix( batchSize, from, height, medium, width, channels, result, matrixSize );
	MathEngine().DataExchangeTyped<T>( matrix.data(), result, matrixSize );

	for( int i = 0; i < matrixSize; ++i ) {
		ASSERT_NEAR( matrix[i], matrixTransposed[i], 1e-3 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CTransposeMatrixTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CTransposeMatrixTestInstantiation, CTransposeMatrixTest,
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

TEST_P( CTransposeMatrixTest, Random )
{
	RUN_TEST_IMPL( transposeMatrixTestImpl<int> )
	RUN_TEST_IMPL( transposeMatrixTestImpl<float> )
}
