/* Copyright © 2017-2022 ABBYY Production LLC

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
static void vectorCumSumAlongDimensionImpl( const T* input, T* output,
	int precedingDimension, int dimension, int followingDimension, bool reverse )
{
	const int step = reverse ? -precedingDimension : precedingDimension;
	const int firstObjIndex = reverse ? ( dimension - 1 ) : 0;
	for( int b = 0; b < followingDimension; ++b ) {
		int objIndex = ( b * dimension + firstObjIndex ) * precedingDimension; 
		for( int dim = 0; dim < dimension; ++dim ) {
			for( int ch = 0; ch < precedingDimension; ++ch ) {
				if( dim == 0 ) {
					output[objIndex + ch] = input[objIndex + ch];
				} else {
					output[objIndex + ch] = input[objIndex + ch] + output[objIndex + ch - step];
				}
			}
			objIndex += step;
		}
	}
}

template<class T>
static void vectorCumSumAlongDimensionTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval precedingInterval = params.GetInterval( "Preceding" );
	const CInterval dimensionInterval = params.GetInterval( "Dimension" );
	const CInterval followingInterval = params.GetInterval( "Following" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int preceding = random.UniformInt( precedingInterval.Begin, precedingInterval.End );
	const int dimension = random.UniformInt( dimensionInterval.Begin, dimensionInterval.End );
	const int following = random.UniformInt( followingInterval.Begin, followingInterval.End );
	const bool reverse = ( random.UniformInt( 0, 9 ) % 2 == 0 );

	CREATE_FILL_ARRAY( T, input, valuesInterval.Begin, valuesInterval.End, preceding * dimension * following, random );
	std::vector<T> expected( input.size() );
	vectorCumSumAlongDimensionImpl( input.data(), expected.data(), preceding, dimension, following, reverse );

	std::vector<T> actual( input.size() );
	MathEngine().VectorCumSumAlongDimension( CARRAY_WRAPPER( T, input ), preceding, dimension, following,
		CARRAY_WRAPPER( T, actual ), reverse );
	
	for( size_t i = 0; i < expected.size(); ++i ) {
		ASSERT_NEAR( static_cast<float>( expected[i] ), static_cast<float>( actual[i] ), 1e-4f );
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CVectorCumSumAlongDimensionsTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CVectorCumSumAlongDimensionsTestInstantiation, CVectorCumSumAlongDimensionsTest,
	::testing::Values(
		CTestParams(
			"Preceding = (1..50);"
			"Dimension = (1..50);"
			"Following = (1..50);"
			"Values = (-100..100);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CVectorCumSumAlongDimensionsTest, Random )
{
	RUN_TEST_IMPL( vectorCumSumAlongDimensionTestImpl<float> )
	RUN_TEST_IMPL( vectorCumSumAlongDimensionTestImpl<int> )
}
