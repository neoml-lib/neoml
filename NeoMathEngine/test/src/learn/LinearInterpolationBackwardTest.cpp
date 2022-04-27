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

static void testLinearInterpolationBackward( std::vector<float>& outputDiff, const std::vector<float>& expected,
	int objectCount, int scaledAxis, int objectSize, int scale )
{
	ASSERT_EQ( outputDiff.size(), static_cast<size_t>( objectCount ) * scaledAxis * scale * objectSize );
	ASSERT_EQ( expected.size(), static_cast<size_t>( objectCount ) * scaledAxis * objectSize );

	std::vector<float> actual( expected.size() );
	MathEngine().LinearInterpolationBackward( CARRAY_FLOAT_WRAPPER( outputDiff ), CARRAY_FLOAT_WRAPPER( actual ),
		objectCount, scaledAxis, objectSize, scale );

	for( size_t i = 0; i < expected.size(); ++i ) {
		ASSERT_NEAR( actual[i], expected[i], 1e-3f );
	}
}

static void naiveLinearInterpolationBackward( const float* outputDiff, float* inputDiff, int objectCount, int scaledAxis,
	int objectSize, int scale )
{
	for( int i = 0; i < objectCount * scaledAxis * objectSize; ++i ) {
		inputDiff[i] = 0;
	}

	for( int obj = 0; obj < objectCount; ++obj ) {
		for( int x = 0; x < scaledAxis; ++x ) {
			for( int inScale = 0; inScale < scale; ++inScale ) {
				for( int elem = 0; elem < objectSize; ++elem ) {
					if( x == scaledAxis - 1 ) {
						inputDiff[elem] += *outputDiff;
					} else {
						inputDiff[elem] += static_cast<float>( scale - inScale ) / scale * *outputDiff;
						inputDiff[elem + objectSize] += static_cast<float>( inScale ) / scale * *outputDiff;
					}
					outputDiff++;
				}
			}
			inputDiff += objectSize;
		}
	}
}

static void testLinearInterpolationBackwardWithParams( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval objectCountInterval = params.GetInterval( "ObjectCount" );
	const CInterval scaledAxisInterval = params.GetInterval( "ScaledAxis" );
	const CInterval objectSizeInterval = params.GetInterval( "ObjectSize" );
	const CInterval scaleInterval = params.GetInterval( "Scale" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int objectCount = random.UniformInt( objectCountInterval.Begin, objectCountInterval.End );
	const int scaledAxis = random.UniformInt( scaledAxisInterval.Begin, scaledAxisInterval.End );
	const int objectSize = random.UniformInt( objectSizeInterval.Begin, objectSizeInterval.End );
	const int scale = random.UniformInt( scaleInterval.Begin, scaleInterval.End );

	CREATE_FILL_FLOAT_ARRAY( outputDiff, valuesInterval.Begin, valuesInterval.End,
		objectCount * scaledAxis * scale * objectSize, random );
	std::vector<float> expected( objectCount * scaledAxis * objectSize );
	naiveLinearInterpolationBackward( outputDiff.data(), expected.data(), objectCount, scaledAxis, objectSize, scale );

	testLinearInterpolationBackward( outputDiff, expected, objectCount, scaledAxis, objectSize, scale );
}

class CMathEngineLinearInterpolationBackwardTest : public CTestFixtureWithParams {
};

TEST_F( CMathEngineLinearInterpolationBackwardTest, Precalc_FlatOnes )
{
	std::vector<float> outputDiff = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
	std::vector<float> expected = { 2.f, 3.f, 4.f };
	testLinearInterpolationBackward( outputDiff, expected, 1, 3, 1, 3 );
}

TEST_F( CMathEngineLinearInterpolationBackwardTest, Precalc_Flat )
{
	std::vector<float> outputDiff = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
	std::vector<float> expected = { 10.f / 3.f, 12.f, 89.f / 3.f };
	testLinearInterpolationBackward( outputDiff, expected, 1, 3, 1, 3 );
}

TEST_F( CMathEngineLinearInterpolationBackwardTest, Precalc_3D )
{
	std::vector<float> outputDiff = {
		1.f, 2.f,
		3.f, 4.f,
		5.f, 6.f,
		7.f, 8.f,

		9.f, 10.f,
		11.f, 12.f,
		13.f, 14.f,
		15.f, 16.f
	};
	std::vector<float> expected = {
		2.5f, 4.f,
		13.5f, 16.f,

		14.5f, 16.f,
		33.5f, 36.f
	};
	testLinearInterpolationBackward( outputDiff, expected, 2, 2, 2, 2 );
}

TEST_P( CMathEngineLinearInterpolationBackwardTest, Random )
{
	RUN_TEST_IMPL( testLinearInterpolationBackwardWithParams );
}

INSTANTIATE_TEST_CASE_P( CMathEngineLinearInterpolationBackwardTestInstantiation, CMathEngineLinearInterpolationBackwardTest,
	::testing::Values(
		CTestParams(
			"ObjectCount = (1..3);"
			"ScaledAxis = (1..3);"
			"ObjectSize = (1..3);"
			"Scale = (1..3);"
			"Values = (-10..15);"
			"TestCount = 100;"
		),
		CTestParams(
			"ObjectCount = (1..10);"
			"ScaledAxis = (1..10);"
			"ObjectSize = (1..10);"
			"Scale = (1..10);"
			"Values = (-10..15);"
			"TestCount = 100;"
		)
	)
);
