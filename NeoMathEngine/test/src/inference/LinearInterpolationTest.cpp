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

static void testLinearInterpolation( std::vector<float>& input, const std::vector<float>& expected,
	int objectCount, int scaledAxis, int objectSize, int scale )
{
	ASSERT_EQ( input.size(), static_cast<size_t>( objectCount ) * scaledAxis * objectSize );
	ASSERT_EQ( expected.size(), static_cast<size_t>( objectCount ) * scaledAxis * scale * objectSize );

	std::vector<float> actual( expected.size() );
	MathEngine().LinearInterpolation( CARRAY_FLOAT_WRAPPER( input ), CARRAY_FLOAT_WRAPPER( actual ),
		objectCount, scaledAxis, objectSize, scale );

	for( size_t i = 0; i < expected.size(); ++i ) {
		ASSERT_NEAR( actual[i], expected[i], 1e-3f );
	}
}

static void naiveLinearInterpolation( const float* input, float* output, int objectCount, int scaledAxis,
	int objectSize, int scale )
{
	for( int obj = 0; obj < objectCount; ++obj ) {
		for( int x = 0; x < scaledAxis; ++x ) {
			for( int inScale = 0; inScale < scale; ++inScale ) {
				for( int elem = 0; elem < objectSize; ++elem ) {
					*output++ = x == scaledAxis - 1 ? input[elem]
						: static_cast<float>( scale - inScale ) / scale * input[elem] + static_cast<float>( inScale ) / scale * input[elem + objectSize];
				}
			}
			input += objectSize;
		}
	}
}

static void testLinearInterpolationWithParams( const CTestParams& params, int seed )
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

	CREATE_FILL_FLOAT_ARRAY( input, valuesInterval.Begin, valuesInterval.End,
		objectCount * scaledAxis * objectSize, random );
	std::vector<float> expected( input.size() * scale );
	naiveLinearInterpolation( input.data(), expected.data(), objectCount, scaledAxis, objectSize, scale );

	testLinearInterpolation( input, expected, objectCount, scaledAxis, objectSize, scale );
}

class CMathEngineLinearInterpolationTest : public CTestFixtureWithParams {
};

TEST_F( CMathEngineLinearInterpolationTest, Precalc_Flat )
{
	std::vector<float> input{ 0.1f, 0.4f, 0.7f };
	std::vector<float> expected{ 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.7f, 0.7f };
	testLinearInterpolation( input, expected, 1, 3, 1, 3 );
}

TEST_F( CMathEngineLinearInterpolationTest, Precal_3D )
{
	std::vector<float> input{
		1, 2,
		3, 4,

		5, 6,
		7, 8
	};

	std::vector<float> expected{
		1, 2,
		2, 3,
		3, 4,
		3, 4,

		5, 6,
		6, 7,
		7, 8,
		7, 8
	};

	testLinearInterpolation( input, expected, 2, 2, 2, 2 );
}

TEST_P( CMathEngineLinearInterpolationTest, Random )
{
	RUN_TEST_IMPL( testLinearInterpolationWithParams );
}

INSTANTIATE_TEST_CASE_P( CMathEngineLinearInterpolationTestInstantiation, CMathEngineLinearInterpolationTest,
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
