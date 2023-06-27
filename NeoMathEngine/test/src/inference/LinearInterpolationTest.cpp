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

static void testLinearInterpolation( TInterpolationCoords coords, TInterpolationRound round, std::vector<float>& input,
	const std::vector<float>& expected, int objectCount, int scaledAxis, int objectSize, float scale )
{
	ASSERT_EQ( input.size(), static_cast<size_t>( objectCount ) * scaledAxis * objectSize );
	ASSERT_EQ( expected.size(), static_cast<size_t>( objectCount ) * static_cast<int>( scaledAxis * scale ) * objectSize );

	std::vector<float> actual( expected.size() );
	MathEngine().LinearInterpolation( CARRAY_FLOAT_WRAPPER( input ), CARRAY_FLOAT_WRAPPER( actual ),
		coords, round, objectCount, scaledAxis, objectSize, scale );

	for( size_t i = 0; i < expected.size(); ++i ) {
		ASSERT_NEAR( actual[i], expected[i], 1e-3f );
	}
}

static void naiveLinearInterpolation( TInterpolationCoords coords, TInterpolationRound round, const float* input, float* output,
	int objectCount, int scaledAxis, int objectSize, float scale )
{
	const int newSize = static_cast<int>( scaledAxis * scale );
	for( int obj = 0; obj < objectCount; ++obj ) {
		for( int xNew = 0; xNew < static_cast<int>( scale * scaledAxis ); ++xNew ) {
			float xOld = -1.f;
			switch( coords ) {
				case TInterpolationCoords::HalfPixel:
					xOld = ( xNew + 0.5f ) / scale - 0.5f;
					break;
				case TInterpolationCoords::PytorchHalfPixel:
					xOld = newSize > 1 ? ( xNew + 0.5f ) / scale - 0.5f : 0;
					break;
				case TInterpolationCoords::AlignCorners:
					xOld = static_cast<float>( xNew * ( scaledAxis - 1 ) ) / ( newSize - 1 );
					break;
				case TInterpolationCoords::Asymmetric:
					xOld = xNew / scale;
					break;
				default:
					ASSERT_TRUE( false ) << "Unknown coordinate system";
			}
			switch( round ) {
				case TInterpolationRound::None:
					break;
				case TInterpolationRound::RoundPreferFloor:
					if( xOld == static_cast<int>( xOld ) + 0.5f ) {
						xOld = ::floorf( xOld );
					} else {
						xOld = ::roundf( xOld );
					}
					break;
				case TInterpolationRound::RoundPreferCeil:
					xOld = ::roundf( xOld );
					break;
				case TInterpolationRound::Floor:
					xOld = ::floorf( xOld );
					break;
				case TInterpolationRound::Ceil:
					xOld = ::ceilf( xOld );
					break;
				default:
					ASSERT_TRUE( false ) << "Unknown rounding";
			}
			const float* currInput = input + obj * scaledAxis * objectSize;
			for( int elem = 0; elem < objectSize; ++elem ) {
				if( xOld <= 0 ) {
					*output++ = *currInput++;
				} else if( xOld >= scaledAxis - 1 ) {
					*output++ = currInput[objectSize * ( scaledAxis - 1 )];
					++currInput;
				} else {
					const int leftCoord = static_cast<int>( xOld );
					const int rightCoord = leftCoord + 1;
					const float rightMul = xOld - ::floorf( xOld );
					const float leftMul = 1 - rightMul;
					*output++ = currInput[objectSize * leftCoord] * leftMul
						+ currInput[objectSize * rightCoord] * rightMul;
					++currInput;
				}
			}
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

	for( int coords = 0; coords < static_cast<int>( TInterpolationCoords::Count ); ++coords ) {
		const int minOutputSize = coords == static_cast<int>( TInterpolationCoords::AlignCorners ) ? 2 : 1;
		const double minScale = static_cast<double>( minOutputSize ) / scaledAxis;
		float scale = static_cast<float>( random.Uniform( std::max( static_cast<double>( scaleInterval.Begin ), minScale ),
			std::max( static_cast<double>( scaleInterval.End ), minScale ) ) );
		while( static_cast<int>( scaledAxis * scale ) < minOutputSize ) {
			scale = static_cast<float>( random.Uniform( static_cast<double>( scaleInterval.Begin ),
				static_cast<double>( scaleInterval.End ) ) );
		}

		for( int round = 0; round < static_cast<int>( TInterpolationRound::Count ); ++round ) {
			CREATE_FILL_FLOAT_ARRAY( input, valuesInterval.Begin, valuesInterval.End,
				objectCount * scaledAxis * objectSize, random );
			std::vector<float> expected( objectCount * static_cast<int>( scaledAxis * scale ) * objectSize );
			naiveLinearInterpolation( static_cast<TInterpolationCoords>( coords ), static_cast<TInterpolationRound>( round ),
				input.data(), expected.data(), objectCount, scaledAxis, objectSize, scale );

			testLinearInterpolation( static_cast<TInterpolationCoords>( coords ), static_cast<TInterpolationRound>( round ),
				input, expected, objectCount, scaledAxis, objectSize, scale );
		}
	}
}

class CMathEngineLinearInterpolationTest : public CTestFixtureWithParams {
};

TEST_F( CMathEngineLinearInterpolationTest, Precalc_FlatAsymmetric )
{
	std::vector<float> input{ 0.1f, 0.4f, 0.7f };
	std::vector<float> expected{ 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.7f, 0.7f };
	testLinearInterpolation( TInterpolationCoords::Asymmetric, TInterpolationRound::None, input, expected, 1, 3, 1, 3.f );
}

TEST_F( CMathEngineLinearInterpolationTest, Precalc_FlatPytorchHalfPixel )
{
	std::vector<float> input{ 0.f, 1.f, 2.f };
	std::vector<float> expected{ 0.f, 0.f, 1.f / 3, 2.f / 3, 1., 4.f / 3, 5.f / 3, 2.f, 2.f };
	testLinearInterpolation( TInterpolationCoords::PytorchHalfPixel, TInterpolationRound::None, input, expected, 1, 3, 1, 3.f );
}

TEST_F( CMathEngineLinearInterpolationTest, Precal_3DAsymmetrict )
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

	testLinearInterpolation( TInterpolationCoords::Asymmetric, TInterpolationRound::None, input, expected, 2, 2, 2, 2.f );
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
			"Scale = (0..3);"
			"Values = (-10..15);"
			"TestCount = 100;"
		),
		CTestParams(
			"ObjectCount = (1..10);"
			"ScaledAxis = (1..10);"
			"ObjectSize = (1..10);"
			"Scale = (0..10);"
			"Values = (-10..15);"
			"TestCount = 100;"
		)
	)
);

