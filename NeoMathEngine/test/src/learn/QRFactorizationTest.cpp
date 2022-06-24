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

class CMathEngineQRFactorizationTest : public CTestFixtureWithParams {
};

static void qrFactorizationTest( int height, int width, std::vector<float> matrix, std::vector<float> expectedQ, std::vector<float> expectedR )
{
	std::vector<float> q( height * std::min( height, width ) );
	std::vector<float> r( height * width );
	for( bool returnQ : { false, true } ) {
		for( bool returnR : { false, true } ) {
			{
				auto qWrapper = CARRAY_FLOAT_WRAPPER( q );
				auto rWrapper = CARRAY_FLOAT_WRAPPER( r );
				CFloatHandle qHandle = qWrapper;
				CFloatHandle rHandle = rWrapper;
				MathEngine().QRFactorization( height, width, CARRAY_FLOAT_WRAPPER( matrix ), returnQ ? &qHandle : nullptr,
					returnR ? &rHandle : nullptr, false, returnQ, returnR );
			}
			if( returnQ ) {
				for( unsigned i = 0; i < q.size(); ++i ) {
					ASSERT_NEAR( expectedQ[i], q[i], 1e-3 );
				}
			}
			if( returnR ) {
				for( unsigned i = 0; i < r.size(); ++i ) {
					ASSERT_NEAR( expectedR[i], r[i], 1e-3 );
				}
			}
		}
	}

	for( bool returnQ : { false, true } ) {
		for( bool returnR : { false, true } ) {
			r = matrix;
			{
				auto qWrapper = CARRAY_FLOAT_WRAPPER( q );
				CFloatHandle qHandle = qWrapper;
				MathEngine().QRFactorization( height, width, CARRAY_FLOAT_WRAPPER( r ), &qHandle, nullptr, true, returnQ, returnR );
			}
			if( returnQ ) {
				for( unsigned i = 0; i < q.size(); ++i ) {
					ASSERT_NEAR( expectedQ[i], q[i], 1e-3 );
				}
			}
			if( returnR ) {
				for( unsigned i = 0; i < r.size(); ++i ) {
					ASSERT_NEAR( expectedR[i], r[i], 1e-3 );
				}
			}
		}
	}
}

TEST_F( CMathEngineQRFactorizationTest, Precalc )
{
	if( MathEngine().GetType() != MET_Cpu ) {
		return;
	}

	qrFactorizationTest( 3, 5, { 6.8, 6.2, 4.0, 3.9, 5.7, 5.9, 7.1, 6.1, 7.2, 2.4, 7.9, 4.2, 9.6, 6.3, 9.0 },
		{ -0.5677, -0.1967, -0.7994, -0.4926, -0.6968, 0.5213, -0.6596, 0.6897, 0.2987 },
		{ -11.9775, -9.7875, -11.6076, -9.9161, -10.3544, 0.0000, -3.2702, 1.5840, -1.4390, 3.4140, 0.0000, 0.0000, 2.8503, 2.5180, -0.6168 } );

	qrFactorizationTest( 4, 4, { 1.6, 6.2, 1.8, 7.0, 4.3, 4.4, 1.7, 7.2, 4.6, 7.8, 3.7, 9.1, 0.8, 5.4, 7.3, 1.3 },
		{ -0.2444, 0.5990, 0.6795, 0.3460, -0.6569, -0.4181, -0.1746, 0.6027, -0.7027, 0.0642, 0.0505, -0.7068, -0.1222, 0.6799, -0.7108, 0.1326 },
		{ -6.5460, -10.5469, -5.0489, -12.9942, 0.0000, 6.0467, 5.5683, 2.6512, 0.0000, 0.0000, -4.0759, 3.0347, 0.0000, 0.0000, 0.0000, 0.5020 } );

	qrFactorizationTest( 5, 3, { 2.1, 0.4, 9.1, 4.6, 9.6, 3.9, 1.1, 7.2, 2.0, 7.9, 6.8, 7.7, 6.0, 7.7, 7.2 },
		{ -0.1877, 0.2724, -0.9184, -0.4111, -0.5016, 0.0918, -0.0983, -0.7374, -0.2657, -0.7061, 0.3582, 0.2722, -0.5362, -0.0472, -0.0587 },
		{ -11.1888, -13.6600, -12.8056, 0.0000, -7.9431, 1.4661, 0.0000, 0.0000, -6.8569, 0, 0, 0, 0, 0, 0 } );
}