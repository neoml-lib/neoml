/* Copyright © 2017-2023 ABBYY

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

// These functions are implemented by MKL. MKL is not available on ARM.
#if FINE_ARCHITECTURE( FINE_X86 ) || FINE_ARCHITECTURE( FINE_X64 )

using namespace NeoML;
using namespace NeoMLTest;

class CMathEngineLUFactorizationTest : public CTestFixtureWithParams {
};

static void luFactorizationTest( int height, int width, const std::vector<float>& matrix, const std::vector<float>& expected )
{
	std::vector<float> actual = matrix;
	MathEngine().LUFactorization( height, width, CARRAY_FLOAT_WRAPPER( actual ) ); //x86 -specific

	for( unsigned i = 0; i < expected.size(); ++i ) {
		ASSERT_NEAR( expected[i], actual[i], 1e-3 );
	}
}

TEST_F( CMathEngineLUFactorizationTest, Precalc )
{
	if( MathEngine().GetType() != MET_Cpu ) {
		return;
	}

	luFactorizationTest( 5, 2, { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f },
		{ 0.f, 1.f, 0.25f, 0.75f, 0.5f, 0.5f, 0.75f, 0.25f, 1.f, 0.f } );

	luFactorizationTest( 2, 5, { 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f },
		{ 0.5f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f } );
}
#endif //FINE_ARCHITECTURE( FINE_X86 ) || FINE_ARCHITECTURE( FINE_X64 )

