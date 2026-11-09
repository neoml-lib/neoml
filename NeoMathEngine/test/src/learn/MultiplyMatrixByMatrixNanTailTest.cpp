/* Copyright © 2017-2026 ABBYY Production LLC

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

// Regression test for the AVX sgemm micro-kernel 6x1 (Kernel_AVX_6x1.h).
// When k % 4 == 1 the kernel loads 8 floats of the interleaved A panel for the last k,
// although only 6 of them are valid, and zeroes the corresponding B lanes instead of the A lanes.
// If the 2 extra floats are NaN, NaN * 0 = NaN gets into the results of the panel rows 0 and 1.
// The extra floats lie in the alignment tail of the temporary A buffer taken from the math engine stack,
// so their content is left from previous stack allocations. The test fills the stack with NaN first.
//
// The custom AVX sgemm is used on non-Intel CPUs (and on Intel only when MKL is not used),
// so on other configurations the test passes regardless of the kernel.

#include <TestFixture.h>
#include <MeTestCommon.h>

#include <cmath>
#include <cstring>
#include <cstdint>

using namespace NeoML;
using namespace NeoMLTest;

// Quiet NaN with a non-default payload, the same as observed in FLEX-760 dumps.
static float poisonNan()
{
	const uint32_t bits = 0xFFFFFFFFu;
	float result;
	::memcpy( &result, &bits, sizeof( result ) );
	return result;
}

// Fills the top of the math engine stack with NaN and releases it,
// so the next stack allocations get the same memory with NaN in their alignment tails.
static void poisonMathEngineStack( IMathEngine& mathEngine )
{
	const int poisonSize = 1 << 16;
	CFloatHandleStackVar poison( mathEngine, poisonSize );
	mathEngine.VectorFill( poison.GetHandle(), poisonNan(), poisonSize );
}

static void multiplyWithPoisonedStack( int batchSize, int firstHeight, int firstWidth, int secondWidth, int seed )
{
	CRandom random( seed );

	CREATE_FILL_FLOAT_ARRAY( first, -1.f, 1.f, batchSize * firstHeight * firstWidth, random )
	CREATE_FILL_FLOAT_ARRAY( second, -1.f, 1.f, batchSize * firstWidth * secondWidth, random )

	const int resultSize = batchSize * firstHeight * secondWidth;
	std::vector<float> expected( resultSize, 0.f );
	std::vector<float> get( resultSize, 0.f );
	batchMultiplyMatrixByMatrixAndAddNaive( batchSize, first, second, firstHeight, firstWidth, secondWidth, expected );

	CFloatWrapper firstWrapper( MathEngine(), first.data(), static_cast<int>( first.size() ) );
	CFloatWrapper secondWrapper( MathEngine(), second.data(), static_cast<int>( second.size() ) );
	{
		CFloatWrapper resultWrapper( MathEngine(), get.data(), resultSize );
		poisonMathEngineStack( MathEngine() );
		MathEngine().MultiplyMatrixByMatrix( batchSize, firstWrapper, firstHeight, firstWidth,
			secondWrapper, secondWidth, resultWrapper, resultSize );
	}

	for( int i = 0; i < resultSize; ++i ) {
		ASSERT_FALSE( std::isnan( get[i] ) ) << "batch=" << batchSize << " m=" << firstHeight
			<< " k=" << firstWidth << " n=" << secondWidth << " index=" << i;
		ASSERT_NEAR( expected[i], get[i], 1e-3 ) << "batch=" << batchSize << " m=" << firstHeight
			<< " k=" << firstWidth << " n=" << secondWidth << " index=" << i;
	}
}

//---------------------------------------------------------------------------------------------------------------------

class CMultiplyMatrixByMatrixNanTailTest : public CTestFixture {
};

// The exact shapes of CGaussianMixtureDistributionFastEstimator for a 13-dimensional single-component mixture:
// inverse covariation (13 x 13) * centered point (13 x 1) and rotated point (1 x 13) * centered point (13 x 1).
TEST_F( CMultiplyMatrixByMatrixNanTailTest, GaussianMixtureShapes )
{
	multiplyWithPoisonedStack( 1, 13, 13, 1, 1 );
	multiplyWithPoisonedStack( 1, 1, 13, 1, 2 );
}

// Other shapes reaching the 6x1 micro-kernel (n % 16 is 1 or 9) with k % 4 == 1.
TEST_F( CMultiplyMatrixByMatrixNanTailTest, MicroKernel6x1KTail )
{
	const int heights[] = { 1, 2, 5, 6, 7, 12, 13, 31 };
	const int widths[] = { 1, 5, 9, 13, 17 };
	const int secondWidths[] = { 1, 9, 17, 25 };
	int seed = 100;
	for( int m : heights ) {
		for( int k : widths ) {
			for( int n : secondWidths ) {
				multiplyWithPoisonedStack( 1, m, k, n, seed++ );
			}
		}
	}
	multiplyWithPoisonedStack( 3, 13, 13, 1, seed++ );
}
