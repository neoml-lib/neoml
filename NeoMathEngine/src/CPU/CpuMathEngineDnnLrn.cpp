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

#include <common.h>
#pragma hdrstop

#include <CpuMathEngine.h>
#include <CpuMathEnginePrivate.h>
#include <MathEngineDnnLrn.h>
#include <MemoryHandleInternal.h>
#include <NeoMathEngine/NeoMathEngineException.h>
#include <NeoMathEngine/OpenMP.h>

namespace NeoML {

static void channelwisePool( const float* input, float* output, int vectorCount, int vectorSize,
	int windowSize, float scale, float bias, int threadCount );

void CCpuMathEngine::lrnImpl( const CLrnDesc& lrnDesc, const CConstFloatHandle& input, const CFloatHandle& invSum,
	const CFloatHandle& invSumBeta, const CFloatHandle& output )
{
	ASSERT_EXPR( input.GetMathEngine() == this );
	ASSERT_EXPR( invSum.GetMathEngine() == this );
	ASSERT_EXPR( invSumBeta.GetMathEngine() == this );

	const CMathEngineLrnDesc& desc = static_cast<const CMathEngineLrnDesc&>( lrnDesc );

	const int vectorCount = desc.Source.ObjectCount() * desc.Source.GeometricalSize();
	const int vectorSize = desc.Source.Channels();
	const int dataSize = vectorCount * vectorSize;
	CFloatHandleStackVar buffer( *this, desc.Source.BlobSize() );

	{
		CFloatHandle sqrBuff = buffer;
		VectorEltwiseMultiply( input, input, sqrBuff, dataSize );
		channelwisePool( GetRaw( sqrBuff ), GetRaw( invSum ), vectorCount, vectorSize, desc.WindowSize,
			desc.Alpha / desc.WindowSize, desc.Bias, threadCount );
	}

	VectorInv( invSum, invSum, dataSize );
	VectorPower( desc.Beta, invSum, invSumBeta, dataSize );
	VectorEltwiseMultiply( invSumBeta, input, output, dataSize );

	/* const float* currInput = GetRaw( input );
	float* currOutput = GetRaw( output );

	for( int vec = 0; vec < vectorCount; ++vec ) {
		for( int ch = 0; ch < vectorSize; ++ch ) {
			double result = 0;
			const int firstC = max( 0, ch - ( desc.WindowSize - 1 ) / 2 );
			const int lastC = min( vectorSize - 1, ch + desc.WindowSize / 2 );
			for( int subC = firstC; subC <= lastC; ++subC ) {
				result += currInput[subC] * currInput[subC];
			}
			result *= desc.Alpha;
			result /= desc.WindowSize;
			result += desc.Bias;
			result = pow( result, static_cast<double>( -desc.Beta ) );
			*currOutput++ = static_cast<float>( result * currInput[ch] );
		}
		currInput += vectorSize;
	} */
}

// --------------------------------------------------------------------------------------------------------------------

CLrnDesc* CCpuMathEngine::InitLrn( const CBlobDesc& source, int windowSize, float bias, float alpha, float beta )
{
	return new CMathEngineLrnDesc( source, windowSize, bias, alpha, beta );
	return nullptr;
}

void CCpuMathEngine::Lrn( const CLrnDesc& lrnDesc, const CConstFloatHandle& input, const CFloatHandle& invSum,
	const CFloatHandle& invSumBeta, const CFloatHandle& output )
{
	lrnImpl( lrnDesc, input, invSum.IsNull() ? output: invSum, invSumBeta.IsNull() ? output : invSumBeta, output );
}

void CCpuMathEngine::LrnBackward( const CLrnDesc& /* desc */, const CConstFloatHandle& /* input */, const CConstFloatHandle& /* output */,
		const CConstFloatHandle& /* outputDiff */, const CConstFloatHandle& /* invSum */, const CConstFloatHandle& /* invSumBeta */,
		const CFloatHandle& /* inputDiff */ )
{
	ASSERT_EXPR( false );
}

// --------------------------------------------------------------------------------------------------------------------

#ifdef NEOML_USE_SSE

static void channelwisePool( const float* input, float* output, int vectorCount, int vectorSize,
	int windowSize, float scale, float bias, int threadCount )
{
	const int curThreadCount = IsOmpRelevant( vectorCount, vectorCount * vectorSize * windowSize ) ? threadCount : 1;
	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int index, count;
		if( OmpGetTaskIndexAndCount( vectorCount, index, count ) ) {
			const float* currInput = input + index * vectorSize;
			float* currOutput = output + index * vectorSize;
			for( int vec = 0; vec < count; ++vec ) {
				for( int ch = 0; ch < vectorSize; ++ch ) {
					const int firstC = max( 0, ch - ( windowSize - 1 ) / 2 );
					const float* windowStart = currInput + firstC;

					const int lastC = min( vectorSize - 1, ch + windowSize / 2 );
					const int currWindowSize = lastC - firstC + 1;

					int sseSize, nonSseSize;
					checkSse( currWindowSize, sseSize, nonSseSize );
					__m128 accum;

					if( nonSseSize > 0 ) {
						accum = LoadSse( windowStart, nonSseSize, 0 );
						windowStart += nonSseSize;
					} else if( sseSize > 0 ) {
						accum = LoadSse4( windowStart );
						windowStart += 4;
						sseSize--;
					} else {
						accum = _mm_set_ps1( 0 );
					}

					for( int i = 0; i < sseSize; ++i ) {
						accum = _mm_add_ps( LoadSse4( windowStart ), accum );
						windowStart += 4;
					}

					float res = _mm_cvtss_f32( HorizontalAddSse( accum ) );
					*currOutput++ = res * scale + bias;
				}
				currInput += vectorSize;
			}
		}
	}
}

#elif NEOML_USE_NEON

#else
#error "Unknown architecure"
#endif

} // namespace NeoML
