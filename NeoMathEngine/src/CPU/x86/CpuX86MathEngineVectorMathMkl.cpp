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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_SSE

#include <CpuMathEngine.h>
#include <CpuX86.h>
#include <float.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>

#ifdef NEOML_USE_MKL
#if FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
#include <mkl.h>
#else
#error Unknown platform
#endif
#endif

namespace NeoML {

void CCpuMathEngine::VectorExp(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

#ifdef NEOML_USE_MKL
	CFloatHandleStackVar minLimit( mathEngine(), 1 );
	minLimit.SetValue( FLT_MIN_LOG );
	CFloatHandleStackVar maxLimit( mathEngine(), 1 );
	maxLimit.SetValue( FLT_MAX_LOG );
	VectorMinMax(firstHandle, resultHandle, vectorSize, minLimit, maxLimit);
	vsExp(vectorSize, GetRaw(resultHandle), GetRaw(resultHandle));
#else
	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	for(int i = 0; i < vectorSize; ++i) {
		*result++ = ExponentFunc(*first++);
	}
#endif
}

void CCpuMathEngine::VectorLog(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

#ifdef NEOML_USE_MKL
	CFloatHandleStackVar minVal( mathEngine(), 1 );
	minVal.SetValue( FLT_MIN );
	CFloatHandleStackVar maxVal( mathEngine(), 1 );
	maxVal.SetValue( FLT_MAX );

	VectorMinMax(firstHandle, resultHandle, vectorSize, minVal, maxVal);
	vsLn(vectorSize, GetRaw(resultHandle), GetRaw(resultHandle));
#else
	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	for(int i = 0; i < vectorSize; ++i) {
		*result++ = logf(min(max(*first, FLT_MIN), FLT_MAX));
		first++;
	}
#endif
}

void CCpuMathEngine::VectorMultiplyAndAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	float mult = *GetRaw(multHandle);

#ifdef NEOML_USE_MKL
	if(first != result) {
		VectorCopy(resultHandle, firstHandle, vectorSize);
	}
	cblas_saxpy(vectorSize, mult, second, 1, result, 1);
#else
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		__m128 multSse = _mm_set_ps1(mult);
		for(int i = 0; i < sseSize; ++i) {
			_mm_storeu_ps(result, _mm_add_ps(_mm_loadu_ps(first), _mm_mul_ps(_mm_loadu_ps(second), multSse)));
			first += 4;
			second += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result++ = *first++ + *second++ * mult;
	}
#endif
}

void CCpuMathEngine::VectorTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

#ifdef NEOML_USE_MKL
	vsTanh(vectorSize, GetRaw(firstHandle), GetRaw(resultHandle));
#else
	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	for(int i = 0; i < vectorSize; ++i) {
		*result++ = -1.f + 2 / (1.f + ExponentFunc(-2 * *first++));
	}
#endif
}

void CCpuMathEngine::VectorPower(float exponent, const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

#ifdef NEOML_USE_MKL
	vsPowx(vectorSize, GetRaw(firstHandle), exponent, GetRaw(resultHandle));
#else
	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	for(int i = 0; i < vectorSize; ++i) {
		*result++ = powf(*first++, exponent);
	}
#endif
}

void CCpuMathEngine::VectorEltwiseLogSumExp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	CFloatHandleVar tempBuffer( mathEngine(), vectorSize );

	// Go through the vector and put the maximum into the result max, and -abs(first - second) into the tempBuffer
	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	float* temp = GetRaw(tempBuffer.GetHandle());

	const __m128 negateMask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
	for(int i = 0; i < sseSize; ++i) {
		__m128 val1 = LoadSse4(first);
		__m128 val2 = LoadSse4(second);

		__m128 maxVal = _mm_max_ps(val1, val2);
		__m128 negDiffVal = _mm_or_ps(negateMask, _mm_sub_ps(val1, val2));

		StoreSse4(maxVal, result);
		StoreSse4(negDiffVal, temp);

		first += 4;
		second += 4;
		result += 4;
		temp += 4;
	}
	if(nonSseSize > 0) {
		__m128 val1 = LoadSse(first, nonSseSize);
		__m128 val2 = LoadSse(second, nonSseSize);

		__m128 maxVal = _mm_max_ps(val1, val2);
		__m128 negDiffVal = _mm_or_ps(negateMask, _mm_sub_ps(val1, val2));

		StoreSse(maxVal, result, nonSseSize);
		StoreSse(negDiffVal, temp, nonSseSize);
	}

	temp = GetRaw(tempBuffer.GetHandle());

	VectorExp(tempBuffer.GetHandle(), tempBuffer.GetHandle(), vectorSize);
#ifdef NEOML_USE_MKL
	vsLog1p(vectorSize, temp, temp);
#else
	for( int i = 0; i < vectorSize; ++i ) {
		*temp = logf( min( 1.f + max( *temp, FLT_MIN ), FLT_MAX ) );
		temp++;
	}
#endif

	VectorAdd(resultHandle, tempBuffer.GetHandle(), resultHandle, vectorSize);
}

} // namespace NeoML

#endif // NEOML_USE_SSE
