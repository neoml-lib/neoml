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

#ifdef NEOML_USE_NEON

#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <CpuArm.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <CpuArmMathEngineVectorMathPrivate.h>

namespace NeoML {

void CCpuMathEngine::VectorFill(const CFloatHandle& result, float value, int vectorSize)
{
	ASSERT_EXPR( result.GetMathEngine() == this );
	CCpuExecutionScope scope;

	vectorFill( GetRaw( result ), value, vectorSize );
}

void CCpuMathEngine::VectorFill(const CIntHandle& result, int value, int vectorSize)
{
	ASSERT_EXPR( result.GetMathEngine() == this );
	CCpuExecutionScope scope;

	vectorFill( GetRaw( result ), value, vectorSize );
}

void CCpuMathEngine::VectorConvert( const CConstFloatHandle& from, const CIntHandle& to, int vectorSize )
{
	ASSERT_EXPR( from.GetMathEngine() == this );
	ASSERT_EXPR( to.GetMathEngine() == this );
	ASSERT_EXPR( vectorSize >= 0 );
	CCpuExecutionScope scope;

	const float* fromPtr = GetRaw( from );
	int* toPtr = GetRaw( to );

	int count = GetCount4( vectorSize );

	for( int i = 0; i < count; ++i ) {
		StoreIntNeon4( vcvtq_s32_f32( LoadNeon4( fromPtr ) ), toPtr );
		toPtr += 4;
		fromPtr += 4;
	}

	if( vectorSize > 0 ) {
		StoreIntNeon( vcvtq_s32_f32( LoadNeon( fromPtr, vectorSize ) ), toPtr, vectorSize );
	}
}

void CCpuMathEngine::VectorConvert( const CConstIntHandle& from, const CFloatHandle& to, int vectorSize )
{
	ASSERT_EXPR( from.GetMathEngine() == this );
	ASSERT_EXPR( to.GetMathEngine() == this );
	ASSERT_EXPR( vectorSize >= 0 );
	CCpuExecutionScope scope;

	const int* fromPtr = GetRaw( from );
	float* toPtr = GetRaw( to );

	int count = GetCount4( vectorSize );

	for( int i = 0; i < count; ++i ) {
		StoreNeon4( vcvtq_f32_s32( LoadIntNeon4( fromPtr ) ), toPtr );
		toPtr += 4;
		fromPtr += 4;
	}

	if( vectorSize > 0 ) {
		StoreNeon( vcvtq_f32_s32( LoadIntNeon( fromPtr, vectorSize ) ), toPtr, vectorSize );
	}
}

void CCpuMathEngine::VectorSumAdd(const CConstFloatHandle& firstHandle, int vectorSize,
	const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	int count = GetCount4(vectorSize);

	float32x4_t sum = vdupq_n_f32(0);

	for(int i = 0; i < count; ++i) {
		sum = vaddq_f32(sum, LoadNeon4(first));
		first += 4;
	}

	if(vectorSize > 0) {
		sum = vaddq_f32(sum, LoadNeon(first, vectorSize, 0));
	}

	float32x2_t sum2 = vpadd_f32(vget_high_f32(sum), vget_low_f32(sum));
	float32x2_t res = vpadd_f32(sum2, sum2);

	*GetRaw(resultHandle) += vget_lane_f32(res, 0);
}

void CCpuMathEngine::VectorReLU(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( upperThresholdHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	float threshold = *GetRaw(upperThresholdHandle);

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int index;
		int count;
		if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
			if( threshold > 0 ) {
				vectorReLU( first + index, result + index, count, threshold );
			} else {
				vectorReLU( first + index, result + index, count );
			}
		}
	}
}

void CCpuMathEngine::VectorEltwiseMax(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	vectorEltwiseMax( first, second, result, vectorSize );
}

void CCpuMathEngine::VectorEltwiseMin(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vminq_f32(LoadNeon4(first), LoadNeon4(second));
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vminq_f32(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize));
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorAbs(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	for(int i = 0; i < count; ++i) {
		float32x4_t val = LoadNeon4(first);
		float32x4_t res = vabsq_f32(val);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t val = LoadNeon(first, vectorSize);
		float32x4_t res = vabsq_f32(val);
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorHinge(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const float32x4_t zero = vdupq_n_f32(0);
	const float32x4_t one = vdupq_n_f32(1);

	for(int i = 0; i < count; ++i) {
		float32x4_t val = LoadNeon4(first);
		float32x4_t res = vmaxq_f32(zero, vsubq_f32(one, val));
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t val = LoadNeon(first, vectorSize);
		float32x4_t res = vmaxq_f32(zero, vsubq_f32(one, val));
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorHuberDerivative(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	CCpuExecutionScope scope;
	VectorHardTanh(firstHandle, resultHandle, vectorSize);
}

void CCpuMathEngine::VectorHardTanh(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	CFloatHandleStackVar minVal( mathEngine() );
	minVal.SetValue( -1 );

	CFloatHandleStackVar maxVal( mathEngine() );
	maxVal.SetValue( 1 );

	VectorMinMax(firstHandle, resultHandle, vectorSize, minVal, maxVal);
}

void CCpuMathEngine::VectorExp(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const CExpNeon expObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t val = LoadNeon4(first);
		float32x4_t res = expObj.Execute(val);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t val = LoadNeon(first, vectorSize);
		float32x4_t res = expObj.Execute(val);
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorLog(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const CLogNeon logObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t val = LoadNeon4(first);
		float32x4_t res = logObj.Execute(val);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t val = LoadNeon(first, vectorSize);
		float32x4_t res = logObj.Execute(val);
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorNegLog(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const CLogNeon logObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t val = LoadNeon4(first);
		float32x4_t res = vnegq_f32(logObj.Execute(val));
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t val = LoadNeon(first, vectorSize);
		float32x4_t res = vnegq_f32(logObj.Execute(val));
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorAdd(const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw(firstHandle);
	const int* second = GetRaw(secondHandle);
	int* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	for(int i = 0; i < count; ++i) {
		int32x4_t res = vaddq_s32(LoadIntNeon4(first), LoadIntNeon4(second));
		StoreIntNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		int32x4_t res = vaddq_s32(LoadIntNeon(first, vectorSize), LoadIntNeon(second, vectorSize));
		StoreIntNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorAddValue(const CConstIntHandle& firstHandle,
	const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& additionHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( additionHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw(firstHandle);
	int* result = GetRaw(resultHandle);
	int addition = *GetRaw(additionHandle);

	int count4 = GetCount4(vectorSize);

	int32x4_t addNeon = vdupq_n_s32(addition);
	for(int i = 0; i < count4; ++i) {
		StoreIntNeon4(vaddq_s32(LoadIntNeon4(first), addNeon), result);
		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		StoreIntNeon(vaddq_s32(LoadIntNeon(first, vectorSize), addNeon), result, vectorSize);
	}
}

void CCpuMathEngine::VectorSub(const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw(firstHandle);
	const int* second = GetRaw(secondHandle);
	int* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	for(int i = 0; i < count; ++i) {
		int32x4_t res = vsubq_s32(LoadIntNeon4(first), LoadIntNeon4(second));
		StoreIntNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		int32x4_t res = vsubq_s32(LoadIntNeon(first, vectorSize), LoadIntNeon(second, vectorSize));
		StoreIntNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorSub(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vsubq_f32(LoadNeon4(first), LoadNeon4(second));
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vsubq_f32(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize));
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorSub(float firstValue, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	float32x4_t first = vdupq_n_f32(firstValue);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vsubq_f32(first, LoadNeon4(second));
		StoreNeon4(res, result);

		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vsubq_f32(first, LoadNeon(second, vectorSize));
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorSub(const CConstFloatHandle& firstHandle, float secondValue,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float32x4_t second = vdupq_n_f32(secondValue);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vsubq_f32(LoadNeon4(first), second);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vsubq_f32(LoadNeon(first, vectorSize), second);
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorMultiplyAndAdd(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& multHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	float32_t mult = *GetRaw(multHandle);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vmlaq_n_f32(LoadNeon4(first), LoadNeon4(second), mult);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vmlaq_n_f32(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize), mult);
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorMultiplyAndSub(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& multHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	float32_t mult = *GetRaw(multHandle);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vmlsq_n_f32(LoadNeon4(first), LoadNeon4(second), mult);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vmlsq_n_f32(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize), mult);
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorNegMultiply(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	CFloatHandleStackVar mult( mathEngine() );
	mult.SetValue( -*GetRaw(multHandle) );

	VectorMultiply(firstHandle, resultHandle, vectorSize, mult);
}

void CCpuMathEngine::VectorEltwiseNegMultiply(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vnegq_f32(vmulq_f32(LoadNeon4(first), LoadNeon4(second)));
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vnegq_f32(vmulq_f32(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize)));
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorEltwiseDivide(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = DivideNeon(LoadNeon4(first), LoadNeon4(second));
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		// set default to 1 for right to work correctly with FPRecipEstimate
		float32x4_t res = DivideNeon(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize, 1));
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorSqrt(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	CSqrtNeon sqrtObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t res = sqrtObj.Execute(LoadNeon4(first));
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = sqrtObj.Execute(LoadNeon(first, vectorSize));
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t filterSmallValuesWorker(const float32x4_t& val, const float32x4_t& thres)
{
	uint32x4_t nonZeroMask = vcageq_f32(val, thres);
	uint32x4_t result = vandq_u32(vreinterpretq_u32_f32(val), nonZeroMask);

	return vreinterpretq_f32_u32(result);
}

void CCpuMathEngine::FilterSmallValues(const CFloatHandle& data, int dataSize, float threshold)
{
	ASSERT_EXPR( data.GetMathEngine() == this );

	float* dataPtr = GetRaw(data);
	int count = GetCount4(dataSize);

	float32x4_t thres = vdupq_n_f32(threshold);

	for(int i = 0; i < count; ++i) {
		float32x4_t val = LoadNeon4(dataPtr);
		float32x4_t res = filterSmallValuesWorker(val, thres);
		StoreNeon4(res, dataPtr);

		dataPtr += 4;
	}

	if(dataSize > 0) {
		float32x4_t val = LoadNeon(dataPtr, dataSize);
		float32x4_t res = filterSmallValuesWorker(val, thres);
		StoreNeon(res, dataPtr, dataSize);
	}
}

void CCpuMathEngine::VectorEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw( firstHandle );
	const int* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );

	const float32x4_t zero = vdupq_n_f32( 0 );
	const float32x4_t one = vdupq_n_f32( 1 );
	const int count = GetCount4( vectorSize );

	for( int i = 0; i < count; ++i ) {
		int32x4_t fi = LoadIntNeon4( first );
		int32x4_t se = LoadIntNeon4( second );
		uint32x4_t mask = vceqq_s32( fi, se );
		float32x4_t res = ConditionNeon( mask, one, zero );
		StoreNeon4( res, result );

		first += 4;
		second += 4;
		result += 4;
	}

	if( vectorSize > 0 ) {
		int32x4_t fi = LoadIntNeon( first, vectorSize );
		int32x4_t se = LoadIntNeon( second, vectorSize );
		uint32x4_t mask = vceqq_s32( fi, se );
		float32x4_t res = ConditionNeon( mask, one, zero );
		StoreNeon( res, result, vectorSize );
	}
}

void CCpuMathEngine::VectorEqualValue( const CConstIntHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstIntHandle& valueHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( valueHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );

	const float32x4_t zero = vdupq_n_f32( 0 );
	const float32x4_t one = vdupq_n_f32( 1 );
	const int32x4_t value = vdupq_n_s32( *GetRaw( valueHandle ) );

	const int count = GetCount4( vectorSize );

	for( int i = 0; i < count; ++i ) {
		int32x4_t fi = LoadIntNeon4( first );
		uint32x4_t mask = vceqq_s32( fi, value );
		float32x4_t res = ConditionNeon( mask, one, zero );
		StoreNeon4( res, result );

		first += 4;
		result += 4;
	}

	if( vectorSize > 0 ) {
		int32x4_t fi = LoadIntNeon( first, vectorSize );
		uint32x4_t mask = vceqq_s32( fi, value );
		float32x4_t res = ConditionNeon( mask, one, zero );
		StoreNeon( res, result, vectorSize );
	}
}

static inline float32x4_t vectorELUWorker(const float32x4_t& val,
	const float32x4_t& alpha, const float32x4_t& zero, const float32x4_t& one, const CExpNeon& expObj)
{
	uint32x4_t upperMask = vcgeq_f32(val, zero);
	float32x4_t lowerRes = vmulq_f32(alpha, vsubq_f32(expObj.Execute(val), one));
	return ConditionNeon(upperMask, val, lowerRes);
}

void CCpuMathEngine::VectorELU(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& alphaHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( alphaHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	float32x4_t alpha = vdupq_n_f32(*GetRaw(alphaHandle));
	const float32x4_t zero = vdupq_n_f32(0);
	const float32x4_t one = vdupq_n_f32(1);
	const CExpNeon expObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t val = LoadNeon4(first);
		float32x4_t res = vectorELUWorker(val, alpha, zero, one, expObj);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t val = LoadNeon(first, vectorSize);
		float32x4_t res = vectorELUWorker(val, alpha, zero, one, expObj);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorELUDiffWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& alpha, const float32x4_t& zero, const CExpNeon& expObj)
{
	uint32x4_t upperMask = vcgeq_f32(first, zero);
	float32x4_t lowerRes = vmulq_f32(second, vmulq_f32(alpha, expObj.Execute(first)));
	return ConditionNeon(upperMask, second, lowerRes);
}

void CCpuMathEngine::VectorELUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alphaHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( alphaHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	float32x4_t alpha = vdupq_n_f32(*GetRaw(alphaHandle));
	const float32x4_t zero = vdupq_n_f32(0);
	const CExpNeon expObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorELUDiffWorker(LoadNeon4(first), LoadNeon4(second), alpha, zero, expObj);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorELUDiffWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize),
			alpha, zero, expObj);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorELUDiffOpWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& alpha, const float32x4_t& zero)
{
	uint32x4_t upperMask = vcgeq_f32(first, zero);
	float32x4_t lowerRes = vmulq_f32(second, vaddq_f32(first, alpha));
	return ConditionNeon(upperMask, second, lowerRes);
}

void CCpuMathEngine::VectorELUDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alphaHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( alphaHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	float32x4_t alpha = vdupq_n_f32(*GetRaw(alphaHandle));
	const float32x4_t zero = vdupq_n_f32(0);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorELUDiffOpWorker(LoadNeon4(first), LoadNeon4(second), alpha, zero);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorELUDiffOpWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize),
			alpha, zero);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorReLUDiffWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& zero)
{
	uint32x4_t mask = vcgtq_f32(first, zero);
	return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(second), mask));
}

static inline float32x4_t vectorReLUDiffWorkerWithThreshold(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& zero, const float32x4_t& threshold)
{
	uint32x4_t mask = vandq_u32(vcgtq_f32(first, zero), vcltq_f32(first, threshold));
	return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(second), mask));
}

void CCpuMathEngine::VectorReLUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( upperThresholdHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);
	float threshold = *GetRaw(upperThresholdHandle);

	const float32x4_t zero = vdupq_n_f32(0);

	if(threshold > 0) {
		const float32x4_t thresholdNeon = vdupq_n_f32(threshold);
		for(int i = 0; i < count; ++i) {
			float32x4_t res = vectorReLUDiffWorkerWithThreshold(LoadNeon4(first),
				LoadNeon4(second), zero, thresholdNeon);
			StoreNeon4(res, result);

			first += 4;
			second += 4;
			result += 4;
		}

		if(vectorSize > 0) {
			float32x4_t res = vectorReLUDiffWorkerWithThreshold(LoadNeon(first, vectorSize),
				LoadNeon(second, vectorSize), zero, thresholdNeon);
			StoreNeon(res, result, vectorSize);
		}
	} else {
		for(int i = 0; i < count; ++i) {
			float32x4_t res = vectorReLUDiffWorker(LoadNeon4(first), LoadNeon4(second), zero);
			StoreNeon4(res, result);

			first += 4;
			second += 4;
			result += 4;
		}

		if(vectorSize > 0) {
			float32x4_t res = vectorReLUDiffWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize), zero);
			StoreNeon(res, result, vectorSize);
		}
	}
}

static inline float32x4_t VectorLeakyReLUWorker(const float32x4_t& val,
	const float32x4_t& alpha, const float32x4_t& zero)
{
	uint32x4_t upperMask = vcgeq_f32(val, zero);
	float32x4_t lowerRes = vmulq_f32(alpha, val);
	return ConditionNeon(upperMask, val, lowerRes);
}

void CCpuMathEngine::VectorLeakyReLU(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& alphaHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( alphaHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	float32x4_t alpha = vdupq_n_f32(*GetRaw(alphaHandle));
	const float32x4_t zero = vdupq_n_f32(0);

	for(int i = 0; i < count; ++i) {
		float32x4_t val = LoadNeon4(first);
		float32x4_t res = VectorLeakyReLUWorker(val, alpha, zero);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t val = LoadNeon(first, vectorSize);
		float32x4_t res = VectorLeakyReLUWorker(val, alpha, zero);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorLeakyReLUDiffWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& alpha, const float32x4_t& zero)
{
	uint32x4_t upperMask = vcgtq_f32(first, zero);
	float32x4_t lowerRes = vmulq_f32(alpha, second);
	return ConditionNeon(upperMask, second, lowerRes);
}

void CCpuMathEngine::VectorLeakyReLUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alphaHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( alphaHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	float32x4_t alpha = vdupq_n_f32(*GetRaw(alphaHandle));
	const float32x4_t zero = vdupq_n_f32(0);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorLeakyReLUDiffWorker(LoadNeon4(first), LoadNeon4(second), alpha, zero);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorLeakyReLUDiffWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize),
			alpha, zero);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorHSwishWorker( const float32x4_t& first, const float32x4_t& three,
	const float32x4_t& minusThree, const float32x4_t& oneSixth )
{
	uint32x4_t middleMask = vandq_u32( vcgtq_f32( first, minusThree ), vcltq_f32( first, three ) );
	float32x4_t middleValue = vmulq_f32( vaddq_f32( first, three ), vmulq_f32( first, oneSixth ) );
	middleValue = vreinterpretq_f32_u32( vandq_u32( vreinterpretq_u32_f32( middleValue ), middleMask ) );
	float32x4_t rightValue = vandq_u32( vreinterpretq_u32_f32( first ), vcgeq_f32( first, three ) );
	return vaddq_f32( middleValue, rightValue );
}

void CCpuMathEngine::VectorHSwish( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );
	int count = GetCount4( vectorSize );

	const float32x4_t three = vdupq_n_f32( 3 );
	const float32x4_t minusThree = vdupq_n_f32( -3 );
	const float32x4_t oneSixth = vdupq_n_f32( 1.f / 6 );

	for( int i = 0; i < count; ++i ) {
		float32x4_t res = vectorHSwishWorker( LoadNeon4( first ), three, minusThree, oneSixth );
		StoreNeon4( res, result );

		first += 4;
		result += 4;
	}

	if( vectorSize > 0 ) {
		float32x4_t res = vectorHSwishWorker( LoadNeon( first, vectorSize ), three, minusThree, oneSixth );
		StoreNeon( res, result, vectorSize );
	}
}
static inline float32x4_t vectorHSwishDiffWorker( const float32x4_t& first, const float32x4_t& second, const float32x4_t& three,
	const float32x4_t& minusThree, const float32x4_t& oneThird, const float32x4_t& half)
{
	uint32x4_t middleMask = vandq_u32( vcgtq_f32( first, minusThree ), vcltq_f32( first, three ) );
	float32x4_t middleValue = vmulq_f32( vaddq_f32( vmulq_f32( first, oneThird ), half ), second );
	middleValue = vreinterpretq_f32_u32( vandq_u32( vreinterpretq_u32_f32( middleValue ), middleMask ) );
	float32x4_t rightValue = vandq_u32( vreinterpretq_u32_f32( second ), vcgeq_f32( first, three ) );
	return vaddq_f32( middleValue, rightValue );
}

void CCpuMathEngine::VectorHSwishDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );
	int count = GetCount4( vectorSize );

	const float32x4_t three = vdupq_n_f32( 3 );
	const float32x4_t minusThree = vdupq_n_f32( -3 );
	const float32x4_t oneThird = vdupq_n_f32( 1.f / 3 );
	const float32x4_t half = vdupq_n_f32( 0.5f );

	for( int i = 0; i < count; ++i ) {
		float32x4_t res = vectorHSwishDiffWorker( LoadNeon4( first ), LoadNeon4( second ), three, minusThree, oneThird, half );
		StoreNeon4( res, result );

		first += 4;
		second += 4;
		result += 4;
	}

	if( vectorSize > 0 ) {
		float32x4_t res = vectorHSwishDiffWorker( LoadNeon( first, vectorSize ), LoadNeon( second, vectorSize ), three, minusThree, oneThird, half );
		StoreNeon( res, result, vectorSize );
	}
}

static inline float32x4_t vectorAbsDiffWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& zero)
{
	return ConditionNeon(vcgtq_f32(first, zero), second, vnegq_f32(second));
}

void CCpuMathEngine::VectorAbsDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const float32x4_t zero = vdupq_n_f32(0);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorAbsDiffWorker(LoadNeon4(first), LoadNeon4(second), zero);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorAbsDiffWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize), zero);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorHingeDiffWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& one)
{
	return vreinterpretq_f32_u32(vandq_u32(vcltq_f32(first, one), vreinterpretq_u32_f32(vnegq_f32(second))));
}

void CCpuMathEngine::VectorHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const float32x4_t one = vdupq_n_f32(1);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorHingeDiffWorker(LoadNeon4(first), LoadNeon4(second), one);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorHingeDiffWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize), one);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorSquaredHingeWorker(const float32x4_t& val,
	const float32x4_t& zero, const float32x4_t& one, const float32x4_t& neg1, const float32x4_t& neg4)
{
	uint32x4_t lowerMask = vcltq_f32(val, neg1);
	float32x4_t hinge = vmaxq_f32(zero, vsubq_f32(one, val));
	return ConditionNeon(lowerMask, vmulq_f32(neg4, val), vmulq_f32(hinge, hinge));
}

void CCpuMathEngine::VectorSquaredHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const float32x4_t zero = vdupq_n_f32(0);
	const float32x4_t one = vdupq_n_f32(1);
	const float32x4_t neg1 = vdupq_n_f32(-1);
	const float32x4_t neg4 = vdupq_n_f32(-4);

	for(int i = 0; i < count; ++i) {
		float32x4_t val = LoadNeon4(first);
		float32x4_t res = vectorSquaredHingeWorker(val, zero, one, neg1, neg4);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t val = LoadNeon(first, vectorSize);
		float32x4_t res = vectorSquaredHingeWorker(val, zero, one, neg1, neg4);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorSquaredHingeDiffWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& zero, const float32x4_t& one,
	const float32x4_t& neg1, const float32x4_t& neg2, const float32x4_t& neg4)
{
	uint32x4_t lowerMask = vcltq_f32(first, neg1);
	float32x4_t mainDerivative = vmulq_f32(neg2, vmaxq_f32(zero, vsubq_f32(one, first)));
	float32x4_t derivative = ConditionNeon(lowerMask, neg4, mainDerivative);
	return vmulq_f32(derivative, second);
}

void CCpuMathEngine::VectorSquaredHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const float32x4_t zero = vdupq_n_f32(0);
	const float32x4_t one = vdupq_n_f32(1);
	const float32x4_t neg1 = vdupq_n_f32(-1);
	const float32x4_t neg2 = vdupq_n_f32(-2);
	const float32x4_t neg4 = vdupq_n_f32(-4);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorSquaredHingeDiffWorker(LoadNeon4(first), LoadNeon4(second),
			zero, one, neg1, neg2, neg4);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorSquaredHingeDiffWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize),
			zero, one, neg1, neg2, neg4);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorHuberWorker(const float32x4_t& val,
	const float32x4_t& half, const float32x4_t& one, const float32x4_t& neg1)
{
	uint32x4_t lowerMask = vcltq_f32(val, neg1);
	uint32x4_t upperMask = vcgtq_f32(val, one);

	float32x4_t lowerVal = vnegq_f32(vaddq_f32(val, half));
	float32x4_t upperVal = vsubq_f32(val, half);
	float32x4_t mainVal = vmulq_f32(half, vmulq_f32(val, val));

	return Condition2Neon(lowerMask, upperMask, lowerVal, upperVal, mainVal);
}

void CCpuMathEngine::VectorHuber(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const float32x4_t half = vdupq_n_f32(0.5);
	const float32x4_t one = vdupq_n_f32(1);
	const float32x4_t neg1 = vdupq_n_f32(-1);

	for(int i = 0; i < count; ++i) {
		float32x4_t val = LoadNeon4(first);
		float32x4_t res = vectorHuberWorker(val, half, one, neg1);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t val = LoadNeon(first, vectorSize);
		float32x4_t res = vectorHuberWorker(val, half, one, neg1);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorHardTanhDiffWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& one, const float32x4_t& neg1)
{
	uint32x4_t mask = vandq_u32(vcgtq_f32(first, neg1), vcltq_f32(first, one));
	return vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(second)));
}

void CCpuMathEngine::VectorHardTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const float32x4_t one = vdupq_n_f32(1);
	const float32x4_t neg1 = vdupq_n_f32(-1);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorHardTanhDiffWorker(LoadNeon4(first), LoadNeon4(second), one, neg1);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorHardTanhDiffWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize),
			one, neg1);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorHardSigmoidWorker( const float32x4_t& val,
	const float32x4_t& zero, const float32x4_t& one, const float32x4_t& slope, const float32x4_t& bias )
{
	float32x4_t mainVal = vaddq_f32( vmulq_f32( val, slope ), bias );
	return vmaxq_f32( zero, vminq_f32( one, mainVal ) );
}

void CCpuMathEngine::VectorHardSigmoid( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );
	int count = GetCount4( vectorSize );

	const float slope = *GetRaw( slopeHandle );
	const float bias = *GetRaw( biasHandle );

	ASSERT_EXPR( slope != 0.f );

	const float32x4_t zero = vdupq_n_f32( 0 );
	const float32x4_t one = vdupq_n_f32( 1 );
	const float32x4_t slope4 = vdupq_n_f32( slope );
	const float32x4_t bias4 = vdupq_n_f32( bias );

	for( int i = 0; i < count; ++i ) {
		float32x4_t val = LoadNeon4( first );
		float32x4_t res = vectorHardSigmoidWorker( val, zero, one, slope4, bias4 );
		StoreNeon4( res, result );

		first += 4;
		result += 4;
	}

	if( vectorSize > 0 ) {
		float32x4_t val = LoadNeon( first, vectorSize );
		float32x4_t res = vectorHardSigmoidWorker( val, zero, one, slope4, bias4 );
		StoreNeon( res, result, vectorSize );
	}
}

static inline float32x4_t vectorHardSigmoidDiffWorker( const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& slope, const float32x4_t& minX, const float32x4_t& maxX )
{
	uint32x4_t mask = vandq_u32( vcgtq_f32( first, minX ), vcltq_f32( first, maxX ) );
	float32x4_t maskedSecond = vreinterpretq_f32_u32( vandq_u32( mask, vreinterpretq_u32_f32( second ) ) );
	return vmulq_f32( maskedSecond, slope );
}

void CCpuMathEngine::VectorHardSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);
	const float slope = *GetRaw( slopeHandle );
	const float bias = *GetRaw( biasHandle );

	ASSERT_EXPR( slope != 0.f );

	const float32x4_t slope4 = vdupq_n_f32( slope );
	const float32x4_t minX = vdupq_n_f32( -bias / slope );
	const float32x4_t maxX = vdupq_n_f32( ( 1.f - bias ) / slope );

	for( int i = 0; i < count; ++i ) {
		float32x4_t res = vectorHardSigmoidDiffWorker( LoadNeon4( first ), LoadNeon4( second ), slope4, minX, maxX );
		StoreNeon4( res, result );

		first += 4;
		second += 4;
		result += 4;
	}

	if( vectorSize > 0 ) {
		float32x4_t res = vectorHardSigmoidDiffWorker( LoadNeon( first, vectorSize ), LoadNeon( second, vectorSize ),
			slope4, minX, maxX );
		StoreNeon( res, result, vectorSize );
	}
}

static inline float32x4_t vectorHardSigmoidDiffOpWorker( const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& zero, const float32x4_t& one, const float32x4_t& slope )
{
	uint32x4_t mask = vandq_u32( vcgtq_f32( first, zero ), vcltq_f32( first, one ) );
	float32x4_t maskedSecond = vreinterpretq_f32_u32( vandq_u32( mask, vreinterpretq_u32_f32( second ) ) );
	return vmulq_f32( maskedSecond, slope );
}

void CCpuMathEngine::VectorHardSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& /*biasHandle*/ )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);
	const float slope = *GetRaw( slopeHandle );

	ASSERT_EXPR( slope != 0.f );

	const float32x4_t zero = vdupq_n_f32( 0 );
	const float32x4_t one = vdupq_n_f32( 1 );
	const float32x4_t slope4 = vdupq_n_f32( slope );

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorHardSigmoidDiffOpWorker( LoadNeon4(first), LoadNeon4(second), zero, one, slope4 );
		StoreNeon4( res, result );

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorHardSigmoidDiffOpWorker( LoadNeon( first, vectorSize ), LoadNeon( second, vectorSize ),
			zero, one, slope4 );
		StoreNeon( res, result, vectorSize );
	}
}

static inline float32x4_t vectorBernulliKLDerivativeWorker(const float32x4_t& val,
	const float32x4_t& target, const float32x4_t& minVal, const float32x4_t& maxVal)
{
	float32x4_t res = DivideNeon(vsubq_f32(val, target), vsubq_f32(val, vmulq_f32(val, val)));
	return vmaxq_f32(minVal, vminq_f32(maxVal, res));
}

void CCpuMathEngine::VectorBernulliKLDerivative(const CConstFloatHandle& estimationHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& targetHandle)
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( estimationHandle.GetMathEngine() == this );
	ASSERT_EXPR( targetHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(estimationHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);
	const float MaxKLDerivative = 10;

	float32x4_t target = vdupq_n_f32(*GetRaw(targetHandle));
	const float32x4_t minVal = vdupq_n_f32(-MaxKLDerivative);
	const float32x4_t maxVal = vdupq_n_f32(MaxKLDerivative);

	for(int i = 0; i < count; ++i) {
		float32x4_t val = LoadNeon4(first);
		float32x4_t res = vectorBernulliKLDerivativeWorker(val, target, minVal, maxVal);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t val = LoadNeon(first, vectorSize);
		float32x4_t res = vectorBernulliKLDerivativeWorker(val, target, minVal, maxVal);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorEltwisePowerWorker(const float32x4_t& first, const float32x4_t& second,
	const CLogNeon& logObj, const CExpNeon& expObj)
{
	return expObj.Execute(vmulq_f32(logObj.Execute(first), second));
}

void CCpuMathEngine::VectorEltwisePower(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const CLogNeon logObj;
	const CExpNeon expObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorEltwisePowerWorker(LoadNeon4(first), LoadNeon4(second), logObj, expObj);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorEltwisePowerWorker(LoadNeon(first, vectorSize),
			LoadNeon(second, vectorSize), logObj, expObj);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorInvWorker(const float32x4_t& val,
	const float32x4_t& zero, const float32x4_t& minVal, const float32x4_t& negMaxVal, const float32x4_t& maxVal)
{
	uint32x4_t smallMask = vcaltq_f32(val, minVal);
	uint32x4_t negMask = vcltq_f32(val, zero);
	float32x4_t res = InvNeon(ConditionNeon(smallMask, minVal, val));
	return Condition2Neon(vandq_u32(smallMask, negMask), vbicq_u32(smallMask, negMask), negMaxVal, maxVal, res);
}

void CCpuMathEngine::VectorInv(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const float32x4_t zero = vdupq_n_f32(0);
	const float32x4_t minVal = vdupq_n_f32(FLT_MIN);
	const float32x4_t negMaxVal = vdupq_n_f32(-FLT_MAX);
	const float32x4_t maxVal = vdupq_n_f32(FLT_MAX);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorInvWorker(LoadNeon4(first), zero, minVal, negMaxVal, maxVal);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorInvWorker(LoadNeon(first, vectorSize), zero, minVal, negMaxVal, maxVal);
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorSigmoid(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	vectorSigmoid( first, result, vectorSize );
}

static inline float32x4_t vectorSigmoidDiffWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& one, const CExpNeon& expObj)
{
	float32x4_t expNeg = expObj.Execute(vnegq_f32(first));
	float32x4_t denom = vaddq_f32(one, expNeg);
	
	return DivideNeon(vmulq_f32(second, expNeg), vmulq_f32(denom, denom));
}

void CCpuMathEngine::VectorSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const float32x4_t one = vdupq_n_f32(1);
	const CExpNeon expObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorSigmoidDiffWorker(LoadNeon4(first), LoadNeon4(second), one, expObj);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorSigmoidDiffWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize),
			one, expObj);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorSigmoidDiffOpWorker(const float32x4_t& first, const float32x4_t& second)
{
	float32x4_t derivative = vsubq_f32(first, vmulq_f32(first, first));
	return vmulq_f32(second, derivative);
}

void CCpuMathEngine::VectorSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorSigmoidDiffOpWorker(LoadNeon4(first), LoadNeon4(second));
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorSigmoidDiffOpWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize));
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	
	vectorTanh( first, result, vectorSize );
}

static inline float32x4_t vectorTanhDiffWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& one, const CExpNeon& expObj)
{
	float32x4_t tanh = vectorTanhWorker(first, one, expObj);
	float32x4_t derivative = vsubq_f32(one, vmulq_f32(tanh, tanh));
	return vmulq_f32(second, derivative);
}

void CCpuMathEngine::VectorTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const float32x4_t one = vdupq_n_f32(1);
	const CExpNeon expObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorTanhDiffWorker(LoadNeon4(first), LoadNeon4(second), one, expObj);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorTanhDiffWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize),
			one, expObj);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorTanhDiffOpWorker(const float32x4_t& first, const float32x4_t& second)
{
	return vsubq_f32(second, vmulq_f32(second, vmulq_f32(first, first)));
}

void CCpuMathEngine::VectorTanhDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorTanhDiffOpWorker(LoadNeon4(first), LoadNeon4(second));
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorTanhDiffOpWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize));
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorPowerWorker(const float32x4_t& val, const float32x4_t& pow, const CLogNeon& logObj, const CExpNeon& expObj)
{
	return expObj.Execute(vmulq_f32(pow, logObj.Execute(val)));
}

void CCpuMathEngine::VectorPower(float exponent, const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	float32x4_t pow = vdupq_n_f32(exponent);
	const CLogNeon logObj;
	const CExpNeon expObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorPowerWorker(LoadNeon4(first), pow, logObj, expObj);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorPowerWorker(LoadNeon(first, vectorSize), pow, logObj, expObj);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorPowerDiffWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& pow, const float32x4_t& pow1, const CLogNeon& logObj, const CExpNeon& expObj)
{
	float32x4_t derivative = vmulq_f32(pow, vectorPowerWorker(first, pow1, logObj, expObj));
	return vmulq_f32(second, derivative);
}

void CCpuMathEngine::VectorPowerDiff(float exponent, const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	float32x4_t pow = vdupq_n_f32(exponent);
	float32x4_t pow1 = vdupq_n_f32(exponent - 1);
	const CLogNeon logObj;
	const CExpNeon expObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorPowerDiffWorker(LoadNeon4(first), LoadNeon4(second), pow, pow1, logObj, expObj);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorPowerDiffWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize),
			pow, pow1, logObj, expObj);
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorPowerDiffOp(float exponent, const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	float32x4_t pow = vdupq_n_f32(exponent);
	float32x4_t pow1 = DivideNeon(vdupq_n_f32(exponent - 1), pow);
	const CLogNeon logObj;
	const CExpNeon expObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorPowerDiffWorker(LoadNeon4(first), LoadNeon4(second), pow, pow1, logObj, expObj);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorPowerDiffWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize),
			pow, pow1, logObj, expObj);
		StoreNeon(res, result, vectorSize);
	}
}

static inline float32x4_t vectorL1DiffAddWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& huberThreshold, const float32x4_t& negHuberThreshold, const float32x4_t& mult)
{
	float32x4_t x = vmaxq_f32(negHuberThreshold, vminq_f32(huberThreshold, second));
	return vaddq_f32(first, vmulq_f32(mult, x));
}

void CCpuMathEngine::VectorL1DiffAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize,
	const CConstFloatHandle& huberThresholdHandle, const CConstFloatHandle& multHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( huberThresholdHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	float32x4_t huberThreshold = vdupq_n_f32(*GetRaw(huberThresholdHandle));
	float32x4_t negHuberThreshold = vnegq_f32(huberThreshold);
	float32x4_t mult = vdupq_n_f32(*GetRaw(multHandle));

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorL1DiffAddWorker(LoadNeon4(first), LoadNeon4(second),
			huberThreshold, negHuberThreshold, mult);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorL1DiffAddWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize),
			huberThreshold, negHuberThreshold, mult);
		StoreNeon(res, result, vectorSize);
	}
}

void CCpuMathEngine::VectorEltwiseNot( const CConstIntHandle& firstHandle, const CIntHandle& resultHandle,
	int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw( firstHandle );
	int* result = GetRaw( resultHandle );
	int count = GetCount4( vectorSize );

	const int32x4_t zeros = vdupq_n_s32( 0 );
	const int32x4_t ones = vdupq_n_s32( 1 );
	for( int i = 0; i < count; ++i ) {
		StoreIntNeon4( vandq_s32( ones, vceqq_s32( LoadIntNeon4( first ), zeros ) ), result );
		first += 4;
		result += 4;
	}

	if( vectorSize > 0 ) {
		StoreIntNeon( vandq_s32( ones, vceqq_s32( LoadIntNeon( first, vectorSize ), zeros ) ), result, vectorSize );
	}
}

void CCpuMathEngine::VectorEltwiseNotNegative( const CConstIntHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );
	int count = GetCount4( vectorSize );

	const float32x4_t one = vdupq_n_f32( 1 );
	const int32x4_t zero = vdupq_n_s32( 0 );
	for( int i = 0; i < count; ++i ) {
		float32x4_t res = vreinterpretq_f32_u32( vandq_u32( vreinterpretq_u32_f32( one ),
			vcgeq_s32( LoadIntNeon4( first ), zero ) ) );
		StoreNeon4( res, result );
		first += 4;
		result += 4;
	}

	if( vectorSize > 0 ) {
		float32x4_t res = vreinterpretq_f32_u32( vandq_u32( vreinterpretq_u32_f32( one ),
			vcgeq_s32( LoadIntNeon( first, vectorSize ), zero ) ) );
		StoreNeon( res, result, vectorSize );
	}
}

static inline float32x4_t vectorEltwiseLogSumExpWorker(const float32x4_t& first, const float32x4_t& second,
	const float32x4_t& one, const CExpNeon& expObj, const CLogNeon& logObj)
{
	float32x4_t maxVal = vmaxq_f32(first, second);
	float32x4_t minVal = vminq_f32(first, second);
	return vaddq_f32(maxVal, logObj.ExecuteNoCheck(vaddq_f32(one, expObj.Execute(vsubq_f32(minVal, maxVal)))));
}

void CCpuMathEngine::vectorEltwiseLogSumExp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	int count = GetCount4(vectorSize);

	const float32x4_t one = vdupq_n_f32(1);
	const CExpNeon expObj;
	const CLogNeon logObj;

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vectorEltwiseLogSumExpWorker(LoadNeon4(first), LoadNeon4(second),
			one, expObj, logObj);
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vectorEltwiseLogSumExpWorker(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize),
			one, expObj, logObj);
		StoreNeon(res, result, vectorSize);
	}
}

} // namespace NeoML

#endif // NEOML_USE_NEON
