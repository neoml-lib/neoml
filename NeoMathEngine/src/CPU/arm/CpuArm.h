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

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_NEON

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <arm_neon.h>
#include <float.h>

namespace NeoML {

// define for logarithms FLT_MIN/MAX.
#define FLT_MIN_LOG -87.33654474f
#define FLT_MAX_LOG 88.f

// Split the vector length into 4-element registers + remainder
inline int GetCount4(int& size)
{
	int ret = size / 4;
	size %= 4;
	return ret;
}

// Load and save
inline float32x4_t LoadNeon4(const float* data)
{
	return vld1q_f32(data);
}

inline float32x4_t LoadNeon(const float* data, int count, float defVal = 0)
{
	float32x4_t ret = vdupq_n_f32(defVal);
	if(count > 0) {
		ret = vld1q_lane_f32(data++, ret, 0);
		if(count > 1) {
			ret = vld1q_lane_f32(data++, ret, 1);
			if(count > 2) {
				ret = vld1q_lane_f32(data++, ret, 2);
			}
		}
	}
	return ret;
}

inline int32x4_t LoadIntNeon4(const int* data)
{
	return vld1q_s32(data);
}

inline int32x4_t LoadIntNeon(const int* data, int count, int defVal = 0)
{
	int32x4_t ret = vdupq_n_s32(defVal);
	if(count > 0) {
		ret = vld1q_lane_s32(data++, ret, 0);
		if(count > 1) {
			ret = vld1q_lane_s32(data++, ret, 1);
			if(count > 2) {
				ret = vld1q_lane_s32(data++, ret, 2);
			}
		}
	}
	return ret;
}

inline void StoreNeon4(const float32x4_t& val, float* data)
{
	return vst1q_f32(data, val);
}

inline void StoreNeon(const float32x4_t& val, float* data, int count)
{
	if(count > 0) {
		*data++ = vgetq_lane_f32(val, 0);
		if(count > 1) {
			*data++ = vgetq_lane_f32(val, 1);
			if(count > 2) {
				*data = vgetq_lane_f32(val, 2);
			}
		}
	}
}

inline void StoreIntNeon4(const int32x4_t& val, int* data)
{
	return vst1q_s32(data, val);
}

inline void StoreIntNeon(const int32x4_t& val, int* data, int count)
{
	if(count > 0) {
		*data++ = vgetq_lane_s32(val, 0);
		if(count > 1) {
			*data++ = vgetq_lane_s32(val, 1);
			if(count > 2) {
				*data = vgetq_lane_s32(val, 2);
			}
		}
	}
}

#define NEON_LOAD_16_FLOATS(varPrefix, src) \
    float32x4_t varPrefix##0 = LoadNeon4(src + 4 * 0); \
    float32x4_t varPrefix##1 = LoadNeon4(src + 4 * 1); \
    float32x4_t varPrefix##2 = LoadNeon4(src + 4 * 2); \
    float32x4_t varPrefix##3 = LoadNeon4(src + 4 * 3);

#define NEON_STORE_16_FLOATS(varPrefix, dst) \
    StoreNeon4(varPrefix##0, dst + 4 * 0); \
    StoreNeon4(varPrefix##1, dst + 4 * 1); \
    StoreNeon4(varPrefix##2, dst + 4 * 2); \
    StoreNeon4(varPrefix##3, dst + 4 * 3);

#define NEON_LOAD_16_INTS(varPrefix, src) \
    int32x4_t varPrefix##0 = LoadIntNeon4(src + 4 * 0); \
    int32x4_t varPrefix##1 = LoadIntNeon4(src + 4 * 1); \
    int32x4_t varPrefix##2 = LoadIntNeon4(src + 4 * 2); \
    int32x4_t varPrefix##3 = LoadIntNeon4(src + 4 * 3);

#define NEON_STORE_16_INTS(varPrefix, dst) \
    StoreIntNeon4(varPrefix##0, dst + 4 * 0); \
    StoreIntNeon4(varPrefix##1, dst + 4 * 1); \
    StoreIntNeon4(varPrefix##2, dst + 4 * 2); \
    StoreIntNeon4(varPrefix##3, dst + 4 * 3);

inline void dataCopy(float* dst, const float* src, int vectorSize)
{
	int count = GetCount4(vectorSize);

	for( ; count >= 4; count -= 4, src += 16, dst += 16 ) {
		StoreNeon4( LoadNeon4( src + 4 * 0 ), dst + 4 * 0);
		StoreNeon4( LoadNeon4( src + 4 * 1 ), dst + 4 * 1);
		StoreNeon4( LoadNeon4( src + 4 * 2 ), dst + 4 * 2);
		StoreNeon4( LoadNeon4( src + 4 * 3 ), dst + 4 * 3);
	}

	for( ; count > 0; --count, src += 4, dst += 4 ) {
		StoreNeon4(LoadNeon4(src), dst);
	}

	for(int i = 0; i < vectorSize; ++i) {
		*dst++ = *src++;
	}
}

inline void dataCopy(int* dst, const int* src, int vectorSize)
{
	int count = GetCount4(vectorSize);

	for( ; count >= 4; count -= 4, src += 16, dst += 16 ) {
		StoreIntNeon4( LoadIntNeon4( src + 4 * 0 ), dst + 4 * 0);
		StoreIntNeon4( LoadIntNeon4( src + 4 * 1 ), dst + 4 * 1);
		StoreIntNeon4( LoadIntNeon4( src + 4 * 2 ), dst + 4 * 2);
		StoreIntNeon4( LoadIntNeon4( src + 4 * 3 ), dst + 4 * 3);
	}

	for( ; count > 0; --count, src += 4, dst += 4 ) {
		StoreIntNeon4(LoadIntNeon4(src), dst);
	}

	for(int i = 0; i < vectorSize; ++i) {
		*dst++ = *src++;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// The templates for Load/Store
// The base class
class CBaseLoadStoreNeon : public CCrtAllocatedObject {
public:
	float32x4_t Load(const float* /*data*/)
	{
		ASSERT_EXPR(false);
		return float32x4_t();
	}

	int32x4_t LoadInt(const int* /*data*/)
	{
		ASSERT_EXPR(false);
		return int32x4_t();
	}

	void Store(const float32x4_t& /*val*/, float* /*data*/)
	{
		ASSERT_EXPR(0);
	}

	void StoreInt(const int32x4_t& /*val*/, int* /*data*/)
	{
		ASSERT_EXPR(0);
	}
};

// The aligned size class
class CLoadStoreNeon4 : public CCrtAllocatedObject {
public:
	float32x4_t Load(const float* data)
	{
		return LoadNeon4(data);
	}

	int32x4_t LoadInt(const int* data)
	{
		return LoadIntNeon4(data);
	}

	void Store(const float32x4_t& val, float* data)
	{
		StoreNeon4(val, data);
	}

	void StoreInt(const int32x4_t& val, int* data)
	{
		StoreIntNeon4(val, data);
	}
};

// The non-aligned size class
class CLoadStoreNeon : public CCrtAllocatedObject {
public:
	const int Len;
	const float DefVal;
	const int DefIntVal;

	explicit CLoadStoreNeon(int len, float defVal = 0, int defIntVal = 0) :
		Len(len), DefVal(defVal), DefIntVal(defIntVal)
	{
	}

	float32x4_t Load(const float* data)
	{
		return LoadNeon(data, Len, DefVal);
	}

	int32x4_t LoadInt(const int* data)
	{
		return LoadIntNeon(data, Len, DefIntVal);
	}

	void Store(const float32x4_t& val, float* data)
	{
		StoreNeon(val, data, Len);
	}

	void StoreInt(const int32x4_t& val, int* data)
	{
		StoreIntNeon(val, data, Len);
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

// Get the lane value
inline float GetLaneNeon(const float32x4_t& val, int lane)
{
	switch(lane) {
		default:
		case 0:
			return vgetq_lane_f32(val, 0);
		case 1:
			return vgetq_lane_f32(val, 1);
		case 2:
			return vgetq_lane_f32(val, 2);
		case 3:
			return vgetq_lane_f32(val, 3);
	}
}

// Set different values to the register
inline float32x4_t SetRegisterNeon(float v0, float v1, float v2, float v3)
{
	float32x4_t ret = vdupq_n_f32(v0);
	ret = vsetq_lane_f32(v1, ret, 1);
	ret = vsetq_lane_f32(v2, ret, 2);
	ret = vsetq_lane_f32(v3, ret, 3);
	return ret;
}

inline int32x4_t SetRegisterIntNeon(int v0, int v1, int v2, int v3)
{
	int32x4_t ret = vdupq_n_s32(v0);
	ret = vsetq_lane_s32(v1, ret, 1);
	ret = vsetq_lane_s32(v2, ret, 2);
	ret = vsetq_lane_s32(v3, ret, 3);
	return ret;
}

// condition
// Similar to the ?: ternary operator. The result is a combined value of condTrueVal 
// for the vector elements that fit the condMask and condFalseVal for the elements that don't fit the mask
inline float32x4_t ConditionNeon(const uint32x4_t& condMask,
	const float32x4_t& condTrueVal, const float32x4_t& condFalseVal)
{
	return vbslq_f32(condMask, condTrueVal, condFalseVal);
}
inline float32x2_t ConditionNeon(const uint32x2_t& condMask,
	const float32x2_t& condTrueVal, const float32x2_t& condFalseVal)
{
	return vbsl_f32(condMask, condTrueVal, condFalseVal);
}
inline int32x4_t ConditionIntNeon(const uint32x4_t& condMask,
	const int32x4_t& condTrueVal, const int32x4_t& condFalseVal)
{
	return vbslq_s32(condMask, condTrueVal, condFalseVal);
}
inline int32x2_t ConditionIntNeon(const uint32x2_t& condMask,
	const int32x2_t& condTrueVal, const int32x2_t& condFalseVal)
{
	return vbsl_s32(condMask, condTrueVal, condFalseVal);
}

// Combines the vectors using two masks. 
// The result is a combined value of condVal1 for the elements that fit the condMask0, 
// condVal1 for the elements that fit the condMask1, and condValNone for the rest. 
// condMask0 and condMask1 must not have intersections
inline float32x4_t Condition2Neon(const uint32x4_t& condMask0, const uint32x4_t& condMask1,
	const float32x4_t& condVal0, const float32x4_t& condVal1, const float32x4_t& condValNone)
{
	return vbslq_f32(condMask0, condVal0, vbslq_f32(condMask1, condVal1, condValNone));
}
inline float32x2_t Condition2Neon(const uint32x2_t& condMask0, const uint32x2_t& condMask1,
	const float32x2_t& condVal0, const float32x2_t& condVal1, const float32x2_t& condValNone)
{
	return vbsl_f32(condMask0, condVal0, vbsl_f32(condMask1, condVal1, condValNone));
}
inline int32x4_t Condition2IntNeon(const uint32x4_t& condMask0, const uint32x4_t& condMask1,
	const int32x4_t& condVal0, const int32x4_t& condVal1, const int32x4_t& condValNone)
{
	return vbslq_s32(condMask0, condVal0, vbslq_s32(condMask1, condVal1, condValNone));
}
inline int32x2_t Condition2IntNeon(const uint32x2_t& condMask0, const uint32x2_t& condMask1,
	const int32x2_t& condVal0, const int32x2_t& condVal1, const int32x2_t& condValNone)
{
	return vbsl_s32(condMask0, condVal0, vbsl_s32(condMask1, condVal1, condValNone));
}

// Gets the "horizontal sum" - the sum of all elements in the register. It is put into each register position
inline float32x2_t HorizontalAddNeon(const float32x4_t& val)
{
	float32x2_t sum01_23 = vpadd_f32(vget_low_f32(val), vget_high_f32(val));
	return vpadd_f32(sum01_23, sum01_23);
}

// Gets the maximum of elements in the register. It is put into each register position
inline float32x2_t HorizontalMaxNeon(const float32x4_t& val)
{
	float32x2_t max01_23 = vpmax_f32(vget_low_f32(val), vget_high_f32(val));
	return vpmax_f32(max01_23, max01_23);
}

// Gets the maximum of elements in the register. It is put into each register position
// The index of the found maximum is set to the indices register
inline void HorizontalMaxWithIndexNeon(const float32x4_t& val, const int32x4_t& index,
	float32x2_t& res, int32x2_t& resIndex)
{
	// Finds the maximum in pairs of 0 and 1 elements, and of 2 and 3 elements
	float32x4x2_t trn = vtrnq_f32(val, val);
	int32x4x2_t trnIndex = vtrnq_s32(index, index);

	float32x4_t max01_23 = vmaxq_f32(trn.val[0], trn.val[1]);
	// Use GE comparison to get the "earlier" indices when the values are the same
	int32x4_t maxIndex01_23 = ConditionIntNeon(vcgeq_f32(trn.val[0], trn.val[1]), trnIndex.val[0], trnIndex.val[1]);

	// Take the maximum from the top and bottom elements
	float32x2_t max01 = vget_low_f32(max01_23);
	float32x2_t max23 = vget_high_f32(max01_23);
	res = vmax_f32(max01, max23);
	// Use GE comparison to get the "earlier" indices when the values are the same
	resIndex = ConditionIntNeon(vcge_f32(max01, max23), vget_low_s32(maxIndex01_23), vget_high_s32(maxIndex01_23));
}

// Zero check: are all vector elements == 0
inline bool IsMaskZeroNeon(const uint32x4_t& val)
{
	uint32x2_t max01_23 = vpmax_u32(vget_low_u32(val), vget_high_u32(val));
	return vget_lane_u32(vpmax_u32(max01_23, max01_23), 0) == 0;
}

// multiply-add: first + second * mul
inline float32x4_t MultiplyAndAddNeon(const float32x4_t& first, const float32x4_t& second, const float32x4_t& mul)
{
#if defined(__clang__) && defined(__arm__)
	// CLANG mistakenly converts vmla to a pair of instructions vmul + vadd
	float32x4_t res = first;
	asm volatile(
		"vmla.f32 %q[res], %q[second], %q[mul]"
		: [res] "+w" (res) : [second] "w" (second), [mul] "w" (mul) : );
	return res;
#elif (defined __ARM_FEATURE_FMA)
	// FMA is faster than MLA, use that
	return vfmaq_f32(first, second, mul);
#else
	return vmlaq_f32(first, second, mul);
#endif
}

///////////////////////////////////////////////////////
// Auxiliary class for a matrix block
class alignas(16) CMatrixBlock4x4 : public CCrtAllocatedObject {
public:
	float32x4_t Rows[4];

	void Load4x4(const float* data, int rowSize)
	{
		Rows[0] = LoadNeon4(data);
		data += rowSize;
		Rows[1] = LoadNeon4(data);
		data += rowSize;
		Rows[2] = LoadNeon4(data);
		data += rowSize;
		Rows[3] = LoadNeon4(data);
	}

	void Load4xX(const float* data, int width, int rowSize)
	{
		Rows[0] = LoadNeon(data, width);
		data += rowSize;
		Rows[1] = LoadNeon(data, width);
		data += rowSize;
		Rows[2] = LoadNeon(data, width);
		data += rowSize;
		Rows[3] = LoadNeon(data, width);
	}

	void LoadYx4(const float* data, int height, int rowSize)
	{
		for(int j = 0; j < height; ++j) {
			Rows[j] = LoadNeon4(data);
			data += rowSize;
		}
		for(int j = height; j < 4; ++j) {
			Rows[j] = vdupq_n_f32(0);
		}
	}

	void LoadYxX(const float* data, int height, int width, int rowSize)
	{
		for(int j = 0; j < height; ++j) {
			Rows[j] = LoadNeon(data, width, 0);
			data += rowSize;
		}
		for(int j = height; j < 4; ++j) {
			Rows[j] = vdupq_n_f32(0);
		}
	}

	void Store4x4(float* data, int rowSize) const
	{
		StoreNeon4(Rows[0], data);
		data += rowSize;
		StoreNeon4(Rows[1], data);
		data += rowSize;
		StoreNeon4(Rows[2], data);
		data += rowSize;
		StoreNeon4(Rows[3], data);
	}

	void Store4xX(float* data, int width, int rowSize) const
	{
		StoreNeon(Rows[0], data, width);
		data += rowSize;
		StoreNeon(Rows[1], data, width);
		data += rowSize;
		StoreNeon(Rows[2], data, width);
		data += rowSize;
		StoreNeon(Rows[3], data, width);
	}

	void StoreYx4(float* data, int height, int rowSize) const
	{
		for(int j = 0; j < height; ++j) {
			StoreNeon4(Rows[j], data);
			data += rowSize;
		}
	}

	void StoreYxX(float* data, int height, int width, int rowSize) const
	{
		for(int j = 0; j < height; ++j) {
			StoreNeon(Rows[j], data, width);
			data += rowSize;
		}
	}

	void Transpose()
	{
		float32x4x2_t r01 = vtrnq_f32(Rows[0], Rows[1]);
		float32x4x2_t r23 = vtrnq_f32(Rows[2], Rows[3]);

		Rows[0] = vcombine_f32(vget_low_f32(r01.val[0]), vget_low_f32(r23.val[0]));
		Rows[1] = vcombine_f32(vget_low_f32(r01.val[1]), vget_low_f32(r23.val[1]));
		Rows[2] = vcombine_f32(vget_high_f32(r01.val[0]), vget_high_f32(r23.val[0]));
		Rows[3] = vcombine_f32(vget_high_f32(r01.val[1]), vget_high_f32(r23.val[1]));
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
// math

// reciprocal
inline float32x4_t InvNeon(const float32x4_t& val)
{
	float32x4_t cur = vrecpeq_f32(val);
	cur = vmulq_f32(vrecpsq_f32(val, cur), cur);
	cur = vmulq_f32(vrecpsq_f32(val, cur), cur);

	return cur;
}

// division
inline float32x4_t DivideNeon(const float32x4_t& left, const float32x4_t& right)
{
	return vmulq_f32(left, InvNeon(right));
}

// 1 / sqrt(val)
inline float32x4_t InvSqrtNeon(const float32x4_t& val)
{
	float32x4_t cur = vrsqrteq_f32(val);
	cur = vmulq_f32(vrsqrtsq_f32(cur, vmulq_f32(val, cur)), cur);
	cur = vmulq_f32(vrsqrtsq_f32(cur, vmulq_f32(val, cur)), cur);

	return cur;
}

// A 7th degree polynomial
inline float32x4_t Polynom8Neon(const float32x4_t& x,
	const float32x4_t& poly0, const float32x4_t& poly1, const float32x4_t& poly2, const float32x4_t& poly3,
	const float32x4_t& poly4, const float32x4_t& poly5, const float32x4_t& poly6, const float32x4_t& poly7)
{
	float32x4_t tail = MultiplyAndAddNeon( poly6, x, poly7 );
	tail = MultiplyAndAddNeon( poly5, x, tail );
	tail = MultiplyAndAddNeon( poly4, x, tail );
	tail = MultiplyAndAddNeon( poly3, x, tail );
	tail = MultiplyAndAddNeon( poly2, x, tail );
	tail = MultiplyAndAddNeon( poly1, x, tail );
	return MultiplyAndAddNeon( poly0, x, tail );
}
inline float32x4_t Polynom8Neon(const float32x4_t& x, const float32x4_t* poly)
{
	return Polynom8Neon(x, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7]);
}

// sqrt
class CSqrtNeon : public CCrtAllocatedObject {
public:
	CSqrtNeon() : MinVal(vdupq_n_f32(FLT_MIN)) {}

	float32x4_t Execute(const float32x4_t& val) const
	{
		return vmulq_f32(val, InvSqrtNeon(vmaxq_f32(MinVal, val)));
	}

private:
	const float32x4_t MinVal;
};

// exponent
// Based on Cephes math library exp implementation https://github.com/jeremybarnes/cephes/blob/master/cmath/exp.c
// We use a polynomial approximation exp(x) of 7 degree over the [-ln(2), ln(2)] interval with the Remez method
// The approximation uses Sollya 6.0 (http://sollya.gforge.inria.fr/)
// > remez(exp(x), 7, [-log(2); log(2)]);
// 0.99999998955224326136737550445628323296402203000823 + 0.99999999071360726125072399571785309591215423779367 * x +
// 0.50000069538458801897792933938736419406478763215156 * x^2 + 0.166666879115002977917564625997839817296176263934024 * x^3 +
// 4.165943865159889781284574394476291521671283507293e-2 * x^4 + 8.3319682865913493265997214713151267773418962940744e-3 * x^5 +
// 1.41290014424469272519724891748572016176588813531325e-3 * x^6 + 2.01747050601364376282735084344647908949985061826444e-4 * x^7
class CExpNeon : public CCrtAllocatedObject {
public:
	CExpNeon() :
		InvLog2(vdupq_n_f32(1.442695040888963407359924681001892137426645954153)),
		NegLog2(vdupq_n_f32(-0.69314718055994530941723212145817656807550013436025)),
		Poly0(vdupq_n_f32(0.99999998955224326136737550445628323296402203000823)),
		Poly1(vdupq_n_f32(0.99999999071360726125072399571785309591215423779367)),
		Poly2(vdupq_n_f32(0.50000069538458801897792933938736419406478763215156)),
		Poly3(vdupq_n_f32(0.166666879115002977917564625997839817296176263934024)),
		Poly4(vdupq_n_f32(4.165943865159889781284574394476291521671283507293e-2)),
		Poly5(vdupq_n_f32(8.3319682865913493265997214713151267773418962940744e-3)),
		Poly6(vdupq_n_f32(1.41290014424469272519724891748572016176588813531325e-3)),
		Poly7(vdupq_n_f32(2.01747050601364376282735084344647908949985061826444e-4)),
		MaxLog(vdupq_n_f32(FLT_MAX_LOG)),
		MinLog(vdupq_n_f32(FLT_MIN_LOG)),
		FloatBias(vdupq_n_s32(127))
	{
	}

	// Calculates the exponent without checking data. Overflow may occur because of values over FLT_MAX_LOG or below FLT_MIN_LOG
	float32x4_t ExecuteNoCheck( const float32x4_t& x ) const
	{
		// The formula: exp(x) = r * 2^n, where n = floor(0.5 + x / ln(2)), r = exp(x - n * ln(2))
		float32x4_t n = MultiplyAndAddNeon( vdupq_n_f32( 0.5f ), x, InvLog2 );

		// Perform a floorf
		float32x4_t toIntAndBack = vcvtq_f32_s32( vcvtq_s32_f32( n ) );
		uint32x4_t mask = vcgtq_f32( toIntAndBack, n );
		mask = vandq_u32( mask, vreinterpretq_u32_f32( vdupq_n_f32( 1.f ) ) );
		n = vsubq_f32( toIntAndBack, vreinterpretq_f32_u32( mask ) );

		// Calculate r (via the polynomial)
		float32x4_t r = Polynom8Neon( MultiplyAndAddNeon( x, n, NegLog2 ),
			Poly0, Poly1, Poly2, Poly3, Poly4, Poly5, Poly6, Poly7 );

		// Calculate r * 2^n. Use the fact that n stores the binary exponent in bit positions from 23 to 30 (the 31 bit stores the sign)
		int32x4_t pow2n = vshlq_n_s32( vaddq_s32( vcvtq_s32_f32( n ), FloatBias ), 23 );
		return vmulq_f32( r, vreinterpretq_f32_s32( pow2n ) );
	}

	// Calculate the exponent with saturation
	float32x4_t Execute( float32x4_t x ) const
	{
		x = vminq_f32( x, MaxLog );
		x = vmaxq_f32( x, MinLog );
		return ExecuteNoCheck( x );
	}

private:
	// The constants used in the algorithm
	const float32x4_t InvLog2;
	const float32x4_t NegLog2;
	const float32x4_t Poly0, Poly1, Poly2, Poly3, Poly4, Poly5, Poly6, Poly7;
	const float32x4_t MaxLog;
	const float32x4_t MinLog;
	const int32x4_t FloatBias;
};

// Logarithm
// We use a polynomial approximation ln(x) of 7 degree over the [1, 2] interval with the Remez method
// The approximation uses Sollya 6.0 (http://sollya.gforge.inria.fr/)
// > remez(log(x), 7, [1; 2]);
// -2.2496354384323994835209829226495141810521420107384 + 4.9448910244549799872106325311521085580266870326504 * x +
// -5.1945351982243051696422411814528666698483563209764 * x^2 + 4.0073882206207432223548376040428533830499620725419 * x^3 +
// -2.06905895742501636916193336058740532558260222207 * x^4 + 0.6779636853241939027852947156614612589585157662135 * x^5 +
// -0.12749724414788236804817747106717809620960121948297 * x^6 + 1.04841000320826930139331087157692539801107792338888e-2 * x^7
// The multipliers are corrected to give a total of exactly 0 (ln(1))
class CLogNeon : public CCrtAllocatedObject {
public:
	CLogNeon() :
		Log2(vdupq_n_f32(0.69314718055994530941723212145817656807550013436025)),
		Poly0(vdupq_n_f32(-2.2496354384323994835209829226495141810521420107384)),
		Poly1(vdupq_n_f32(4.9448910244549799872106325311521085580266870326504)),
		Poly2(vdupq_n_f32(-5.1945351982243051696422411814528666698483563209764)),
		Poly3(vdupq_n_f32(4.0073882206207432223548376040428533830499620725419)),
		Poly4(vdupq_n_f32(-2.06905895742501636916193336058740532558260222207)),
		Poly5(vdupq_n_f32(0.6779636853241939027852947156614612589585157662135)),
		Poly6(vdupq_n_f32(-0.12749724414788236804817747106717809620960121948297)),
		Poly7(vdupq_n_f32(1.04841000320826930139331087157692539801107792338888e-2)),
		MinValue(vdupq_n_f32(FLT_MIN)),
		FloatBias(vdupq_n_s32(127))
	{
	}

	// Calculates the logarithm without checking data. Overflow may occur because of values below FLT_MIN
	float32x4_t ExecuteNoCheck(const float32x4_t& x) const
	{
		// The formula: ln(x) = r + n * ln(2), where n = [log2(x)], r = ln(x / 2^n)
		// Use the polynomial approximation to calculate r
		// When calculating [log2(x)], use the fact that bits 23 to 30 contain the binary exponent
		// the bits 0 to 22 contain the binary mantissa, which is in the [1, 2) range for normalized values
		// Does not work for negative values (sign bit = 1) or denormalized values (mantissa in [0, 1) range)
		int32x4_t n = vsubq_s32(vshrq_n_s32(vreinterpretq_s32_f32(x), 23), FloatBias);

		// Calculate r (via the polynomial)
		float32x4_t r = Polynom8Neon(vreinterpretq_f32_s32(vsubq_s32(x, vshlq_n_s32(n, 23))),
			Poly0, Poly1, Poly2, Poly3, Poly4, Poly5, Poly6, Poly7);

		return vaddq_f32(r, vmulq_f32(vcvtq_f32_s32(n), Log2));
	}

	// Calculate the logarithm with saturation
	float32x4_t Execute(const float32x4_t& x) const
	{
		return ExecuteNoCheck(vmaxq_f32(x, MinValue));
	}

private:
	// The constants used in the algorithm
	const float32x4_t Log2;
	const float32x4_t Poly0, Poly1, Poly2, Poly3, Poly4, Poly5, Poly6, Poly7;
	const float32x4_t MinValue;
	const int32x4_t FloatBias;
};

} // namespace NeoML

#endif // NEOML_USE_NEON
