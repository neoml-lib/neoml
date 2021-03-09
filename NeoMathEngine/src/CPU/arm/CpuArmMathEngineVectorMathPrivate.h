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

// These functions work with raw pointers, may be called from OMP sections and perform no parameter checks

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_NEON

#include <CpuArm.h>

namespace NeoML {

inline void channelwiseConvolution1x3Kernel( const float* source0, const float* source1, const float* source2, const float* source3,
	const float* filter0, const float* filter1, const float* filter2,
	float* result0, float* result1 )
{
	float32x4_t filter_4 = vld1q_f32(filter0);

	float32x4_t source0_4 = vld1q_f32(source0);
	float32x4_t source1_4 = vld1q_f32(source1);

	float32x4_t result0_4 = vld1q_f32(result0);
	float32x4_t result1_4 = vld1q_f32(result1);

	result0_4 = MultiplyAndAddNeon( result0_4, source0_4, filter_4 );
	result1_4 = MultiplyAndAddNeon( result1_4, source1_4, filter_4 );

	filter_4 = vld1q_f32(filter1);

	result0_4 = MultiplyAndAddNeon( result0_4, source1_4, filter_4 );

	source0_4 = vld1q_f32(source2);
	source1_4 = vld1q_f32(source3);

	result1_4 = MultiplyAndAddNeon( result1_4, source0_4, filter_4 );

	filter_4 = vld1q_f32(filter2);

	result0_4 = MultiplyAndAddNeon( result0_4, source0_4, filter_4 );
	result1_4 = MultiplyAndAddNeon( result1_4, source1_4, filter_4 );

	vst1q_f32(result0, result0_4);
	vst1q_f32(result1, result1_4);
}

inline void channelwise1x3( const float* source, const float* filter0, const float* filter1, const float* filter2, float* result, int channels )
{
	const int shift1 = channels;
	const int shift2 = 2 * channels;
	const int shift3 = 3 * channels;
	while( channels > 0 ) {
		channelwiseConvolution1x3Kernel( source, source + shift1, source + shift2, source + shift3,
			filter0, filter1, filter2,
			result, result + shift1 );

		source += 4;
		filter0 += 4; filter1 += 4; filter2 += 4;
		result += 4;
		channels -= 4;
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorFill( float* result, float value, int vectorSize )
{
	int coord = 0;

	float32x4_t val = vdupq_n_f32(value);
	for( ; coord <= vectorSize - 16; coord += 16 ) {
		StoreNeon4(val, result + 4 * 0);
		StoreNeon4(val, result + 4 * 1);
		StoreNeon4(val, result + 4 * 2);
		StoreNeon4(val, result + 4 * 3);
		
		result += 16;
	}

	for( ; coord <= vectorSize - 4; coord += 4 ) {
		StoreNeon4(val, result);
		result += 4;
	}

	for( ; coord < vectorSize; ++coord ) {
		*result++ = value;
	}
}

inline void vectorFill( int* result, int value, int vectorSize )
{
	int coord = 0;

	int32x4_t val = vdupq_n_s32(value);
	for( ; coord <= vectorSize - 16; coord += 16 ) {
		StoreIntNeon4(val, result + 4 * 0);
		StoreIntNeon4(val, result + 4 * 1);
		StoreIntNeon4(val, result + 4 * 2);
		StoreIntNeon4(val, result + 4 * 3);
		
		result += 16;
	}

	for( ; coord <= vectorSize - 4; coord += 4 ) {
		StoreIntNeon4(val, result);
		result += 4;
	}

	for( ; coord < vectorSize; ++coord ) {
		*result++ = value;
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorFill0( float* result, int vectorSize )
{
	vectorFill( result, 0, vectorSize );
}

//------------------------------------------------------------------------------------------------------------

inline void vectorEltwiseMax( const float* first, const float* second, float* result, int vectorSize )
{
	int count = GetCount4(vectorSize);

	for( ; count >= 4; count -= 4, first += 16, second += 16, result += 16 ) {
		NEON_LOAD_16_FLOATS(first, first);
		NEON_LOAD_16_FLOATS(second, second);

		float32x4_t result0 = vmaxq_f32(first0, second0);
		float32x4_t result1 = vmaxq_f32(first1, second1);
		float32x4_t result2 = vmaxq_f32(first2, second2);
		float32x4_t result3 = vmaxq_f32(first3, second3);

		NEON_STORE_16_FLOATS(result, result);
	}

	for( int i = 0; i < count; ++i ) {
		float32x4_t res = vmaxq_f32(LoadNeon4(first), LoadNeon4(second));
		StoreNeon4(res, result);

		first += 4;
		second += 4;
		result += 4;
	}

	if( vectorSize > 0 ) {
		float32x4_t res = vmaxq_f32(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize));
		StoreNeon(res, result, vectorSize);
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorAdd( const float* first, const float* second, float* result, int vectorSize )
{
	int coord = 0;

	for( ; coord <= vectorSize - 16; coord += 16 ) {
		NEON_LOAD_16_FLOATS(first, first);
		first += 16;

		NEON_LOAD_16_FLOATS(second, second);
		second += 16;

		float32x4_t result0 = vaddq_f32(first0, second0);
		float32x4_t result1 = vaddq_f32(first1, second1);
		float32x4_t result2 = vaddq_f32(first2, second2);
		float32x4_t result3 = vaddq_f32(first3, second3);

		NEON_STORE_16_FLOATS(result, result);
		result += 16;
	}

	for( ; coord <= vectorSize - 4; coord += 4 ) {
		float32x4_t first0 = LoadNeon4(first);
		first += 4;

		float32x4_t second0 = LoadNeon4(second);
		second += 4;

		float32x4_t result0 = vaddq_f32(first0, second0);

		StoreNeon4(result0, result);
		result += 4;
	}

	vectorSize -= coord;
	if(vectorSize > 0) {
		float32x4_t res = vaddq_f32(LoadNeon(first, vectorSize), LoadNeon(second, vectorSize));
		StoreNeon(res, result, vectorSize);
	}
}

//------------------------------------------------------------------------------------------------------------

inline void alignedVectorAdd( float* first, const float* second, int vectorSize )
{
	int coord = 0;

	for( ; coord <= vectorSize - 16; coord += 16, first += 16, second += 16 ) {
		NEON_LOAD_16_FLOATS(first, first);
		NEON_LOAD_16_FLOATS(second, second);

		float32x4_t result0 = vaddq_f32(first0, second0);
		float32x4_t result1 = vaddq_f32(first1, second1);
		float32x4_t result2 = vaddq_f32(first2, second2);
		float32x4_t result3 = vaddq_f32(first3, second3);

		NEON_STORE_16_FLOATS(result, first);
	}

	for( ; coord <= vectorSize - 4; coord += 4, first += 4, second += 4 ) {
		float32x4_t first0 = LoadNeon4(first);
		float32x4_t second0 = LoadNeon4(second);

		float32x4_t result0 = vaddq_f32(first0, second0);

		StoreNeon4(result0, first);
	}
}

inline void alignedVectorMultiplyAndAdd( const float* first, const float* second,
	float* result, int vectorSize, const float* mult )
{
	int coord = 0;

	for( ; coord <= vectorSize - 16; coord += 16 ) {
		NEON_LOAD_16_FLOATS(first, first);
		first += 16;

		NEON_LOAD_16_FLOATS(second, second);
		second += 16;

		float32x4_t result0 = vmlaq_n_f32(first0, second0, *mult);
		float32x4_t result1 = vmlaq_n_f32(first1, second1, *mult);
		float32x4_t result2 = vmlaq_n_f32(first2, second2, *mult);
		float32x4_t result3 = vmlaq_n_f32(first3, second3, *mult);

		NEON_STORE_16_FLOATS(result, result);
		result += 16;
	}

	for( ; coord <= vectorSize - 4; coord += 4 ) {
		float32x4_t first0 = LoadNeon4(first);
		first += 4;

		float32x4_t second0 = LoadNeon4(second);
		second += 4;

		float32x4_t result0 = vmlaq_n_f32(first0, second0, *mult);

		StoreNeon4(result0, result);
		result += 4;
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorEltwiseMultiplyAdd( const float* first, const float* second, float* result, int vectorSize )
{
	int coord = 0;

	for( ; coord <= vectorSize - 16; coord += 16 ) {
		NEON_LOAD_16_FLOATS(first, first);
		first += 16;

		NEON_LOAD_16_FLOATS(second, second);
		second += 16;

		NEON_LOAD_16_FLOATS(result, result);

		result0 = MultiplyAndAddNeon(result0, first0, second0);
		result1 = MultiplyAndAddNeon(result1, first1, second1);
		result2 = MultiplyAndAddNeon(result2, first2, second2);
		result3 = MultiplyAndAddNeon(result3, first3, second3);

		NEON_STORE_16_FLOATS(result, result);
		result += 16;
	}

	for( ; coord <= vectorSize - 4; coord += 4 ) {
		float32x4_t first0 = LoadNeon4(first);
		first += 4;

		float32x4_t second0 = LoadNeon4(second);
		second += 4;

		float32x4_t result0 = LoadNeon4(result);

		result0 = MultiplyAndAddNeon(result0, first0, second0);

		StoreNeon4(result0, result);
		result += 4;
	}

	vectorSize -= coord;
	if( vectorSize > 0 ) {
		float32x4_t res = MultiplyAndAddNeon(LoadNeon(result, vectorSize),
			LoadNeon(first, vectorSize), LoadNeon(second, vectorSize));
		StoreNeon(res, result, vectorSize);
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorReLU( const float* first, float* result, int vectorSize )
{
	int coord = 0;

	for( ; coord <= vectorSize - 16; coord += 16 ) {
		NEON_LOAD_16_FLOATS(first, first);
		first += 16;

		float32x4_t result0 = vmaxq_f32(vdupq_n_f32(0), first0);
		float32x4_t result1 = vmaxq_f32(vdupq_n_f32(0), first1);
		float32x4_t result2 = vmaxq_f32(vdupq_n_f32(0), first2);
		float32x4_t result3 = vmaxq_f32(vdupq_n_f32(0), first3);

		NEON_STORE_16_FLOATS(result, result);
		result += 16;
	}

	for( ; coord <= vectorSize - 4; coord += 4 ) {
		float32x4_t first0 = LoadNeon4(first);
		first += 4;

		float32x4_t result0 = vmaxq_f32(vdupq_n_f32(0), first0);

		StoreNeon4(result0, result);
		result += 4;
	}

	vectorSize -= coord;
	if( vectorSize > 0 ) {
		float32x4_t first0 = LoadNeon(first, vectorSize);
		float32x4_t result0 = vmaxq_f32(vdupq_n_f32(0), first0);
		StoreNeon(result0, result, vectorSize);
	}
}

inline void vectorReLU( const float* first, float* result, int vectorSize, float threshold )
{
	int coord = 0;

	for( ; coord <= vectorSize - 16; coord += 16 ) {
		NEON_LOAD_16_FLOATS(first, first);
		first += 16;

		float32x4_t result0 = vminq_f32(vmaxq_f32(vdupq_n_f32(0), first0), vdupq_n_f32(threshold));
		float32x4_t result1 = vminq_f32(vmaxq_f32(vdupq_n_f32(0), first1), vdupq_n_f32(threshold));
		float32x4_t result2 = vminq_f32(vmaxq_f32(vdupq_n_f32(0), first2), vdupq_n_f32(threshold));
		float32x4_t result3 = vminq_f32(vmaxq_f32(vdupq_n_f32(0), first3), vdupq_n_f32(threshold));

		NEON_STORE_16_FLOATS(result, result);
		result += 16;
	}

	for( ; coord <= vectorSize - 4; coord += 4 ) {
		float32x4_t first0 = LoadNeon4(first);
		first += 4;

		float32x4_t result0 = vminq_f32(vmaxq_f32(vdupq_n_f32(0), first0), vdupq_n_f32(threshold));

		StoreNeon4(result0, result);
		result += 4;
	}

	vectorSize -= coord;
	if( vectorSize > 0 ) {
		float32x4_t first0 = LoadNeon(first, vectorSize);
		float32x4_t result0 = vminq_f32(vmaxq_f32(vdupq_n_f32(0), first0), vdupq_n_f32(threshold));
		StoreNeon(result0, result, vectorSize);
	}
}

} // namespace NeoML

#endif
