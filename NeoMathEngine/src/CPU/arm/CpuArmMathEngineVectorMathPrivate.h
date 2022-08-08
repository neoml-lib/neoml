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

inline void vectorAdd( const int* first, const int* second, int* result, int vectorSize )
{
	int coord = 0;

	for( ; coord <= vectorSize - 16; coord += 16 ) {
		NEON_LOAD_16_INTS( first, first );
		first += 16;

		NEON_LOAD_16_INTS( second, second );
		second += 16;

		int32x4_t result0 = vaddq_s32( first0, second0 );
		int32x4_t result1 = vaddq_s32( first1, second1 );
		int32x4_t result2 = vaddq_s32( first2, second2 );
		int32x4_t result3 = vaddq_s32( first3, second3 );

		NEON_STORE_16_INTS( result, result );
		result += 16;
	}

	for( ; coord <= vectorSize - 4; coord += 4 ) {
		int32x4_t first0 = LoadIntNeon4( first );
		first += 4;

		int32x4_t second0 = LoadIntNeon4( second );
		second += 4;

		int32x4_t result0 = vaddq_s32( first0, second0 );

		StoreIntNeon4( result0, result );
		result += 4;
	}

	vectorSize -= coord;
	if( vectorSize > 0 ) {
		int32x4_t res = vaddq_s32( LoadIntNeon( first, vectorSize ), LoadIntNeon( second, vectorSize ) );
		StoreIntNeon( res, result, vectorSize );
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

inline void vectorMultiply( const float* first, float* result, float multiplier, int vectorSize )
{
	int count = GetCount4(vectorSize);
	float32x4_t mult = vdupq_n_f32(multiplier);

	for(int i = 0; i < count; ++i) {
		float32x4_t res = vmulq_f32(LoadNeon4(first), mult);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		float32x4_t res = vmulq_f32(LoadNeon(first, vectorSize), mult);
		StoreNeon(res, result, vectorSize);
	}
}

inline void vectorMultiply( const int* first, int* result, int multiplier, int vectorSize )
{
	int count = GetCount4(vectorSize);
	int32x4_t mult = vdupq_n_s32(multiplier);

	for(int i = 0; i < count; ++i) {
		StoreIntNeon4(vmulq_s32(LoadIntNeon4(first), mult), result);
		first += 4;
		result += 4;
	}

	if(vectorSize > 0) {
		StoreIntNeon(vmulq_s32(LoadIntNeon(first, vectorSize), mult), result, vectorSize);
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorEltwiseMultiply( const float* first, const float* second, float* result, int neonSize, int nonNeonSize )
{
	while( neonSize >= 4 ) {
		NEON_LOAD_16_FLOATS(first, first);
		first += 16;

		NEON_LOAD_16_FLOATS(second, second);
		second += 16;

		float32x4_t result0 = vmulq_f32( first0, second0 );
		float32x4_t result1 = vmulq_f32( first1, second1 );
		float32x4_t result2 = vmulq_f32( first2, second2 );
		float32x4_t result3 = vmulq_f32( first3, second3 );

		NEON_STORE_16_FLOATS(result, result);
		result += 16;

		neonSize -= 4;
	}

	while( neonSize > 0 ) {
		float32x4_t first0 = LoadNeon4( first );
		first += 4;

		float32x4_t second0 = LoadNeon4( second );
		second += 4;

		float32x4_t res0 = vmulq_f32( first0, second0 );
		StoreNeon4( res0, result );
		result += 4;

		neonSize--;
	}

	if( nonNeonSize ) {
		float32x4_t first0 = LoadNeon( first, nonNeonSize );
		float32x4_t second0 = LoadNeon( second, nonNeonSize );
		float32x4_t res0 = vmulq_f32( first0, second0 );
		StoreNeon( res0, result, nonNeonSize );
	}
}

inline void vectorEltwiseMultiply( const float* first, const float* second, float* result, int vectorSize )
{
	vectorEltwiseMultiply( first, second, result, vectorSize / 4, vectorSize % 4 );
}

inline void vectorEltwiseMultiply( const int* first, const int* second, int* result, int vectorSize )
{
	int neonSize = vectorSize / 4;
	int nonNeonSize = vectorSize % 4;

	while( neonSize >= 4 ) {
		NEON_LOAD_16_INTS(first, first);
		first += 16;

		NEON_LOAD_16_INTS(second, second);
		second += 16;

		int32x4_t result0 = vmulq_s32( first0, second0 );
		int32x4_t result1 = vmulq_s32( first1, second1 );
		int32x4_t result2 = vmulq_s32( first2, second2 );
		int32x4_t result3 = vmulq_s32( first3, second3 );

		NEON_STORE_16_INTS(result, result);
		result += 16;

		neonSize -= 4;
	}

	while( neonSize > 0 ) {
		int32x4_t first0 = LoadIntNeon4( first );
		first += 4;

		int32x4_t second0 = LoadIntNeon4( second );
		second += 4;

		int32x4_t res0 = vmulq_s32( first0, second0 );
		StoreIntNeon4( res0, result );
		result += 4;

		neonSize--;
	}

	if( nonNeonSize ) {
		int32x4_t first0 = LoadIntNeon( first, nonNeonSize );
		int32x4_t second0 = LoadIntNeon( second, nonNeonSize );
		int32x4_t res0 = vmulq_s32( first0, second0 );
		StoreIntNeon( res0, result, nonNeonSize );
	}
}

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

//------------------------------------------------------------------------------------------------------------

inline void vectorAddValue( const float* first, float* result, int vectorSize, float value )
{
	float32x4_t addition = vdupq_n_f32(value);

	int coord = 0;
	for( ; coord <= vectorSize - 4; coord += 4 ) {
		float32x4_t res = vaddq_f32(LoadNeon4(first), addition);
		StoreNeon4(res, result);

		first += 4;
		result += 4;
	}

	vectorSize -= coord;
	if(vectorSize > 0) {
		float32x4_t res = vaddq_f32(LoadNeon(first, vectorSize), addition);
		StoreNeon(res, result, vectorSize);
	}
}

//------------------------------------------------------------------------------------------------------------

inline void vectorDotProduct( const float* first, const float* second, int vectorSize, float* result )
{
	float32x4_t acc = vdupq_n_f32(0);

	int coord = 0;
	for( ; coord <= vectorSize - 4; coord += 4 ) {
		float32x4_t res = vmulq_f32(LoadNeon4(first), LoadNeon4(second));
		acc = vaddq_f32(acc, res);

		first += 4;
		second += 4;
	}

	vectorSize -= coord;
	if(vectorSize > 0) {
		float32x4_t res = vmulq_f32(LoadNeon(first, vectorSize, 0), LoadNeon(second, vectorSize, 0));
		acc = vaddq_f32(acc, res);
	}

	*result = vget_lane_f32(HorizontalAddNeon(acc), 0);
}

//------------------------------------------------------------------------------------------------------------

// QRNN primitives

// res = z * ( 1 - f )
static inline void qrnnFPoolingFirstStep( const float* z, const float* f,
	float* res, int neonSize, int nonNeonSize )
{
	float32x4_t ones = vdupq_n_f32( 1.f );
	while( neonSize >= 4 ) {
		NEON_LOAD_16_FLOATS( z, z );
		z += 16;

		NEON_LOAD_16_FLOATS( f, f );
		f += 16;

		float32x4_t res0 = vmulq_f32( z0, vsubq_f32( ones, f0 ) );
		float32x4_t res1 = vmulq_f32( z1, vsubq_f32( ones, f1 ) );
		float32x4_t res2 = vmulq_f32( z2, vsubq_f32( ones, f2 ) );
		float32x4_t res3 = vmulq_f32( z3, vsubq_f32( ones, f3 ) );

		NEON_STORE_16_FLOATS( res, res );
		res += 16;

		neonSize -= 4;
	}

	while( neonSize > 0 ) {
		float32x4_t z0 = LoadNeon4( z );
		z += 4;

		float32x4_t f0 = LoadNeon4( f );
		f += 4;

		float32x4_t res0 = vmulq_f32( z0, vsubq_f32( ones, f0 ) );
		StoreNeon4( res0, res );
		res += 4;

		--neonSize;
	}

	if( nonNeonSize > 0 ) {
		float32x4_t z0 = LoadNeon( z, nonNeonSize );
		float32x4_t f0 = LoadNeon( f, nonNeonSize );
		float32x4_t res0 = vmulq_f32( z0, vsubq_f32( ones, f0 ) );
		StoreNeon( res0, res, nonNeonSize );
	}
}

// res = f * h + (1 - f) * z
// where h - res of previous step
static inline void qrnnFPoolingStep( const float* z, const float* f, const float* h,
	float* res, int neonSize, int nonNeonSize )
{
	float32x4_t ones = vdupq_n_f32( 1.f );
	while( neonSize >= 4 ) {
		NEON_LOAD_16_FLOATS(z, z);
		z += 16;

		NEON_LOAD_16_FLOATS(f, f);
		f += 16;

		NEON_LOAD_16_FLOATS(h, h);
		h += 16;

		float32x4_t res0 = vaddq_f32( vmulq_f32( f0, h0 ), vmulq_f32( vsubq_f32( ones, f0 ), z0 ) );
		float32x4_t res1 = vaddq_f32( vmulq_f32( f1, h1 ), vmulq_f32( vsubq_f32( ones, f1 ), z1 ) );
		float32x4_t res2 = vaddq_f32( vmulq_f32( f2, h2 ), vmulq_f32( vsubq_f32( ones, f2 ), z2 ) );
		float32x4_t res3 = vaddq_f32( vmulq_f32( f3, h3 ), vmulq_f32( vsubq_f32( ones, f3 ), z3 ) );

		NEON_STORE_16_FLOATS(res, res);
		res += 16;

		neonSize -= 4;
	}

	while( neonSize > 0 ) {
		float32x4_t z0 = LoadNeon4( z );
		z += 4;

		float32x4_t f0 = LoadNeon4( f );
		f += 4;

		float32x4_t h0 = LoadNeon4( h );
		h += 4;

		float32x4_t res0 = vaddq_f32( vmulq_f32( f0, h0 ), vmulq_f32( vsubq_f32( ones, f0 ), z0 ) );
		StoreNeon4( res0, res );
		res += 4;

		neonSize--;
	}

	if( nonNeonSize > 0 ) {
		float32x4_t z0 = LoadNeon( z, nonNeonSize );
		float32x4_t f0 = LoadNeon( f, nonNeonSize );
		float32x4_t h0 = LoadNeon( h, nonNeonSize );
		float32x4_t res0 = vaddq_f32( vmulq_f32( f0, h0 ), vmulq_f32( vsubq_f32( ones, f0 ), z0 ) );
		StoreNeon( res0, res, nonNeonSize );
	}
}

// res = f * h + i * z
// where h is res of previous step
static inline void qrnnIfPoolingStep( const float* z, const float* f, const float* i, const float* h,
	float* res, int neonSize, int nonNeonSize )
{
	while( neonSize >= 4 ) {
		NEON_LOAD_16_FLOATS(z, z);
		z += 16;

		NEON_LOAD_16_FLOATS(f, f);
		f += 16;

		NEON_LOAD_16_FLOATS(i, i);
		i += 16;

		NEON_LOAD_16_FLOATS(h, h);
		h += 16;

		float32x4_t res0 = vaddq_f32( vmulq_f32( f0, h0 ), vmulq_f32( i0, z0 ) );
		float32x4_t res1 = vaddq_f32( vmulq_f32( f1, h1 ), vmulq_f32( i1, z1 ) );
		float32x4_t res2 = vaddq_f32( vmulq_f32( f2, h2 ), vmulq_f32( i2, z2 ) );
		float32x4_t res3 = vaddq_f32( vmulq_f32( f3, h3 ), vmulq_f32( i3, z3 ) );

		NEON_STORE_16_FLOATS(res, res);
		res += 16;

		neonSize -= 4;
	}

	while( neonSize > 0 ) {
		float32x4_t z0 = LoadNeon4( z );
		z += 4;

		float32x4_t f0 = LoadNeon4( f );
		f += 4;

		float32x4_t h0 = LoadNeon4( h );
		h += 4;

		float32x4_t i0 = LoadNeon4( i );
		i += 4;

		float32x4_t res0 = vaddq_f32( vmulq_f32( f0, h0 ), vmulq_f32( i0, z0 ) );
		StoreNeon4( res0, res );
		res += 4;

		neonSize--;
	}

	if( nonNeonSize > 0 ) {
		float32x4_t z0 = LoadNeon( z, nonNeonSize );
		float32x4_t f0 = LoadNeon( f, nonNeonSize );
		float32x4_t h0 = LoadNeon( h, nonNeonSize );
		float32x4_t i0 = LoadNeon( i, nonNeonSize );
		float32x4_t res0 = vaddq_f32( vmulq_f32( f0, h0 ), vmulq_f32( i0, z0 ) );
		StoreNeon( res0, res, nonNeonSize );
	}
}

inline void vectorMinMax( const float* first, float* result, const float minValue, const float maxValue, int vectorSize )
{
	int count = GetCount4(vectorSize);

	float32x4_t minVal = vdupq_n_f32(minValue);
	float32x4_t maxVal = vdupq_n_f32(maxValue);

	while( count >= 4 ) {
		NEON_LOAD_16_FLOATS( first, first );
		first += 16;

		float32x4_t res0 = vmaxq_f32(minVal, vminq_f32(maxVal, first0));
		float32x4_t res1 = vmaxq_f32(minVal, vminq_f32(maxVal, first1));
		float32x4_t res2 = vmaxq_f32(minVal, vminq_f32(maxVal, first2));
		float32x4_t res3 = vmaxq_f32(minVal, vminq_f32(maxVal, first3));

		NEON_STORE_16_FLOATS( res, result );
		result += 16;

		count -= 4;
	}

	while( count > 0 ) {
		float32x4_t res = vmaxq_f32(minVal, vminq_f32(maxVal, LoadNeon4(first)));
		StoreNeon4(res, result);

		first += 4;
		result += 4;
		--count;
	}

	if(vectorSize > 0) {
		float32x4_t res = vmaxq_f32(minVal, vminq_f32(maxVal, LoadNeon(first, vectorSize)));
		StoreNeon(res, result, vectorSize);
	}
}

inline float32x4_t vectorTanhWorker( const float32x4_t& val, const float32x4_t& one, const CExpNeon& expObj )
{
	float32x4_t expVal = expObj.Execute( vnegq_f32( vaddq_f32( val, val ) ) );
	float32x4_t inv = InvNeon( vaddq_f32( one, expVal ) );
	return vsubq_f32( vaddq_f32( inv, inv ), one );
}

inline void vectorTanh( const float* first, float* result, int vectorSize )
{
	int count = GetCount4( vectorSize );

	const float32x4_t one = vdupq_n_f32( 1.f );
	const CExpNeon expObj;

	for( int i = 0; i < count; ++i ) {
		float32x4_t res = vectorTanhWorker( LoadNeon4( first ), one, expObj );
		StoreNeon4( res, result );

		first += 4;
		result += 4;
	}

	if( vectorSize > 0 ) {
		float32x4_t res = vectorTanhWorker( LoadNeon( first, vectorSize ), one, expObj );
		StoreNeon( res, result, vectorSize );
	}
}

inline float32x4_t vectorSigmoidWorker( const float32x4_t& val, const float32x4_t& one, const CExpNeon& expObj )
{
	return InvNeon( vaddq_f32( one, expObj.Execute( vnegq_f32( val ) ) ) );
}

inline void vectorSigmoid( const float* first, float* result, int vectorSize )
{
	int count = GetCount4( vectorSize );

	const float32x4_t one = vdupq_n_f32( 1.f );
	const CExpNeon expObj;

	for( int i = 0; i < count; ++i ) {
		float32x4_t res = vectorSigmoidWorker( LoadNeon4( first ), one, expObj );
		StoreNeon4( res, result );

		first += 4;
		result += 4;
	}

	if( vectorSize > 0 ) {
		float32x4_t res = vectorSigmoidWorker( LoadNeon( first, vectorSize ), one, expObj );
		StoreNeon( res, result, vectorSize );
	}
}

} // namespace NeoML

#endif
