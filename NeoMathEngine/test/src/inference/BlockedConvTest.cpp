/* Copyright © 2017-2022 ABBYY Production LLC

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

#include <immintrin.h>

#include <TestFixture.h>
#include <MeTestCommon.h>

#include <stdio.h>

using namespace NeoML;
using namespace NeoMLTest;

struct KernelFrame {
	int Padding;
	const float* Input;
	const float* Filter;
	const float* Bias;
	float* Output;
	int StrideWidth;
	int DilationWidth;
	int FilterCount;
	int InputStride;
	int FilterStride;
	int OutputStride;
	int KernelHeight;
	int KernelWidth;
	const float* InputBase;
	int InputWidth;
	int DilatedInputWidth;
	int OutputCountLeftPad;
	int OutputCount;
	int OutputCountRightPad;
	int Flags;
};

#define ACCUMULATE_OUTPUT 1
#define ADD_BIAS 2

/*
// This macro multiplies and accumulates for FilterCount by OutputCount block of the output buffer.
#define COMPUTE_BLOCK(FilterCount, OutputCount, VectorOffset, BroadcastOffset) \
		if( OutputCount >= 1 )  acc13 = _mm256_broadcastss_ps( _mm256_loadu_ps( input + BroadcastOffset ) ); \
		if( OutputCount >= 2 )  acc14 = _mm256_broadcastss_ps( _mm256_loadu_ps( input + strideWidth + BroadcastOffset ) ); \
		if( OutputCount >= 3 )  acc14 = _mm256_broadcastss_ps( _mm256_loadu_ps( input + strideWidth * 2 + BroadcastOffset ) ); \
		if( OutputCount == 1 ) { \
			if( FilterCount >= 1 ) acc0 = _mm256_fmadd_ps( acc13, _mm256_loadu_ps( filter + VectorOffset ), acc0 ); \
			if( FilterCount >= 2 ) acc1 = _mm256_fmadd_ps( acc13, _mm256_loadu_ps( filter + filterStride + VectorOffset ), acc1 ); \
			if( FilterCount >= 3 ) acc2 = _mm256_fmadd_ps( acc13, _mm256_loadu_ps( shiftedFilter + VectorOffset ), acc2 ); \
			if( FilterCount >= 4 ) acc3 = _mm256_fmadd_ps( acc13, _mm256_loadu_ps( shiftedFilter + filterStride + VectorOffset ), acc3 ); \
		} else { \
			if( FilterCount >= 1 ) acc12 = _mm256_loadu_ps( filter + VectorOffset ); \
			if( FilterCount >= 1 && OutputCount >= 1 ) acc0 = _mm256_fmadd_ps( acc13, acc12, acc0 ); \
			if( FilterCount >= 1 && OutputCount >= 2 ) acc4 = _mm256_fmadd_ps( acc14, acc12, acc4 ); \
			if( FilterCount >= 1 && OutputCount >= 3 ) acc8 = _mm256_fmadd_ps( acc15, acc12, acc8 ); \
			if( FilterCount >= 2 ) acc12 = _mm256_loadu_ps( filter + filterStride + VectorOffset ); \
			if( FilterCount >= 2 && OutputCount >= 1 ) acc1 = _mm256_fmadd_ps( acc13, acc12, acc1 ); \
			if( FilterCount >= 2 && OutputCount >= 2 ) acc5 = _mm256_fmadd_ps( acc14, acc12, acc5 ); \
			if( FilterCount >= 2 && OutputCount >= 3 ) acc9 = _mm256_fmadd_ps( acc15, acc12, acc9 ); \
			if( FilterCount >= 3 ) acc12 = _mm256_loadu_ps( shiftedFilter + VectorOffset ); \
			if( FilterCount >= 3 && OutputCount >= 1 ) acc2 = _mm256_fmadd_ps( acc13, acc12, acc2 ); \
			if( FilterCount >= 3 && OutputCount >= 2 ) acc6 = _mm256_fmadd_ps( acc14, acc12, acc6 ); \
			if( FilterCount >= 3 && OutputCount >= 3 ) acc10 = _mm256_fmadd_ps( acc15, acc12, acc10 ); \
			if( FilterCount >= 4 ) acc12 = _mm256_loadu_ps( shiftedFilter + filterStride + VectorOffset ); \
			if( FilterCount >= 4 && OutputCount >= 1 ) acc3 = _mm256_fmadd_ps( acc13, acc12, acc3 ); \
			if( FilterCount >= 4 && OutputCount >= 2 ) acc7 = _mm256_fmadd_ps( acc14, acc12, acc7 ); \
			if( FilterCount >= 4 && OutputCount >= 3 ) acc11 = _mm256_fmadd_ps( acc15, acc12, acc11 ); \
		}
*/

namespace NeoMLTest {

// This macro multiplies and accumulates for FilterCount by OutputCount block of the output buffer.
template<int FilterCount, int OutputCount, int VectorOffset, int BroadcastOffset>
void ComputeBlock( const float* input, const float* filter, int filterStride, const float* shiftedFilter, int strideWidth,
	__m256& acc0, __m256& acc1, __m256& acc2, __m256& acc3, __m256& acc4, __m256& acc5,
	__m256& acc6, __m256& acc7, __m256& acc8, __m256& acc9, __m256& acc10, __m256& acc11 )
{
	__m256 acc12, acc13, acc14, acc15;
	if( OutputCount >= 1 )  acc13 = _mm256_broadcastss_ps( _mm_set1_ps( *( input + BroadcastOffset ) ) );
	if( OutputCount >= 2 )  acc14 = _mm256_broadcastss_ps( _mm_set1_ps( *( input + strideWidth + BroadcastOffset ) ) );
	if( OutputCount >= 3 )  acc15 = _mm256_broadcastss_ps( _mm_set1_ps( *( input + strideWidth * 2 + BroadcastOffset ) ) );

	if( OutputCount == 1 ) {
		if( FilterCount >= 1 ) acc0 = _mm256_fmadd_ps( acc13, _mm256_loadu_ps( filter + VectorOffset ), acc0 );
		if( FilterCount >= 2 ) acc1 = _mm256_fmadd_ps( acc13, _mm256_loadu_ps( filter + filterStride + VectorOffset ), acc1 );
		if( FilterCount >= 3 ) acc2 = _mm256_fmadd_ps( acc13, _mm256_loadu_ps( shiftedFilter + VectorOffset ), acc2 );
		if( FilterCount >= 4 ) acc3 = _mm256_fmadd_ps( acc13, _mm256_loadu_ps( shiftedFilter + filterStride + VectorOffset ), acc3 );
	} else {
		if( FilterCount >= 1 ) acc12 = _mm256_loadu_ps( filter + VectorOffset );
		if( FilterCount >= 1 && OutputCount >= 1 ) acc0 = _mm256_fmadd_ps( acc13, acc12, acc0 );
		if( FilterCount >= 1 && OutputCount >= 2 ) acc4 = _mm256_fmadd_ps( acc14, acc12, acc4 );
		if( FilterCount >= 1 && OutputCount >= 3 ) acc8 = _mm256_fmadd_ps( acc15, acc12, acc8 );

		if( FilterCount >= 2 ) acc12 = _mm256_loadu_ps( filter + filterStride + VectorOffset );
		if( FilterCount >= 2 && OutputCount >= 1 ) acc1 = _mm256_fmadd_ps( acc13, acc12, acc1 );
		if( FilterCount >= 2 && OutputCount >= 2 ) acc5 = _mm256_fmadd_ps( acc14, acc12, acc5 );
		if( FilterCount >= 2 && OutputCount >= 3 ) acc9 = _mm256_fmadd_ps( acc15, acc12, acc9 );

		if( FilterCount >= 3 ) acc12 = _mm256_loadu_ps( shiftedFilter + VectorOffset );
		if( FilterCount >= 3 && OutputCount >= 1 ) acc2 = _mm256_fmadd_ps( acc13, acc12, acc2 );
		if( FilterCount >= 3 && OutputCount >= 2 ) acc6 = _mm256_fmadd_ps( acc14, acc12, acc6 );
		if( FilterCount >= 3 && OutputCount >= 3 ) acc10 = _mm256_fmadd_ps( acc15, acc12, acc10 );

		if( FilterCount >= 4 ) acc12 = _mm256_loadu_ps( shiftedFilter + filterStride + VectorOffset );
		if( FilterCount >= 4 && OutputCount >= 1 ) acc3 = _mm256_fmadd_ps( acc13, acc12, acc3 );
		if( FilterCount >= 4 && OutputCount >= 2 ) acc7 = _mm256_fmadd_ps( acc14, acc12, acc7 );
		if( FilterCount >= 4 && OutputCount >= 3 ) acc11 = _mm256_fmadd_ps( acc15, acc12, acc11 );
	}
}

/*
;   This macro generates code to process an output block after the inner
;   convolution kernel has executed and then stores the output block to the
;   output buffer.
*/
template<int FilterCount, int OutputCount>
void PostProcessing( const int flags, float*& output, int outputStride, const float* bias,
	__m256& acc0, __m256& acc1, __m256& acc2, __m256& acc3, __m256& acc4, __m256& acc5,
	__m256& acc6, __m256& acc7, __m256& acc8, __m256& acc9, __m256& acc10, __m256& acc11 )
{
	float* shiftedOutput = nullptr;
	if( FilterCount > 2 ) shiftedOutput = output + 2 * outputStride;

	if( ( flags & ACCUMULATE_OUTPUT ) != 0 ) {
		if( FilterCount >= 1 && OutputCount >= 1 ) acc0 = _mm256_add_ps( acc0, _mm256_loadu_ps( output ) );
		if( FilterCount >= 1 && OutputCount >= 2 ) acc4 = _mm256_add_ps( acc4, _mm256_loadu_ps( output + 8 ) );
		if( FilterCount >= 1 && OutputCount >= 3 ) acc8 = _mm256_add_ps( acc8, _mm256_loadu_ps( output + 16 ) );

		if( FilterCount >= 2 && OutputCount >= 1 ) acc1 = _mm256_add_ps( acc1, _mm256_loadu_ps( output + outputStride ) );
		if( FilterCount >= 2 && OutputCount >= 2 ) acc5 = _mm256_add_ps( acc5, _mm256_loadu_ps( output + outputStride + 8 ) );
		if( FilterCount >= 2 && OutputCount >= 3 ) acc9 = _mm256_add_ps( acc9, _mm256_loadu_ps( output + outputStride + 16 ) );

		if( FilterCount >= 3 && OutputCount >= 1 ) acc2 = _mm256_add_ps( acc2, _mm256_loadu_ps( shiftedOutput ) );
		if( FilterCount >= 3 && OutputCount >= 2 ) acc6 = _mm256_add_ps( acc6, _mm256_loadu_ps( shiftedOutput + 8 ) );
		if( FilterCount >= 3 && OutputCount >= 3 ) acc10 = _mm256_add_ps( acc10, _mm256_loadu_ps( shiftedOutput + 16 ) );

		if( FilterCount >= 4 && OutputCount >= 1 ) acc3 = _mm256_add_ps( acc3, _mm256_loadu_ps( shiftedOutput + outputStride ) );
		if( FilterCount >= 4 && OutputCount >= 2 ) acc7 = _mm256_add_ps( acc7, _mm256_loadu_ps( shiftedOutput + outputStride + 8 ) );
		if( FilterCount >= 4 && OutputCount >= 3 ) acc11 = _mm256_add_ps( acc11, _mm256_loadu_ps( shiftedOutput + outputStride + 16 ) );
	}

	if( ( flags & ADD_BIAS ) != 0 ) {
		if( OutputCount == 1 ) {
			if( FilterCount >= 1 ) acc0 = _mm256_add_ps( acc0, _mm256_loadu_ps( bias ) );
			if( FilterCount >= 2 ) acc1 = _mm256_add_ps( acc1, _mm256_loadu_ps( bias + 8 ) );
			if( FilterCount >= 3 ) acc2 = _mm256_add_ps( acc2, _mm256_loadu_ps( bias + 16 ) );
			if( FilterCount >= 4 ) acc3 = _mm256_add_ps( acc3, _mm256_loadu_ps( bias + 24 ) );
		} else {
			__m256 acc12, acc13, acc14, acc15;
			if( FilterCount >= 1 ) acc12 = _mm256_loadu_ps( bias );
			if( FilterCount >= 2 ) acc13 = _mm256_loadu_ps( bias + 8 );
			if( FilterCount >= 3 ) acc14 = _mm256_loadu_ps( bias + 16 );
			if( FilterCount >= 4 ) acc15 = _mm256_loadu_ps( bias + 24 );

			if( FilterCount >= 1 && OutputCount >= 1 ) acc0 = _mm256_add_ps( acc0, acc12 );
			if( FilterCount >= 1 && OutputCount >= 2 ) acc4 = _mm256_add_ps( acc4, acc12 );
			if( FilterCount >= 1 && OutputCount >= 3 ) acc8 = _mm256_add_ps( acc8, acc12 );

			if( FilterCount >= 2 && OutputCount >= 1 ) acc1 = _mm256_add_ps( acc1, acc13 );
			if( FilterCount >= 2 && OutputCount >= 2 ) acc5 = _mm256_add_ps( acc5, acc13 );
			if( FilterCount >= 2 && OutputCount >= 3 ) acc9 = _mm256_add_ps( acc9, acc13 );

			if( FilterCount >= 3 && OutputCount >= 1 ) acc2 = _mm256_add_ps( acc2, acc14 );
			if( FilterCount >= 3 && OutputCount >= 2 ) acc6 = _mm256_add_ps( acc6, acc14 );
			if( FilterCount >= 3 && OutputCount >= 3 ) acc10 = _mm256_add_ps( acc10, acc14 );

			if( FilterCount >= 4 && OutputCount >= 1 ) acc3 = _mm256_add_ps( acc3, acc15 );
			if( FilterCount >= 4 && OutputCount >= 2 ) acc7 = _mm256_add_ps( acc7, acc15 );
			if( FilterCount >= 4 && OutputCount >= 3 ) acc11 = _mm256_add_ps( acc11, acc15 );
		}
	}

	if( FilterCount >= 1 && OutputCount >= 1 ) _mm256_storeu_ps( output, acc0 );
	if( FilterCount >= 1 && OutputCount >= 2 ) _mm256_storeu_ps( output + 8, acc4 );
	if( FilterCount >= 1 && OutputCount >= 3 ) _mm256_storeu_ps( output + 16, acc8 );

	if( FilterCount >= 2 && OutputCount >= 1 ) _mm256_storeu_ps( output + outputStride, acc1 );
	if( FilterCount >= 2 && OutputCount >= 2 ) _mm256_storeu_ps( output + outputStride + 8, acc5 );
	if( FilterCount >= 2 && OutputCount >= 3 ) _mm256_storeu_ps( output + outputStride + 16, acc9 );

	if( FilterCount >= 3 && OutputCount >= 1 ) _mm256_storeu_ps( shiftedOutput, acc2 );
	if( FilterCount >= 3 && OutputCount >= 2 ) _mm256_storeu_ps( shiftedOutput + 8, acc6 );
	if( FilterCount >= 3 && OutputCount >= 3 ) _mm256_storeu_ps( shiftedOutput + 16, acc10 );

	if( FilterCount >= 4 && OutputCount >= 1 ) _mm256_storeu_ps( shiftedOutput + outputStride, acc3 );
	if( FilterCount >= 4 && OutputCount >= 2 ) _mm256_storeu_ps( shiftedOutput + outputStride + 8, acc7 );
	if( FilterCount >= 4 && OutputCount >= 3 ) _mm256_storeu_ps( shiftedOutput + outputStride + 16, acc11 );

	output += OutputCount * 8;
}

// This macro generates code to clear the block accumulators.
template<int FilterCount, int OutputCount>
void ClearBlock( __m256& acc0, __m256& acc1, __m256& acc2, __m256& acc3, __m256& acc4, __m256& acc5,
	__m256& acc6, __m256& acc7, __m256& acc8, __m256& acc9, __m256& acc10, __m256& acc11 )
{
	if( FilterCount >= 1 && OutputCount >= 1 ) acc0 = _mm256_xor_ps( acc0, acc0 );
	if( FilterCount >= 1 && OutputCount >= 2 ) acc4 = _mm256_xor_ps( acc0, acc0 );
	if( FilterCount >= 1 && OutputCount >= 3 ) acc8 = _mm256_xor_ps( acc0, acc0 );
	if( FilterCount >= 2 && OutputCount >= 1 ) acc1 = _mm256_xor_ps( acc0, acc0 );
	if( FilterCount >= 2 && OutputCount >= 2 ) acc5 = _mm256_xor_ps( acc0, acc0 );
	if( FilterCount >= 2 && OutputCount >= 3 ) acc9 = _mm256_xor_ps( acc0, acc0 );
	if( FilterCount >= 3 && OutputCount >= 1 ) acc2 = _mm256_xor_ps( acc0, acc0 );
	if( FilterCount >= 3 && OutputCount >= 2 ) acc6 = _mm256_xor_ps( acc0, acc0 );
	if( FilterCount >= 3 && OutputCount >= 3 ) acc10 = _mm256_xor_ps( acc0, acc0 );
	if( FilterCount >= 4 && OutputCount >= 1 ) acc3 = _mm256_xor_ps( acc0, acc0 );
	if( FilterCount >= 4 && OutputCount >= 2 ) acc7 = _mm256_xor_ps( acc0, acc0 );
	if( FilterCount >= 4 && OutputCount >= 3 ) acc11 = _mm256_xor_ps( acc0, acc0 );
}

/*
;   This macro generates code to compute the convolution for a vector of input
;   blocks and a vector of filter blocks to produce a matrix of output blocks.
;
;   OutputCount=1 generates special case code to handle padding blocks. All
;   other output counts assume no padding.
*/
template<int FilterCount, int OutputCount>
void ProcessOutputCountN( const KernelFrame& frame, const float* input, int filterStride,
	int dilationWidth, float*& output, int strideWidth, int inputStride )
{
	const float* filter = frame.Filter;

	__m256 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11;
	ClearBlock<FilterCount, OutputCount>( acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );

	const float* r13;
	if( OutputCount == 1 ) r13 = frame.InputBase;

	// rax = r12
	for( int row = 0; row < frame.KernelHeight; ++row ) {
		for( int col = 0; col < frame.KernelWidth; ++col ) {
			if( OutputCount != 1 || size_t(input) - size_t(r13) < size_t(frame.InputWidth * sizeof(float)) ) {
				ComputeBlock<FilterCount, OutputCount, 0 * 8, 0>(input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11);
				ComputeBlock<FilterCount, OutputCount, 1 * 8, 1>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				ComputeBlock<FilterCount, OutputCount, 2 * 8, 2>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				ComputeBlock<FilterCount, OutputCount, 3 * 8, 3>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				ComputeBlock<FilterCount, OutputCount, 4 * 8, 4>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				ComputeBlock<FilterCount, OutputCount, 5 * 8, 5>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				ComputeBlock<FilterCount, OutputCount, 6 * 8, 6>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				ComputeBlock<FilterCount, OutputCount, 7 * 8, 7>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				/*ComputeBlock<FilterCount, OutputCount, ( 0 - 4 ) * 8, 0>(input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11);
				ComputeBlock<FilterCount, OutputCount, ( 1 - 4 ) * 8, 1>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				ComputeBlock<FilterCount, OutputCount, ( 2 - 4 ) * 8, 2>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				ComputeBlock<FilterCount, OutputCount, ( 3 - 4 ) * 8, 3>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				ComputeBlock<FilterCount, OutputCount, ( 4 - 4 ) * 8, 4>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				ComputeBlock<FilterCount, OutputCount, ( 5 - 4 ) * 8, 5>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				ComputeBlock<FilterCount, OutputCount, ( 6 - 4 ) * 8, 6>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
				ComputeBlock<FilterCount, OutputCount, ( 7 - 4 ) * 8, 7>( input, filter, filterStride, filter + 2 * filterStride, strideWidth, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );*/
			}

			input += dilationWidth;
			filter += 8 * 8;
		}
		input += inputStride;
		if( OutputCount == 1 ) r13 += frame.DilatedInputWidth;
	}

	PostProcessing<FilterCount, OutputCount>( frame.Flags, output, frame.OutputStride, frame.Bias,
		acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 );
}

template<int FilterCount>
void SingleKernel( const KernelFrame& frame, const float*& input, int filterStride,
	int dilationWidth, float*& output, int strideWidth, int inputStride, int outputCount )
{
	for( int i = 0; i < outputCount; ++i ) {
		ProcessOutputCountN<FilterCount, 1>( frame, input, filterStride, dilationWidth, output, strideWidth, inputStride );
		input += strideWidth;
	}
}

template<int FilterCount>
void ProcessFilterCountN( const KernelFrame& frame, const float*& input, int filterStride,
	int dilationWidth, float*& output, int strideWidth, int inputStride )
{
	if( frame.OutputCountLeftPad > 0 ) {
		SingleKernel<FilterCount>( frame, input, filterStride, dilationWidth, output, strideWidth, inputStride, frame.OutputCountLeftPad );
	}

	int remOutputCount = frame.OutputCount;
	while( remOutputCount > 3 ) {
		ProcessOutputCountN<FilterCount, 3>( frame, input, filterStride, dilationWidth, output, strideWidth, inputStride );
		input += 3 * strideWidth;
		remOutputCount -= 3;
	}

	if( remOutputCount >= 2 ) {
		ProcessOutputCountN<FilterCount, 2>( frame, input, filterStride, dilationWidth, output, strideWidth, inputStride );
		input += 2 * strideWidth;
		remOutputCount -= 2;
	}

	remOutputCount += frame.OutputCountRightPad;
	if( remOutputCount > 0 ) {
		SingleKernel<FilterCount>( frame, input, filterStride, dilationWidth, output, strideWidth, inputStride, remOutputCount );
	}
}

void ConvKernelFunction( const KernelFrame& frame, int filterCount )
{
	const float* input = frame.Input;
	float* output = frame.Output;
	switch( filterCount ) {
		case 1:
			ProcessFilterCountN<1>( frame, input, frame.FilterStride, frame.DilationWidth, output, frame.StrideWidth, frame.InputStride );
			return;
		case 2:
			ProcessFilterCountN<2>( frame, input, frame.FilterStride, frame.DilationWidth, output, frame.StrideWidth, frame.InputStride );
			return;
		case 3:
			ProcessFilterCountN<3>( frame, input, frame.FilterStride, frame.DilationWidth, output, frame.StrideWidth, frame.InputStride );
			return;
		case 4:
			ProcessFilterCountN<4>( frame, input, frame.FilterStride, frame.DilationWidth, output, frame.StrideWidth, frame.InputStride );
	}
}

void RunConv( const float* Input, const float* OrigFilter, const float* OrigBias, float* Output,
	int batch, int hIn, int wIn, int chIn, int hOut, int wOut, int chOut,
	int hKer, int wKer, int hStride, int wStride,
	int hPad, int wPad, int hDil, int wDil )
{
	const int hSpan = ( hKer - 1 ) * hDil + 1;
	const int hOutCountWithLeftPad = ( hIn + hPad >= hSpan ) ? ( hIn + hPad - hSpan ) / hStride + 1 : 0;
	const int hOutCountLeftPad = std::min( hOutCountWithLeftPad, ( hPad + hStride - 1 ) / hStride );
	const int hOutCount = hOutCountWithLeftPad - hOutCountLeftPad;
	const int hOutCountRightPad = hOut - hOutCountWithLeftPad;

	const int wSpan = ( wKer - 1 ) * wDil + 1;
	const int wOutCountWithLeftPad = ( wIn + wPad >= wSpan ) ? ( wIn + wPad - wSpan ) / wStride + 1 : 0;
	const int wOutCountLeftPad = std::min( wOutCountWithLeftPad, ( wPad + wStride - 1 ) / wStride );
	const int wOutCount = wOutCountWithLeftPad - wOutCountLeftPad;
	const int wOutCountRightPad = wOut - wOutCountWithLeftPad;

	const int filterSetCount = ( chOut + 31 ) / 32;
	const int totalWork = batch * filterSetCount * hOut;
	int workIndex = 0;
	int workRem = totalWork;
	int ph = workIndex % hOut;
	int filterSet = ( workIndex / hOut ) % filterSetCount;
	int b = ( workIndex / hOut ) / filterSetCount;

	Input += b * chIn * hIn * wIn;
	
	Output += b * chOut * hOut * wOut;
	Output += 8 * filterSet * 4 * hOut * wOut;

	const float* Filter = OrigFilter + 8 * filterSet * 4 * chIn * hKer * wKer;
	
	const float* Bias = OrigBias + 8 * filterSet * 4;

	int filterCount = std::min( 4, ( chOut / 8 ) - filterSet * 4 );

	const int strideWidth = 8 * wStride;
	const int dilationWidth = 8 * wDil;
	const int filterStride = 8 * chIn * hKer * wKer;
	const int outputStride = 8 * hOut * wOut;
	const int inputWidth = 8 * wIn;
	const int dilatedInputWidth = 8 * hDil * wIn;
	const int inputStride = dilatedInputWidth - wKer * dilationWidth;

	const int blockOutputWidth = 8 * wOut;

	while( workRem > 0 ) {
		const int workThisIter = std::min( workRem, hOut - ph );
		for( int ic = 0; ic < chIn; ic += 8 ) {
			int flags = ic == 0 ? 0 : ACCUMULATE_OUTPUT;

			if( ic + 8 == chIn ) {
				flags |= ADD_BIAS;
			}

			const float* input = Input + ic * hIn * wIn;
			float* output = Output + ph * blockOutputWidth;

			for( int work = 0; work < workThisIter; ++work ) {
				const float* filter = Filter + 8 * ic * hKer * wKer;
				size_t ih = ( ph + work ) * hStride - hPad;
				size_t effectiveKernelHeight = hKer;

				if( ( size_t(ph) + work ) - size_t(hOutCountLeftPad) >= size_t(hOutCount) ) {
					size_t ihStep = ih;
					for( int kh = 0; kh < hKer; ++kh ) {
						if( ihStep >= size_t(hIn) ) {
							if( ihStep == ih ) {
								ih += hDil;
								filter += 8 * 8 * wKer;
							}
							effectiveKernelHeight--;
						}
						ihStep += hDil;
					}
				}

				KernelFrame frame;
				frame.Input = input + 8 * ( ih * wIn - wPad );
				frame.Filter = filter;
				frame.Output = output;
				frame.StrideWidth = strideWidth;
				frame.DilationWidth = dilationWidth;
				frame.FilterCount = filterCount;
				frame.InputStride = inputStride;
				frame.FilterStride = filterStride;
				frame.OutputStride = outputStride;
				frame.KernelHeight = effectiveKernelHeight;
				frame.KernelWidth = wKer;
				frame.InputBase = input + 8 * ( ih * wIn );
				frame.InputWidth = inputWidth;
				frame.DilatedInputWidth = dilatedInputWidth;
				frame.OutputCountLeftPad = wOutCountLeftPad;
				frame.OutputCount = wOutCount;
				frame.OutputCountRightPad = wOutCountRightPad;
				frame.Bias = Bias;
				frame.Flags = flags;

				ConvKernelFunction( frame, filterCount );

				output += blockOutputWidth;
			}
		}

		workRem -= workThisIter;
		if( ph + workThisIter == hOut ) {
			int blockedFilterCount = 8 * filterCount;
			Output += blockedFilterCount * hOut * wOut;
			Filter += blockedFilterCount * chIn * hKer * wKer;
			if( Bias != nullptr ) {
				Bias += blockedFilterCount;
			}

			if( ++filterSet == filterSetCount ) {
				Input += chIn * hIn * wIn;
				Filter = OrigFilter;
				Bias = OrigBias;
				filterSet = 0;
			}

			filterCount = std::min( 4, ( chOut / 8 ) - filterSet * 4 );
			ph = 0;
		}
	}
}

std::vector<float> PackData( const float* original, int batch, int height, int width, int channels )
{
	assert( channels % 8 == 0 );

	std::vector<float> result( batch * height * width * channels );

	for( int b = 0; b < batch; ++b ) {
		for( int h = 0; h < height; ++h ) {
			for( int w = 0; w < width; ++w ) {
				for( int C = 0; C < channels / 8; ++C ) {
					for( int c = 0; c < 8; ++c ) {
						const int idx = c + 8 * ( w + width * ( h + height * ( C + channels / 8 * b ) ) );
						result[idx] = *original;
						original++;
					}
				}
			}
		}
	}

	return result;
}

std::vector<float> UnpackData( const float* original, int batch, int height, int width, int channels )
{
	assert( channels % 8 == 0 );

	std::vector<float> result( batch * height * width * channels );

	for( int b = 0; b < batch; ++b ) {
		for( int C = 0; C < channels / 8; ++C ) {
			for( int h = 0; h < height; ++h ) {
				for( int w = 0; w < width; ++w ) {
					for( int c = 0; c < 8; ++c ) {
						const int idx = c + 8 * ( C + channels / 8 * ( w + width * ( h + height * b ) ) );
						result[idx] = *original;
						original++;
					}
				}
			}
		}
	}

	return result;
}

std::vector<float> PackFilter( const float* original, int filterCount, int height, int width, int channels )
{
	assert( filterCount % 8 == 0 );
	assert( channels % 8 == 0 );

	std::vector<float> result( filterCount * height * width * channels );

	for( int O = 0; O < filterCount / 8; ++O ) {
		for( int o = 0; o < 8; ++o ) {
			for( int h = 0; h < height; ++h ) {
				for( int w = 0; w < width; ++w ) {
					for( int I = 0; I < channels / 8; ++I ) {
						for( int i = 0; i < 8; ++i ) {
							const int idx = o + 8 * ( i + 8 * ( w + width * ( h + height * ( I + channels / 8 * O ) ) ) );
							result[idx] = *original;
							original++;
						}
					}
				}
			}
		}
	}

	return result;
}

} // namespace NeoMLTest

class CBlockedConvTest : public CTestFixtureWithParams {
};



TEST_P( CBlockedConvTest, Run )
{
	const CTestParams& params = GetParam();
	CRandom random( params.GetValue<int>( "Seed" ) );
	const int batch = params.GetValue<int>( "Batch" );
	const int height = params.GetValue<int>( "Height" );
	const int width = params.GetValue<int>( "Width" );
	const int channels = params.GetValue<int>( "Channels" );
	const int filterCount = params.GetValue<int>( "FilterCount" );
	const int filterHeight = params.GetValue<int>( "FilterHeight" );
	const int filterWidth = params.GetValue<int>( "FilterWidth" );
	const int strideHeight = params.GetValue<int>( "StrideHeight" );
	const int strideWidth = params.GetValue<int>( "StrideWidth" );
	const int paddingHeight = params.GetValue<int>( "PaddingHeight" );
	const int paddingWidth = params.GetValue<int>( "PaddingWidth" );
	const int dilationHeight = params.GetValue<int>( "DilationHeight" );
	const int dilationWidth = params.GetValue<int>( "DilationWidth" );

	const int outputHeight = calcConvOutputSize( height, paddingHeight, filterHeight, dilationHeight, strideHeight );
	const int outputWidth = calcConvOutputSize( width, paddingWidth, filterWidth, dilationWidth, strideWidth );

	CREATE_FILL_FLOAT_ARRAY( neomlInput, -2.f, 2.f, batch * height * width * channels, random );
	CBlobDesc inputDesc( { 1, batch, 1, height, width, 1, channels } );

	CREATE_FILL_FLOAT_ARRAY( neomlFilter, -2.f, 2.f, filterCount * filterHeight * filterWidth * channels, random );
	CBlobDesc filterDesc( { 1, filterCount, 1, filterHeight, filterWidth, 1, channels } );

	CREATE_FILL_FLOAT_ARRAY( bias, -5.f, 5.f, filterCount, random );

	CBlobDesc outputDesc( { 1, batch, 1, outputHeight, outputWidth, 1, filterCount } );
	std::vector<float> expectedOutput( batch * outputHeight * outputWidth * filterCount );

	{
		CConvolutionDesc* convDesc = MathEngine().InitBlobConvolution( inputDesc, paddingHeight, paddingWidth, strideHeight,
			strideWidth, dilationHeight, dilationWidth, filterDesc, outputDesc );
		auto biasHandle = CARRAY_FLOAT_WRAPPER( bias );
		MathEngine().BlobConvolution( *convDesc, CARRAY_FLOAT_WRAPPER( neomlInput ), CARRAY_FLOAT_WRAPPER( neomlFilter ),
			&static_cast<CConstFloatHandle>( biasHandle ), CARRAY_FLOAT_WRAPPER( expectedOutput ) );
	}

	std::vector<float> blockedInput = PackData( neomlInput.data(), batch, height, width, channels );
	std::vector<float> blockedFilter = PackFilter( neomlFilter.data(), filterCount, filterHeight, filterWidth, channels );
	std::vector<float> blockedOutput( expectedOutput.size() );

	RunConv( blockedInput.data(), blockedFilter.data(), bias.data(), blockedOutput.data(), batch, height, width, channels, outputHeight, outputWidth,
		filterCount, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, dilationHeight, dilationWidth );
	std::vector<float> actualOutput = UnpackData( blockedOutput.data(), batch, outputHeight, outputWidth, filterCount );

	for( size_t i = 0; i < actualOutput.size(); ++i ) {
		if( ::fabsf( actualOutput[i] - expectedOutput[i] ) > 1e-2f ) {
			//__debugbreak();
		}
		ASSERT_NEAR( actualOutput[i], expectedOutput[i], 1e-2f ) << "at #" << i;
	}
}

struct BlockedConvTestNameGenerator {
	std::string operator()(const testing::TestParamInfo<CTestParams>& paramInfo)
	{
		const CTestParams& params = paramInfo.param;
		if( !params.Name().empty() ) {
			return params.Name();
		}

		const int batch = params.GetValue<int>( "Batch" );
		const int height = params.GetValue<int>( "Height" );
		const int width = params.GetValue<int>( "Width" );
		const int channels = params.GetValue<int>( "Channels" );
		const int filterCount = params.GetValue<int>( "FilterCount" );
		const int filterHeight = params.GetValue<int>( "FilterHeight" );
		const int filterWidth = params.GetValue<int>( "FilterWidth" );
		const int strideHeight = params.GetValue<int>( "StrideHeight" );
		const int strideWidth = params.GetValue<int>( "StrideWidth" );
		const int paddingHeight = params.GetValue<int>( "PaddingHeight" );
		const int paddingWidth = params.GetValue<int>( "PaddingWidth" );
		const int dilationHeight = params.GetValue<int>( "DilationHeight" );
		const int dilationWidth = params.GetValue<int>( "DilationWidth" );


		const int bufferSize = 1024;
		int currLen = 0;
		std::vector<char> buffer;
		buffer.resize( 1024 );

		currLen += ::sprintf_s( buffer.data() + currLen, bufferSize - currLen, "I_%dx%dx%dx%d__", batch, height, width, channels );
		currLen += ::sprintf_s( buffer.data() + currLen, bufferSize - currLen, "F_%dx%dx%d__", filterCount, filterHeight, filterWidth );
		currLen += ::sprintf_s( buffer.data() + currLen, bufferSize - currLen, "St_%dx%d__", strideHeight, strideWidth );
		currLen += ::sprintf_s( buffer.data() + currLen, bufferSize - currLen, "P_%dx%d__", paddingHeight, paddingWidth );
		currLen += ::sprintf_s( buffer.data() + currLen, bufferSize - currLen, "D_%dx%d", dilationHeight, dilationWidth );

		return std::string( buffer.data(), currLen );
	}
};

INSTANTIATE_TEST_SUITE_P( Trivial, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Minimal"
		),
		CTestParams(
			"Batch = 2;"
			"Height = 1;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Batch"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 2;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"InputHeight"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 2;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"InputWidth"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 1;"
			"Channels = 16;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"InputChannels"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 16;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"FilterCount"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 2;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 2;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"FilterHeight"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 2;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 2;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"FilterWidth"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 2;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 3;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"PaddingHeight"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 3;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"TrickyPaddingHeight"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 2;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"PaddingWidth"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"TrickyPaddingWidth"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 3;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"StrideHeight"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 3;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 2;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"StrideWidth"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 3;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 2;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 2;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"DilationHeight"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 3;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 2;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 2;"
			"Seed = 348;",
			"DilationWidth"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P(Yolox0, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 320;"
			"Width = 640;"
			"Channels = 32;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_16_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 160;"
			"Width = 320;"
			"Channels = 64;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_19_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 160;"
			"Width = 320;"
			"Channels = 32;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_25_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 160;"
			"Width = 320;"
			"Channels = 32;"
			"FilterCount = 32;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_28_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 160;"
			"Width = 320;"
			"Channels = 64;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_22_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 160;"
			"Width = 320;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_33_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 160;"
			"Width = 320;"
			"Channels = 64;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_36_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_39_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_45_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_48_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_52_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_55_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_59_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_62_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_42_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_67_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 256;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_70_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_73_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_79_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_82_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_86_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_89_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_93_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_96_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_76_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_101_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 512;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_104_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_107_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 1024;"
			"FilterCount = 512;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_114_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_117_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_123_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_126_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_120_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 512;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_130_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_133_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 512;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_138_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_144_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_147_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 512;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_141_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_151_Op"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P(Yolox1, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_154_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 256;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_159_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_165_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_168_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 256;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_162_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_172_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_215_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_218_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_221_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_175_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_179_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_185_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_188_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_182_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_192_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_233_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_236_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_239_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_195_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_199_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_205_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_208_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_202_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 512;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_212_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_251_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_254_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_257_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_224_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_227_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_242_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_245_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_260_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_263_Op"
		)
	), BlockedConvTestNameGenerator() );
