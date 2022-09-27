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

#include <immintrin.h>

#include <NeoMathEngine/SimdMathEngine.h>
#include <AvxMathEngine.h>

namespace NeoML {

struct CAvxBlockedConvDesc : public CConvolutionDesc {
	CAvxBlockedConvDesc( const CBlobDesc& source, const CBlobDesc& result, const CBlobDesc& filter, int strideHeight,
			int strideWidth, int paddingHeight, int paddingWidth, int dilationHeight, int dilationWidth ) :
		Source( source ),
		Result( result ),
		Filter( filter ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth ),
		PaddingHeight( paddingHeight ),
		PaddingWidth( paddingWidth ),
		DilationHeight( dilationHeight ),
		DilationWidth( dilationWidth )
	{
	}


	~CAvxBlockedConvDesc() override = default;

	const CBlobDesc Source;
	const CBlobDesc Result;
	const CBlobDesc Filter;
	const int StrideHeight;
	const int StrideWidth;
	const int PaddingHeight;
	const int PaddingWidth;
	const int DilationHeight;
	const int DilationWidth;
};

CConvolutionDesc* CAvxMathEngine::InitBlockedConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
	int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter,
	const CBlobDesc& result ) const
{
	const int filterCount = filter.ObjectCount();
	const int inputChannels = source.Depth() * source.Channels();
	if( filterCount % 8 == 0 && inputChannels % 8 == 0 ) {
		return new CAvxBlockedConvDesc( source, result, filter, strideHeight, strideWidth, paddingHeight, paddingWidth,
			dilationHeight, dilationWidth );
	}

	return nullptr;
}

void CAvxMathEngine::PackBlockedData( const CBlobDesc& desc, const float* source, float* result ) const
{
	const int channels = desc.Depth() * desc.Channels();
	assert( channels % 8 == 0 );

	const int height = desc.Height();
	const int width = desc.Width();
	const int geomSize = height * width;
	const int batch = desc.ObjectCount();

	for( int b = 0; b < batch; ++b ) {
		for( int hw = 0; hw < geomSize; ++hw ) {
			float* channelResult = result + hw * 8;
			for( int C = 0; C < channels / 8; ++C ) {
				_mm256_storeu_ps( channelResult, _mm256_loadu_ps( source ) );
				source += 8;
				channelResult += geomSize * 8;
			}
		}
		result += channels * geomSize;
	}
}

void CAvxMathEngine::UnpackBlockedData( const CBlobDesc& desc, const float* source, float* result ) const
{
	const int channels = desc.Depth() * desc.Channels();
	assert( channels % 8 == 0 );

	const int height = desc.Height();
	const int width = desc.Width();
	const int geomSize = height * width;
	const int batch = desc.ObjectCount();

	for( int b = 0; b < batch; ++b ) {
		for( int C = 0; C < channels / 8; ++C ) {
			float* geomResult = result + C * 8;
			for( int hw = 0; hw < geomSize; ++hw ) {
				_mm256_storeu_ps( geomResult, _mm256_loadu_ps( source ) );
				source += 8;
				geomResult += channels;
			}
		}
		result += channels * geomSize;
	}
}

void CAvxMathEngine::PackBlockedFilter( const CBlobDesc& desc, const float* source, float* result ) const
{
	const int filterCount = desc.ObjectCount();
	const int channels = desc.Depth() * desc.Channels();
	const int height = desc.Height();
	const int width = desc.Width();
	assert( filterCount % 8 == 0 );
	assert( channels % 8 == 0 );

	for( int O = 0; O < filterCount / 8; ++O ) {
		for( int o = 0; o < 8; ++o ) {
			for( int h = 0; h < height; ++h ) {
				for( int w = 0; w < width; ++w ) {
					for( int I = 0; I < channels / 8; ++I ) {
						for( int i = 0; i < 8; ++i ) {
							const int idx = o + 8 * ( i + 8 * ( w + width * ( h + height * ( I + channels / 8 * O ) ) ) );
							result[idx] = *source;
							source++;
						}
					}
				}
			}
		}
	}
}

// Main part with convolution itself

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

// This macro multiplies and accumulates for FilterCount by OutputCount block of the output buffer.
#define COMPUTE_BLOCK(FilterCount, OutputCount, VectorOffset, BroadcastOffset) \
{ \
	if( OutputCount >= 1 )  acc13 = _mm256_set1_ps( *( input + BroadcastOffset ) ); \
	if( OutputCount >= 2 )  acc14 = _mm256_set1_ps( *( input + strideWidth + BroadcastOffset ) ); \
	if( OutputCount >= 3 )  acc15 = _mm256_set1_ps( *( input + strideWidth * 2 + BroadcastOffset ) ); \
	\
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
		\
		if( FilterCount >= 2 ) acc12 = _mm256_loadu_ps( filter + filterStride + VectorOffset ); \
		if( FilterCount >= 2 && OutputCount >= 1 ) acc1 = _mm256_fmadd_ps( acc13, acc12, acc1 ); \
		if( FilterCount >= 2 && OutputCount >= 2 ) acc5 = _mm256_fmadd_ps( acc14, acc12, acc5 ); \
		if( FilterCount >= 2 && OutputCount >= 3 ) acc9 = _mm256_fmadd_ps( acc15, acc12, acc9 ); \
		\
		if( FilterCount >= 3 ) acc12 = _mm256_loadu_ps( shiftedFilter + VectorOffset ); \
		if( FilterCount >= 3 && OutputCount >= 1 ) acc2 = _mm256_fmadd_ps( acc13, acc12, acc2 ); \
		if( FilterCount >= 3 && OutputCount >= 2 ) acc6 = _mm256_fmadd_ps( acc14, acc12, acc6 ); \
		if( FilterCount >= 3 && OutputCount >= 3 ) acc10 = _mm256_fmadd_ps( acc15, acc12, acc10 ); \
		\
		if( FilterCount >= 4 ) acc12 = _mm256_loadu_ps( shiftedFilter + filterStride + VectorOffset ); \
		if( FilterCount >= 4 && OutputCount >= 1 ) acc3 = _mm256_fmadd_ps( acc13, acc12, acc3 ); \
		if( FilterCount >= 4 && OutputCount >= 2 ) acc7 = _mm256_fmadd_ps( acc14, acc12, acc7 ); \
		if( FilterCount >= 4 && OutputCount >= 3 ) acc11 = _mm256_fmadd_ps( acc15, acc12, acc11 ); \
	} \
}
/*
;   This macro generates code to process an output block after the inner
;   convolution kernel has executed and then stores the output block to the
;   output buffer.
*/
template<int FilterCount, int OutputCount>
static void postProcessing( const int flags, float*& output, int outputStride, const float* bias,
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
#define CLEAR_BLOCK(FilterCount, OutputCount) \
{ \
	if( FilterCount >= 1 && OutputCount >= 1 ) acc0 = _mm256_set1_ps( 0 ); \
	if( FilterCount >= 1 && OutputCount >= 2 ) acc4 = _mm256_set1_ps( 0 ); \
	if( FilterCount >= 1 && OutputCount >= 3 ) acc8 = _mm256_set1_ps( 0 ); \
	if( FilterCount >= 2 && OutputCount >= 1 ) acc1 = _mm256_set1_ps( 0 ); \
	if( FilterCount >= 2 && OutputCount >= 2 ) acc5 = _mm256_set1_ps( 0 ); \
	if( FilterCount >= 2 && OutputCount >= 3 ) acc9 = _mm256_set1_ps( 0 ); \
	if( FilterCount >= 3 && OutputCount >= 1 ) acc2 = _mm256_set1_ps( 0 ); \
	if( FilterCount >= 3 && OutputCount >= 2 ) acc6 = _mm256_set1_ps( 0 ); \
	if( FilterCount >= 3 && OutputCount >= 3 ) acc10 = _mm256_set1_ps( 0 ); \
	if( FilterCount >= 4 && OutputCount >= 1 ) acc3 = _mm256_set1_ps( 0 ); \
	if( FilterCount >= 4 && OutputCount >= 2 ) acc7 = _mm256_set1_ps( 0 ); \
	if( FilterCount >= 4 && OutputCount >= 3 ) acc11 = _mm256_set1_ps( 0 ); \
}

/*
;   This macro generates code to compute the convolution for a vector of input
;   blocks and a vector of filter blocks to produce a matrix of output blocks.
;
;   OutputCount=1 generates special case code to handle padding blocks. All
;   other output counts assume no padding.
*/
#define PROCESS_OUTPUT_COUNT_N(FilterCount, OutputCount) \
{ \
	const float* prevInput = input; \
	const float* filter = frame.Filter; \
	\
	__m256 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11; \
	CLEAR_BLOCK(FilterCount, OutputCount); \
	\
	const float* r13; \
	if( OutputCount == 1 ) r13 = frame.InputBase; \
	\
	for( int row = 0; row < frame.KernelHeight; ++row ) { \
		for( int col = 0; col < frame.KernelWidth; ++col ) { \
			if( OutputCount != 1 || size_t(input) - size_t(r13) < size_t(frame.InputWidth * sizeof(float)) ) { \
				const float* shiftedFilter; \
				if( FilterCount >= 3 )  shiftedFilter = filter + 2 * filterStride; \
				__m256 acc12, acc13, acc14, acc15; \
				COMPUTE_BLOCK( FilterCount, OutputCount, 0 * 8, 0 ); \
				COMPUTE_BLOCK( FilterCount, OutputCount, 1 * 8, 1 ); \
				COMPUTE_BLOCK( FilterCount, OutputCount, 2 * 8, 2 ); \
				COMPUTE_BLOCK( FilterCount, OutputCount, 3 * 8, 3 ); \
				COMPUTE_BLOCK( FilterCount, OutputCount, 4 * 8, 4 ); \
				COMPUTE_BLOCK( FilterCount, OutputCount, 5 * 8, 5 ); \
				COMPUTE_BLOCK( FilterCount, OutputCount, 6 * 8, 6 ); \
				COMPUTE_BLOCK( FilterCount, OutputCount, 7 * 8, 7 ); \
			} \
			\
			input += dilationWidth; \
			filter += 8 * 8; \
		} \
		input += inputStride; \
		if( OutputCount == 1 ) r13 += frame.DilatedInputWidth; \
	} \
	\
	postProcessing<FilterCount, OutputCount>( frame.Flags, output, frame.OutputStride, frame.Bias, \
		acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11 ); \
	input = prevInput; \
}

template<int FilterCount>
static void singleKernel( const KernelFrame& frame, const float*& input, int filterStride,
	int dilationWidth, float*& output, int strideWidth, int inputStride, int outputCount )
{
	for( int i = 0; i < outputCount; ++i ) {
		PROCESS_OUTPUT_COUNT_N( FilterCount, 1 )
			input += strideWidth;
	}
}

#define PROCESS_FILTER_COUNT_N(FilterCount) \
{ \
	if( frame.OutputCountLeftPad > 0 ) { \
		singleKernel<FilterCount>( frame, input, filterStride, dilationWidth, output, strideWidth, inputStride, frame.OutputCountLeftPad ); \
	} \
	\
	int remOutputCount = frame.OutputCount; \
	while( remOutputCount > 3 ) { \
		PROCESS_OUTPUT_COUNT_N( FilterCount, 3 ) \
		input += 3 * strideWidth; \
		remOutputCount -= 3; \
	} \
	\
	if( remOutputCount >= 2 ) { \
		PROCESS_OUTPUT_COUNT_N( FilterCount, 2 ) \
		input += 2 * strideWidth; \
		remOutputCount -= 2; \
	} \
	\
	remOutputCount += frame.OutputCountRightPad; \
	if( remOutputCount > 0 ) { \
		singleKernel<FilterCount>( frame, input, filterStride, dilationWidth, output, strideWidth, inputStride, remOutputCount ); \
	} \
}

static void convKernelFunction( const KernelFrame& frame, int filterCount )
{
	const float* input = frame.Input;
	float* output = frame.Output;
	int filterStride = frame.FilterStride;
	int dilationWidth = frame.DilationWidth;
	int strideWidth = frame.StrideWidth;
	int inputStride = frame.InputStride;
	switch( filterCount ) {
		case 1:
			PROCESS_FILTER_COUNT_N( 1 );
			return;
		case 2:
			PROCESS_FILTER_COUNT_N( 2 );
			return;
		case 3:
			PROCESS_FILTER_COUNT_N( 3 );
			return;
		case 4:
			PROCESS_FILTER_COUNT_N( 4 );
	}
}

static void runConv( const float* Input, const float* OrigFilter, const float* OrigBias, float* Output,
	int batch, int hIn, int wIn, int chIn, int hOut, int wOut, int chOut,
	int hKer, int wKer, int hStride, int wStride,
	int hPad, int wPad, int hDil, int wDil )
{
	const int hSpan = ( hKer - 1 ) * hDil + 1;
	const int hOutCountWithLeftPad = ( hIn + hPad >= hSpan ) ? ( hIn + hPad - hSpan ) / hStride + 1 : 0;
	const int hOutCountLeftPad = min( hOutCountWithLeftPad, ( hPad + hStride - 1 ) / hStride );
	const int hOutCount = hOutCountWithLeftPad - hOutCountLeftPad;
	const int hOutCountRightPad = hOut - hOutCountWithLeftPad;

	const int wSpan = ( wKer - 1 ) * wDil + 1;
	const int wOutCountWithLeftPad = ( wIn + wPad >= wSpan ) ? ( wIn + wPad - wSpan ) / wStride + 1 : 0;
	const int wOutCountLeftPad = min( wOutCountWithLeftPad, ( wPad + wStride - 1 ) / wStride );
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

	int filterCount = min( 4, ( chOut / 8 ) - filterSet * 4 );

	const int strideWidth = 8 * wStride;
	const int dilationWidth = 8 * wDil;
	const int filterStride = 8 * chIn * hKer * wKer;
	const int outputStride = 8 * hOut * wOut;
	const int inputWidth = 8 * wIn;
	const int dilatedInputWidth = 8 * hDil * wIn;
	const int inputStride = dilatedInputWidth - wKer * dilationWidth;

	const int blockOutputWidth = 8 * wOut;

	while( workRem > 0 ) {
		const int workThisIter = min( workRem, hOut - ph );
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
				frame.KernelHeight = static_cast<int>( effectiveKernelHeight );
				frame.KernelWidth = wKer;
				frame.InputBase = input + 8 * ( ih * wIn );
				frame.InputWidth = inputWidth;
				frame.DilatedInputWidth = dilatedInputWidth;
				frame.OutputCountLeftPad = wOutCountLeftPad;
				frame.OutputCount = wOutCount;
				frame.OutputCountRightPad = wOutCountRightPad;
				frame.Bias = Bias;
				frame.Flags = flags;

				convKernelFunction( frame, filterCount );

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

			filterCount = min( 4, ( chOut / 8 ) - filterSet * 4 );
			ph = 0;
		}
	}
}

void CAvxMathEngine::BlockedConvolution( const CConvolutionDesc& convDesc, const float* packedSource,
	const float* packedFilter, const float* freeTerm, float* packedResult ) const
{
	const CAvxBlockedConvDesc& desc = static_cast<const CAvxBlockedConvDesc&>( convDesc );
	runConv( packedSource, packedFilter, freeTerm, packedResult, desc.Source.ObjectCount(), desc.Source.Height(),
		desc.Source.Width(), desc.Source.Depth() * desc.Source.Channels(), desc.Result.Height(), desc.Result.Width(),
		desc.Filter.ObjectCount(), desc.Filter.Height(), desc.Filter.Width(), desc.StrideHeight, desc.StrideWidth,
		desc.PaddingHeight, desc.PaddingWidth, desc.DilationHeight, desc.DilationWidth );
}

} // namespace NeoML
