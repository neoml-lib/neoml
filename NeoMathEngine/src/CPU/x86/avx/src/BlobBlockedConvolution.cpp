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
#include <cstddef>

#include <NeoMathEngine/SimdMathEngine.h>
#include <AvxMathEngine.h>
#include <JitCommon.h>

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

// --------------------------------------------------------------------------------------------------------------------

// Main part with convolution itself

struct CJitGen {
	CJitCommon gen;
	std::mutex lock;
};

std::array<CJitGen, static_cast<size_t>( 4 * 3 )> computeBlockGens;

struct JitCallParams {
	const float* Input;
	size_t StrideWidth;
	const float* Filter;
	size_t FilterStride;
	const float* InputBase;
	size_t InputWidth;
	size_t KernelHeight;
	size_t KernelWidth;
	size_t DilationWidth;
	size_t DilatedInputWidth;
	size_t InputStride;
	const float* Bias;
	float* Output;
	size_t OutputStride;
	size_t Flags;
};

#define ACCUMULATE_OUTPUT 1
#define ADD_BIAS 2

// Clears the block accumulators.
static void clearUsedYmms( int filterCount, int outputCount, Xbyak::CodeGenerator& gen )
{
	for( int filter = 0; filter < filterCount; ++filter ) {
		for( int output = 0; output < outputCount; ++output ) {
			gen.vxorps( Xbyak::Ymm( output * 4 + filter ), Xbyak::Ymm( output * 4 + filter ) );
		}
	}
}

static void initComputeBlocks( int filterCount, int outputCount )
{
	using namespace Xbyak;
	using namespace Xbyak::util;

	const int genIndex = ( outputCount - 1 ) + 3 * ( filterCount - 1 );
	CJitCommon& gen = computeBlockGens[genIndex].gen;
	CodeGenerator& baseGen = static_cast<CodeGenerator&>( gen );

	ymmVec_t acc( 16 );
	for( int i = 0; i < 16; ++i ) {
		acc[i] = ymm_t( i );
	}

	reg64Vec_t preservedGPR = { rax, rbx, rcx, rdx, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15 };
	preservedGPR.erase( std::find( preservedGPR.begin(), preservedGPR.end(), Param1 ) );
	int regUsed = 0;

	Address stackArgsPtr = gen.Prologue( preservedGPR, acc );

	const reg64_t regNotUsed = rax;

	const reg64_t regFramePtr = Param1;

	const reg64_t regInput = preservedGPR[regUsed++];
	gen.mov( regInput, ptr[regFramePtr + offsetof( JitCallParams, Input )] );
	const reg64_t regStrideWidth = preservedGPR[regUsed++];
	gen.mov( regStrideWidth, ptr[regFramePtr + offsetof( JitCallParams, StrideWidth )] );
	const reg64_t regFilter = preservedGPR[regUsed++];
	gen.mov( regFilter, ptr[regFramePtr + offsetof( JitCallParams, Filter )] );
	const reg64_t regFilterStride = preservedGPR[regUsed++];
	gen.mov( regFilterStride, ptr[regFramePtr + offsetof( JitCallParams, FilterStride )] );
	const reg64_t regDilationWidth = preservedGPR[regUsed++];
	gen.mov( regDilationWidth, ptr[regFramePtr + offsetof( JitCallParams, DilationWidth )] );
	const reg64_t regInputStride = preservedGPR[regUsed++];
	gen.mov( regInputStride, ptr[regFramePtr + offsetof( JitCallParams, InputStride )] );

	const reg64_t regNegInputBase = outputCount == 1 ? preservedGPR[regUsed++] : regNotUsed;
	const reg64_t regInputWidth = outputCount == 1 ? preservedGPR[regUsed++] : regNotUsed;
	if( outputCount == 1 ) {
		gen.mov( regNegInputBase, ptr[regFramePtr + offsetof( JitCallParams, InputBase )] );
		gen.neg( regNegInputBase );
		gen.mov( regInputWidth, ptr[regFramePtr + offsetof( JitCallParams, InputWidth )] );
	}

	clearUsedYmms( filterCount, outputCount, gen );

	const reg64_t regRemRows = preservedGPR[regUsed++];
	gen.mov( regRemRows, ptr[regFramePtr + offsetof( JitCallParams, KernelHeight )] );

	Label rowCycleStart;
	gen.L( rowCycleStart );

	const reg64_t regRemCols = preservedGPR[regUsed++];
	gen.mov( regRemCols, ptr[regFramePtr + offsetof( JitCallParams, KernelWidth )] );

	Label colCycleStart;
	gen.L( colCycleStart );

	Label skipPadding;
	if( outputCount == 1 ) {
		const reg64_t tempReg = preservedGPR[regUsed++];
		gen.lea( tempReg, ptr[regInput + regNegInputBase] );
		gen.cmp( tempReg, regInputWidth );
		gen.jae( skipPadding, CodeGenerator::T_NEAR );
	}

	const reg64_t regShiftedInput = outputCount >= 3 ? preservedGPR[regUsed++] : regNotUsed;
	if( outputCount >= 3 ) {
		gen.lea( regShiftedInput, ptr[regInput + 2 * regStrideWidth] );
	}

	const reg64_t regShiftedFilter = filterCount >= 3 ? preservedGPR[regUsed++] : regNotUsed;
	if( filterCount >= 3 ) {
		gen.lea( regShiftedFilter, ptr[regFilter + 2 * regFilterStride] );
	}

	auto genSingleBlock = [&]( int broadcast )
	{
		const int broadcastOffset = broadcast * sizeof( float );
		const int vectorOffset = broadcast * 8 * sizeof( float );

		// This macro multiplies and accumulates for FilterCount by OutputCount block of the output buffer.
		Ymm inputYmm[3] = { Ymm( 13 ), Ymm( 14 ), Ymm( 15 ) };
		Address inputAddr[3] = { gen.ptr[regInput + broadcastOffset], gen.ptr[regInput + regStrideWidth + broadcastOffset],
			gen.ptr[regShiftedInput + broadcastOffset] };
		for( int output = 0; output < 3; ++output ) {
			gen.vbroadcastss( inputYmm[output], inputAddr[output] );
		}

		Address filterAddr[4] = { gen.ptr[regFilter + vectorOffset], gen.ptr[regFilter + regFilterStride + vectorOffset],
			gen.ptr[regShiftedFilter + vectorOffset], gen.ptr[regShiftedFilter + regFilterStride + vectorOffset] };

		if( outputCount == 1 ) {
			for( int filter = 0; filter < filterCount; ++filter ) {
				baseGen.vfmadd231ps( acc[filter], inputYmm[0], filterAddr[filter] );
			}
		} else {
			for( int filter = 0; filter < filterCount; ++filter ) {
				Ymm filterYmm( 12 );
				gen.vmovups( filterYmm, filterAddr[filter] );
				for( int output = 0; output < outputCount; ++output ) {
					baseGen.vfmadd231ps( acc[output * 4 + filter], inputYmm[output], filterYmm );
				}
			}
		}
	};

	for( int broadcast = 0; broadcast < 8; ++broadcast ) {
		genSingleBlock( broadcast );
	}

	gen.L( skipPadding );
	gen.add( regInput, regDilationWidth );
	gen.add( regFilter, 8 * 8 * sizeof( float ) );
	gen.dec( regRemCols );
	gen.jnz( colCycleStart, CodeGenerator::T_NEAR );

	gen.add( regInput, regInputStride );
	if( outputCount == 1 ) {
		gen.sub( regNegInputBase, ptr[regFramePtr + offsetof( JitCallParams, DilatedInputWidth )] );
	}

	gen.dec( regRemRows );
	gen.jnz( rowCycleStart, CodeGenerator::T_NEAR );

	/*
	;   This macro generates code to process an output block after the inner
	;   convolution kernel has executed and then stores the output block to the
	;   output buffer.
	*/

	const reg64_t regOutput = regInput;
	gen.mov( regOutput, ptr[regFramePtr + offsetof( JitCallParams, Output )] );
	const reg64_t regOutputStride = regInputStride;
	gen.mov( regOutputStride, ptr[regFramePtr + offsetof( JitCallParams, OutputStride )] );
	const reg64_t regShiftedOutput = regShiftedFilter;
	if( filterCount >= 3 ) {
		gen.lea( regShiftedOutput, ptr[regOutput + 2 * regOutputStride] );
	}
	
	const reg64_t regFlags = regRemRows;
	gen.mov( regFlags, ptr[regFramePtr + offsetof( JitCallParams, Flags )] );

	Label skipAccumulateOutput;
	gen.test( regFlags, ACCUMULATE_OUTPUT );
	gen.jz( skipAccumulateOutput, CodeGenerator::T_NEAR );

	reg64_t outputRegs[2] = { regOutput, regShiftedOutput };
	for( int filter = 0; filter < filterCount; ++filter ) {
		for( int output = 0; output < outputCount; ++output ) {
			baseGen.vaddps( acc[output * 4 + filter], acc[output * 4 + filter],
				ptr[outputRegs[filter / 2] + ( filter % 2 ) * regOutputStride + output * SizeOfYmm] );
		}
	}

	gen.L( skipAccumulateOutput );

	Label skipBias;
	gen.test( regFlags, ADD_BIAS );
	gen.jz( skipBias, CodeGenerator::T_NEAR );

	const reg64_t regBias = regFilter;
	gen.mov( regBias, ptr[regFramePtr + offsetof( JitCallParams, Bias )] );

	Address biasAddr[4] = { ptr[regBias], ptr[regBias + SizeOfYmm], ptr[regBias + 2 * SizeOfYmm],
		ptr[regBias + 3 * SizeOfYmm] };

	if( outputCount == 1 ) {
		for( int filter = 0; filter < filterCount; ++filter ) {
			baseGen.vaddps( Ymm( filter ), Ymm( filter ), biasAddr[filter] );
		}
	} else {
		const int biasYmmIdx = 12;
		for( int filter = 0; filter < filterCount; ++filter ) {
			baseGen.vmovups( Ymm( biasYmmIdx + filter ), biasAddr[filter] );
		}
		for( int output = 0; output < outputCount; ++output ) {
			for( int filter = 0; filter < filterCount; ++filter ) {
				baseGen.vaddps( Ymm( output * 4 + filter ), Ymm( output * 4 + filter ), Ymm( biasYmmIdx + filter ) );
			}
		}
	}

	gen.L( skipBias );

	for( int output = 0; output < outputCount; ++output ) {
		for( int filter = 0; filter < filterCount; ++filter ) {
			baseGen.vmovups( ptr[outputRegs[filter / 2] + ( filter % 2 ) * regOutputStride + output * SizeOfYmm],
				Ymm( output * 4 + filter ) );
		}
	}

	gen.Epilogue( preservedGPR, acc );
	gen.ret();

	// printf( "code size:\t%d\n", static_cast<int>( gen.getSize() ) );
}

static void runComputeBlocks( int filterCount, int outputCount, const float* input, int strideWidth,
	const float* filter, int filterStride, const float* inputBase,
	int inputWidth, int kernelHeight, int kernelWidth, int dilationWidth, int dilatedInputWidth,
	int inputStride, const float* bias, float* output, int outputStride, int flags )
{
	const int genIndex = ( outputCount - 1 ) + 3 * ( filterCount - 1 );
	CJitCommon& gen = computeBlockGens[genIndex].gen;
	if( gen.getSize() == 0 ) {
		initComputeBlocks( filterCount, outputCount );
	}

	//gen.dump();
	typedef void (*TComputeBlockJitFunc)( JitCallParams* );
	JitCallParams callFrame = { input, strideWidth * sizeof( float ), filter, filterStride * sizeof( float ),
		inputBase, inputWidth * sizeof( float ), static_cast<size_t>( kernelHeight ), static_cast<size_t>( kernelWidth ),
		dilationWidth * sizeof( float ), dilatedInputWidth * sizeof( float ), inputStride * sizeof( float ),
		bias, output, outputStride * sizeof( float ), static_cast<size_t>( flags ) };
	gen.getCode<TComputeBlockJitFunc>()( &callFrame );
}

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

/*
;   This macro generates code to compute the convolution for a vector of input
;   blocks and a vector of filter blocks to produce a matrix of output blocks.
;
;   OutputCount=1 generates special case code to handle padding blocks. All
;   other output counts assume no padding.
*/
#define PROCESS_OUTPUT_COUNT_JIT_N(FilterCount, OutputCount) \
{ \
	const float* prevInput = input; \
	const float* filter = frame.Filter; \
	\
	runComputeBlocks( FilterCount, OutputCount, input, strideWidth, filter, filterStride, \
		frame.InputBase, frame.InputWidth, frame.KernelHeight, frame.KernelWidth, \
		frame.DilationWidth, frame.DilatedInputWidth, frame.InputStride, frame.Bias, \
		output, frame.OutputStride, frame.Flags ); \
	\
	output += OutputCount * 8; \
	input = prevInput; \
}

template<int FilterCount>
static void singleKernel( const KernelFrame& frame, const float*& input, int filterStride,
	int dilationWidth, float*& output, int strideWidth, int inputStride, int outputCount )
{
	for( int i = 0; i < outputCount; ++i ) {
		PROCESS_OUTPUT_COUNT_JIT_N( FilterCount, 1 )
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
		PROCESS_OUTPUT_COUNT_JIT_N( FilterCount, 3 ) \
		input += 3 * strideWidth; \
		remOutputCount -= 3; \
	} \
	\
	if( remOutputCount >= 2 ) { \
		PROCESS_OUTPUT_COUNT_JIT_N( FilterCount, 2 ) \
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
