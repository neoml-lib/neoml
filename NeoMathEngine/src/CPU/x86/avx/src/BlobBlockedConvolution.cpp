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

using namespace Xbyak::util;
using namespace Xbyak;

#define ACCUMULATE_OUTPUT 1
#define ADD_BIAS 2

class CBlockedConvGen : public Xbyak::CodeGenerator {
public:
	struct CParams {
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
		size_t OutputCountLeftPad;
		size_t OutputCount;
		size_t OutputCountRightPad;
		float* PrevRsp;
	};

	// TODO: increase if needed
	explicit CBlockedConvGen( int filterCount );

	void Run( CParams& params );

private:
	// Prologue
	void genPrologue();

	// Compute-blocks part (ORT ProcessOutputCountN)
	void genComputeBlocks( int filterCount, int outputCount );
	void genComputeBlocksPrologue( int outputCount );
	void genClearYmms( int filterCount, int outputCount );
	void genComputeBlockLoops( int filterCount, int outputCount );
	void genSingleComputeBlock( int filterCount, int outputCount, int broadcast );
	void genPostProcessing( int filterCount, int outputCount );

	// Padding-processing kernel
	void genPaddingProcessing( int filterCount );

	// ORT ProcessFilterCountN
	void genProcessFilterCount( int filterCount );

	// Epilogue
	void genEpilogue();

	const int _filterCount;

	const reg64_t regGlobalInput = rdi;
	const reg64_t regRemOutputCount = r10;

	const reg64_t regInput = rcx;
	const reg64_t regStrideWidth = r9;
	const reg64_t regFilter = rdx;
	const reg64_t regFilterStride = rsi;
	const reg64_t regDilationWidth = rbp;
	const reg64_t regInputStride = r15;
	const reg64_t regNegInputBase = r13;
	const reg64_t regInputWidth = r14;
	const reg64_t regOutput = r8;
	const reg64_t regKernelHeight = r11;
	const reg64_t regKernelWidth = r12;
	const reg64_t regShiftedInput = r14;
	const reg64_t regShiftedFilter = rbx;

};

CBlockedConvGen::CBlockedConvGen( int filterCount ) :
	CodeGenerator( 8192 * 1024 ),
	_filterCount( filterCount )
{
}

void CBlockedConvGen::Run( CParams& params )
{
	if( getSize() == 0 ) {
		genPrologue();
		genProcessFilterCount( _filterCount );
		genEpilogue();
	}

	typedef void (*TComputeBlockJitFunc)( CParams* );
	getCode<TComputeBlockJitFunc>()( &params );
}

void CBlockedConvGen::genPrologue()
{
	push( rbp );
	mov( rbp, rsp );

	sub( rsp, static_cast< uint32_t >( 16 * SizeOfYmm ) );
	for( int i = 0; i < 16; i++ ) {
		vmovdqu( ptr[rsp + i * SizeOfYmm], Ymm( i ) );
	}

	reg64Vec_t gprs = { rax, rbx, rbp, rcx, rdx, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15 };
	for( int i = 0; i < gprs.size(); i++ ) {
		push( gprs[i] );
	}

	mov( ptr[Param1 + offsetof( CParams, PrevRsp )], rsp );
	mov( rsp, Param1 );

	mov( regGlobalInput, ptr[rsp + offsetof( CParams, Input )] );
	mov( regFilterStride, ptr[rsp + offsetof( CParams, FilterStride )] );
	mov( regDilationWidth, ptr[rsp + offsetof( CParams, DilationWidth )] );
	mov( regOutput, ptr[rsp + offsetof( CParams, Output )] );
	mov( regStrideWidth, ptr[rsp + offsetof( CParams, StrideWidth )] );
	mov( regInputStride, ptr[rsp + offsetof( CParams, InputStride )] );
}

void CBlockedConvGen::genComputeBlocks( int filterCount, int outputCount )
{
	genComputeBlocksPrologue( outputCount );
	genClearYmms( filterCount, outputCount );
	genComputeBlockLoops( filterCount, outputCount );
	genPostProcessing( filterCount, outputCount );
}

void CBlockedConvGen::genComputeBlocksPrologue( int outputCount )
{
	mov( regInput, regGlobalInput );
	mov( regFilter, ptr[rsp + offsetof( CParams, Filter )] );
	mov( regKernelHeight, ptr[rsp + offsetof( CParams, KernelHeight )] );
	mov( regKernelWidth, ptr[rsp + offsetof( CParams, KernelWidth )] );
	if( outputCount == 1 ) {
		mov( regNegInputBase, ptr[rsp + offsetof( CParams, InputBase )] );
		neg( regNegInputBase );
		mov( regInputWidth, ptr[rsp + offsetof( CParams, InputWidth )] );
	}
}

void CBlockedConvGen::genClearYmms( int filterCount, int outputCount )
{
	for( int filter = 0; filter < filterCount; ++filter ) {
		for( int output = 0; output < outputCount; ++output ) {
			vxorps( Xbyak::Ymm( output * 4 + filter ), Xbyak::Ymm( output * 4 + filter ) );
		}
	}
}

void CBlockedConvGen::genComputeBlockLoops( int filterCount, int outputCount )
{
	mov( regKernelHeight, ptr[rsp + offsetof( CParams, KernelHeight )] );

	Label rowCycleStart;
	L( rowCycleStart );

	const reg64_t regRemCols = rax;
	// TODO: remove this load when figure out how to get rid of registers
	mov( regRemCols, regKernelWidth );

	Label colCycleStart;
	L( colCycleStart );

	Label skipPadding;
	if( outputCount == 1 ) {
		lea( rbx, ptr[regInput + regNegInputBase] );
		cmp( rbx, regInputWidth );
		jae( skipPadding, CodeGenerator::T_NEAR );
	}

	if( outputCount >= 3 ) {
		lea( regShiftedInput, ptr[regInput + 2 * regStrideWidth] );
	}

	if( filterCount >= 3 ) {
		lea( regShiftedFilter, ptr[regFilter + 2 * regFilterStride] );
	}

	for( int broadcast = 0; broadcast < 8; ++broadcast ) {
		genSingleComputeBlock( filterCount, outputCount, broadcast );
	}

	L( skipPadding );
	add( regInput, regDilationWidth );
	add( regFilter, 8 * 8 * sizeof( float ) );
	dec( regRemCols );
	jnz( colCycleStart, CodeGenerator::T_NEAR );

	add( regInput, regInputStride );
	if( outputCount == 1 ) {
		sub( regNegInputBase, ptr[rsp + offsetof( CParams, DilatedInputWidth )] );
	}

	dec( regKernelHeight );
	jnz( rowCycleStart, CodeGenerator::T_NEAR );
}

void CBlockedConvGen::genSingleComputeBlock( int filterCount, int outputCount, int broadcast )
{
	const int broadcastOffset = broadcast * sizeof( float );
	const int vectorOffset = broadcast * 8 * sizeof( float );

	// This macro multiplies and accumulates for FilterCount by OutputCount block of the output buffer.
	Ymm inputYmm[3] = { Ymm( 13 ), Ymm( 14 ), Ymm( 15 ) };
	Address inputAddr[3] = { ptr[regInput + broadcastOffset], ptr[regInput + regStrideWidth + broadcastOffset],
		ptr[regShiftedInput + broadcastOffset] };
	for( int output = 0; output < outputCount; ++output ) {
		vbroadcastss( inputYmm[output], inputAddr[output] );
	}

	Address filterAddr[4] = { ptr[regFilter + vectorOffset], ptr[regFilter + regFilterStride + vectorOffset],
		ptr[regShiftedFilter + vectorOffset], ptr[regShiftedFilter + regFilterStride + vectorOffset] };

	if( outputCount == 1 ) {
		for( int filter = 0; filter < filterCount; ++filter ) {
			vfmadd231ps( Ymm( filter ), inputYmm[0], filterAddr[filter]);
		}
	} else {
		for( int filter = 0; filter < filterCount; ++filter ) {
			Ymm filterYmm( 12 );
			vmovups( filterYmm, filterAddr[filter] );
			for( int output = 0; output < outputCount; ++output ) {
				vfmadd231ps( Ymm( output * 4 + filter ), inputYmm[output], filterYmm );
			}
		}
	}
}

void CBlockedConvGen::genPostProcessing( int filterCount, int outputCount )
{
	const reg64_t regOutputStride = rax;
	mov( regOutputStride, ptr[rsp + offsetof( CParams, OutputStride )] );
	const reg64_t regShiftedOutput = rbx;
	if( filterCount >= 3 ) {
		lea( regShiftedOutput, ptr[regOutput + 2 * regOutputStride] );
	}
	const reg64_t regFlags = rdx;
	mov( regFlags, ptr[rsp + offsetof( CParams, Flags )] );

	Label skipAccumulateOutput;
	test( regFlags, ACCUMULATE_OUTPUT );
	jz( skipAccumulateOutput, CodeGenerator::T_NEAR );

	reg64_t outputRegs[2] = { regOutput, regShiftedOutput };
	for( int filter = 0; filter < filterCount; ++filter ) {
		for( int output = 0; output < outputCount; ++output ) {
			vaddps( Ymm( output * 4 + filter ), Ymm( output * 4 + filter ),
				ptr[outputRegs[filter / 2] + ( filter % 2 ) * regOutputStride + output * SizeOfYmm] );
		}
	}

	L( skipAccumulateOutput );

	Label skipBias;
	test( regFlags, ADD_BIAS );
	jz( skipBias, CodeGenerator::T_NEAR );

	const reg64_t regBias = rcx;
	mov( regBias, ptr[rsp + offsetof( CParams, Bias )] );

	if( outputCount == 1 ) {
		for( int filter = 0; filter < filterCount; ++filter ) {
			vaddps( Ymm( filter ), Ymm( filter ), ptr[regBias + filter * SizeOfYmm] );
		}
	} else {
		const int biasYmmIdx = 12;
		for( int filter = 0; filter < filterCount; ++filter ) {
			vmovups( Ymm( biasYmmIdx + filter ), ptr[regBias + filter * SizeOfYmm] );
		}
		for( int output = 0; output < outputCount; ++output ) {
			for( int filter = 0; filter < filterCount; ++filter ) {
				vaddps( Ymm( output * 4 + filter ), Ymm( output * 4 + filter ), Ymm( biasYmmIdx + filter ) );
			}
		}
	}

	L( skipBias );

	for( int output = 0; output < outputCount; ++output ) {
		for( int filter = 0; filter < filterCount; ++filter ) {
			vmovups( ptr[outputRegs[filter / 2] + ( filter % 2 ) * regOutputStride + output * SizeOfYmm],
				Ymm( output * 4 + filter ) );
		}
	}

	add( regOutput, outputCount * 8 * sizeof( float ) );
}

void CBlockedConvGen::genPaddingProcessing( int filterCount )
{
	Label processNextOutput;

	L( processNextOutput );
	genComputeBlocks( filterCount, 1 );
	add( regGlobalInput, regStrideWidth );
	dec( regRemOutputCount );
	jnz( processNextOutput, T_NEAR );
}

void CBlockedConvGen::genProcessFilterCount( int filterCount )
{
	Label skipLeftPad;
	mov( regRemOutputCount, ptr[rsp + offsetof( CParams, OutputCountLeftPad )] );
	test( regRemOutputCount, regRemOutputCount );
	jz( skipLeftPad, T_NEAR );
	genPaddingProcessing( filterCount );
	L( skipLeftPad );

	Label processThreeOutputs;
	Label processRemainingOutputs;
	Label processRemainingWithRightPad;

	mov( regRemOutputCount, ptr[rsp + offsetof( CParams, OutputCount )] );
	sub( regRemOutputCount, 3 );
	jb( processRemainingOutputs, T_NEAR );

	L( processThreeOutputs );
	genComputeBlocks( filterCount, 3 );
	const reg64_t regTemp = rax;
	lea( regTemp, ptr[regStrideWidth + 2 * regStrideWidth] );
	add( regGlobalInput, regTemp );
	sub( regRemOutputCount, 3 );
	jae( processThreeOutputs, T_NEAR );

	L( processRemainingOutputs );
	add( regRemOutputCount, 3 );
	jz( processRemainingWithRightPad, T_NEAR );
	cmp( regRemOutputCount, 2 );
	jb( processRemainingWithRightPad, T_NEAR );
	genComputeBlocks( filterCount, 2 );
	lea( regGlobalInput, ptr[regGlobalInput + 2 * regStrideWidth] );
	sub( regRemOutputCount, 2 );

	Label skipRightPad;
	L( processRemainingWithRightPad );
	add( regRemOutputCount, ptr[rsp + offsetof( CParams, OutputCountRightPad )] );
	jz( skipRightPad, T_NEAR );
	genPaddingProcessing( filterCount );
	L( skipRightPad );
}

void CBlockedConvGen::genEpilogue()
{
	mov( rsp, ptr[rsp + offsetof( CParams, PrevRsp )] );

	reg64Vec_t gprs = { rax, rbx, rbp, rcx, rdx, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15 };
	for( int i = static_cast<int>( gprs.size() - 1 ); i >= 0; i-- ) {
		pop( gprs[i] );
	}

	for( int i = 0; i < 16; i++ ) {
		vmovdqu( Ymm( i ), ptr[rsp + static_cast<uint32_t>( i * SizeOfYmm )]);
	}

	leave();
	ret();
}

std::array<std::unique_ptr<CBlockedConvGen>, 4> computeBlockGens;

static void runProcessFilterCount( int filterCount, const float* input, int strideWidth,
	const float* filter, int filterStride, const float* inputBase,
	int inputWidth, int kernelHeight, int kernelWidth, int dilationWidth, int dilatedInputWidth,
	int inputStride, const float* bias, float* output, int outputStride, int flags,
	int outputCountLeftPad, int outputCount, int outputCountRightPad )
{
	const int genIndex = filterCount - 1;
	if( computeBlockGens[genIndex] == nullptr ) {
		computeBlockGens[genIndex].reset( new CBlockedConvGen( filterCount ) );
	}
	CBlockedConvGen::CParams callParams = { input, strideWidth * sizeof( float ), filter, filterStride * sizeof( float ),
		inputBase, inputWidth * sizeof( float ), static_cast<size_t>( kernelHeight ), static_cast<size_t>( kernelWidth ),
		dilationWidth * sizeof( float ), dilatedInputWidth * sizeof( float ), inputStride * sizeof( float ),
		bias, output, outputStride * sizeof( float ), static_cast<size_t>( flags ), static_cast<size_t>( outputCountLeftPad ),
		static_cast<size_t>( outputCount ), static_cast<size_t>( outputCountRightPad ) };
	computeBlockGens[genIndex]->Run( callParams );
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
#define PROCESS_FILTER_COUNT_JIT_N(FilterCount) \
{ \
	runProcessFilterCount( FilterCount, input, strideWidth, frame.Filter, filterStride, \
		frame.InputBase, frame.InputWidth, frame.KernelHeight, frame.KernelWidth, \
		frame.DilationWidth, frame.DilatedInputWidth, frame.InputStride, frame.Bias, \
		output, frame.OutputStride, frame.Flags, frame.OutputCountLeftPad, frame.OutputCount, \
		frame.OutputCountRightPad ); \
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
			PROCESS_FILTER_COUNT_JIT_N( 1 );
			return;
		case 2:
			PROCESS_FILTER_COUNT_JIT_N( 2 );
			return;
		case 3:
			PROCESS_FILTER_COUNT_JIT_N( 3 );
			return;
		case 4:
			PROCESS_FILTER_COUNT_JIT_N( 4 );
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
