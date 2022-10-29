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

// Blocked convolution is a special algo when input and output channels are divided into blocks of size, used in SIMD
// In our case (AVX) the block size is 8
// This algo requires data to be packed from
// N H W C_orig -> N C H W c
// where c == BlockSize and C == C_Orig / BlockSize
// This applies to both input and output
// As a result this algo requires both C_Input and C_Output to be a multiple of BlockSize

// Descriptor for blocked convolution
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

	const CBlobDesc Source; // Input blob size
	const CBlobDesc Result; // Output blob size
	const CBlobDesc Filter; // Filter blob size
	const int StrideHeight; // Step along height
	const int StrideWidth; // Step along width
	const int PaddingHeight; // Padding rows
	const int PaddingWidth; // Paddinc columns
	const int DilationHeight; // Dilation along height
	const int DilationWidth; // Dilation along width
};

CConvolutionDesc* CAvxMathEngine::InitBlockedConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
	int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter,
	const CBlobDesc& result ) const
{
	const int filterCount = filter.ObjectCount();
	const int inputChannels = source.Depth() * source.Channels();
	if( filterCount % 8 != 0 || inputChannels % 8 != 0 ) {
		// Algorithmic restrictions
		return nullptr;
	}

	/*if( filter.Height() == 1 && filter.Width() == 1 ) {
		// We can't outperform matrix multiplication
		return nullptr;
	}

	// Heuristics tuned for better effectiveness
	// Number of operations in convolution == outputBlobsSize * filterHeight * filterWidth * inputChannels
	// Packing each data (input/output/filter) is linear

	// Ratio between number of operations in convolution and different packing operations
	const size_t inputRatio = static_cast<size_t>( result.ObjectSize() ) * filter.Height() * filter.Width()
		/ source.Height() / source.Width();
	const size_t outputRatio = static_cast<size_t>( filter.Height() ) * filter.Width() * source.Depth() * source.Channels();
	const size_t filterRatio = static_cast<size_t>( result.ObjectCount() * result.Height() * result.Width() );
	// If any of ratio is less than this value then packing takes too much time...
	const size_t minRatio = 200;
	if( inputRatio < minRatio || outputRatio < minRatio || filterRatio < minRatio ) {
		return nullptr;
	}
	
	// If both input and output ratios are slightly above min value the algo is still slow
	const size_t minIOMult = 65536;
	if( inputRatio * outputRatio < minIOMult ) {
		return nullptr;
	}*/

	return new CAvxBlockedConvDesc( source, result, filter, strideHeight, strideWidth, paddingHeight, paddingWidth,
		dilationHeight, dilationWidth );
}

// Packs NHWC data from source into NCHWc result
// The desc must contain data dims in NHWC format
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
	_mm256_zeroupper();
}

// Unpacks NCHWc data from source into NHWC result
// The desc must contain data dims in NHWC format
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
	_mm256_zeroupper();
}

// This algo requires special pack for filters from O_Orig x H x W x I_Orig into O x I x H x W x i x o
// where O_Orig and I_Orig are the output and input channels in convolution
// O = O_Orig / BlockSize, I = I_Orig / BlockSize, and i = o = BlockSize
// The desc must contain filter data dims in OHWI format
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
						__m256 simdSource = _mm256_loadu_ps( source );
						for( int i = 0; i < 8; ++i ) {
							const int idx = o + 8 * ( i + 8 * ( w + width * ( h + height * ( I + channels / 8 * O ) ) ) );
							result[idx] = simdSource.m256_f32[i];
						}
						source += 8;
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
	// JIT call parameters and their order
	struct CParams {
		const float* Input;
		size_t StrideWidthBytes;
		const float* Filter;
		size_t FilterStrideBytes;
		const float* InputAfterLeftPad;
		size_t InputWidthBytes;
		size_t FilterHeight;
		size_t FilterWidth;
		size_t DilationWidthBytes;
		size_t DilatedInputWidthBytes;
		size_t InputStrideBytes;
		const float* Bias;
		float* Output;
		size_t OutputStrideBytes;
		size_t Flags;
		size_t OutputCountLeftPad;
		size_t OutputCount;
		size_t OutputCountRightPad;
		size_t FilterCount;
		float* PrevRsp;
	};

	CBlockedConvGen();

	// Runs generated code with the given parameters
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

	void genConvKernel();

	// Epilogue
	void genEpilogue();

	const reg64_t regInput = rdi;
	const reg64_t regRemOutputCount = r10;

	const reg64_t regBlockInput = rcx;
	const reg64_t regStrideWidth = r9;
	const reg64_t regFilter = rdx;
	const reg64_t regFilterStride = rsi;
	const reg64_t regDilationWidth = rbp;
	const reg64_t regInputStride = r15;
	const reg64_t regNegInputAfterLeftPad = r13;
	const reg64_t regInputWidth = r14;
	const reg64_t regOutput = r8;
	const reg64_t regFilterHeight = r11;
	const reg64_t regFilterWidth = r12;
	const reg64_t regShiftedInput = r14;
	const reg64_t regShiftedFilter = rbx;

	const Ymm acc[3][4] = {
		{ Ymm( 0 ), Ymm( 1 ), Ymm( 2 ), Ymm( 3 ) },
		{ Ymm( 4 ), Ymm( 5 ), Ymm( 6 ), Ymm( 7 ) },
		{ Ymm( 8 ), Ymm( 9 ), Ymm( 10 ), Ymm( 11 ) }
	};
};

CBlockedConvGen::CBlockedConvGen() : CodeGenerator( 16 * 1024 )
{
	genPrologue();
	genConvKernel();
	genEpilogue();
}

void CBlockedConvGen::Run( CParams& params )
{
	typedef void (*TComputeBlockJitFunc)( CParams* );
	getCode<TComputeBlockJitFunc>()( &params );
}

void CBlockedConvGen::genPrologue()
{
	push( rbp );
	mov( rbp, rsp );

	sub( rsp, static_cast<uint32_t>( 16 * SizeOfYmm ) );
	for( int i = 0; i < 16; i++ ) {
		vmovdqu( ptr[rsp + i * SizeOfYmm], Ymm( i ) );
	}

	reg64Vec_t gprs = { rax, rbx, rbp, rcx, rdx, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15 };
	for( int i = 0; i < gprs.size(); i++ ) {
		push( gprs[i] );
	}

	mov( ptr[Param1 + offsetof( CParams, PrevRsp )], rsp );
	mov( rsp, Param1 );

	mov( regInput, ptr[rsp + offsetof( CParams, Input )] );
	mov( regFilterStride, ptr[rsp + offsetof( CParams, FilterStrideBytes )] );
	mov( regDilationWidth, ptr[rsp + offsetof( CParams, DilationWidthBytes )] );
	mov( regOutput, ptr[rsp + offsetof( CParams, Output )] );
	mov( regStrideWidth, ptr[rsp + offsetof( CParams, StrideWidthBytes )] );
	mov( regInputStride, ptr[rsp + offsetof( CParams, InputStrideBytes )] );
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
	mov( regBlockInput, regInput );
	mov( regFilter, ptr[rsp + offsetof( CParams, Filter )] );
	mov( regFilterHeight, ptr[rsp + offsetof( CParams, FilterHeight )] );
	mov( regFilterWidth, ptr[rsp + offsetof( CParams, FilterWidth )] );
	if( outputCount == 1 ) {
		mov( regNegInputAfterLeftPad, ptr[rsp + offsetof( CParams, InputAfterLeftPad )] );
		neg( regNegInputAfterLeftPad );
		mov( regInputWidth, ptr[rsp + offsetof( CParams, InputWidthBytes )] );
	}
}

void CBlockedConvGen::genClearYmms( int filterCount, int outputCount )
{
	for( int filter = 0; filter < filterCount; ++filter ) {
		for( int output = 0; output < outputCount; ++output ) {
			vxorps( acc[output][filter], acc[output][filter] );
		}
	}
}

void CBlockedConvGen::genComputeBlockLoops( int filterCount, int outputCount )
{
	mov( regFilterHeight, ptr[rsp + offsetof( CParams, FilterHeight )] );

	// In case if whole kernel is in padding
	Label emptyKernelSkip;
	test( regFilterHeight, regFilterHeight );
	jz( emptyKernelSkip, T_NEAR );

	Label rowCycleStart;
	L( rowCycleStart );

	const reg64_t regRemCols = rax;
	// TODO: remove this load when figure out how to get rid of registers
	mov( regRemCols, regFilterWidth );

	Label colCycleStart;
	L( colCycleStart );

	Label skipPadding;
	if( outputCount == 1 ) {
		lea( rbx, ptr[regBlockInput + regNegInputAfterLeftPad] );
		cmp( rbx, regInputWidth );
		jae( skipPadding, CodeGenerator::T_NEAR );
	}

	if( outputCount >= 3 ) {
		lea( regShiftedInput, ptr[regBlockInput + 2 * regStrideWidth] );
	}

	if( filterCount >= 3 ) {
		lea( regShiftedFilter, ptr[regFilter + 2 * regFilterStride] );
	}

	for( int broadcast = 0; broadcast < 8; ++broadcast ) {
		genSingleComputeBlock( filterCount, outputCount, broadcast );
	}

	L( skipPadding );
	add( regBlockInput, regDilationWidth );
	add( regFilter, 8 * 8 * sizeof( float ) );
	dec( regRemCols );
	jnz( colCycleStart, CodeGenerator::T_NEAR );

	add( regBlockInput, regInputStride );
	if( outputCount == 1 ) {
		sub( regNegInputAfterLeftPad, ptr[rsp + offsetof( CParams, DilatedInputWidthBytes )] );
	}

	dec( regFilterHeight );
	jnz( rowCycleStart, CodeGenerator::T_NEAR );

	L( emptyKernelSkip );
}

void CBlockedConvGen::genSingleComputeBlock( int filterCount, int outputCount, int broadcast )
{
	const int broadcastOffset = broadcast * sizeof( float );
	const int vectorOffset = broadcast * 8 * sizeof( float );

	// This macro multiplies and accumulates for FilterCount by OutputCount block of the output buffer.
	Ymm inputYmm[3] = { Ymm( 13 ), Ymm( 14 ), Ymm( 15 ) };
	Address inputAddr[3] = { ptr[regBlockInput + broadcastOffset], ptr[regBlockInput + regStrideWidth + broadcastOffset],
		ptr[regShiftedInput + broadcastOffset] };
	for( int output = 0; output < outputCount; ++output ) {
		vbroadcastss( inputYmm[output], inputAddr[output] );
	}

	Address filterAddr[4] = { ptr[regFilter + vectorOffset], ptr[regFilter + regFilterStride + vectorOffset],
		ptr[regShiftedFilter + vectorOffset], ptr[regShiftedFilter + regFilterStride + vectorOffset] };

	if( outputCount == 1 ) {
		for( int filter = 0; filter < filterCount; ++filter ) {
			vfmadd231ps( acc[0][filter], inputYmm[0], filterAddr[filter] );
		}
	} else {
		for( int filter = 0; filter < filterCount; ++filter ) {
			Ymm filterYmm( 12 );
			vmovups( filterYmm, filterAddr[filter] );
			for( int output = 0; output < outputCount; ++output ) {
				vfmadd231ps( acc[output][filter], inputYmm[output], filterYmm);
			}
		}
	}
}

void CBlockedConvGen::genPostProcessing( int filterCount, int outputCount )
{
	const reg64_t regOutputStride = rax;
	mov( regOutputStride, ptr[rsp + offsetof( CParams, OutputStrideBytes )] );
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
			vaddps( acc[output][filter], acc[output][filter],
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
			vaddps( acc[0][filter], acc[0][filter], ptr[regBias + filter * SizeOfYmm]);
		}
	} else {
		const Ymm biasYmm[4] = { Ymm( 12 ), Ymm( 13 ), Ymm( 14 ), Ymm( 15 ) };
		for( int filter = 0; filter < filterCount; ++filter ) {
			vmovups( biasYmm[filter], ptr[regBias + filter * SizeOfYmm]);
		}
		for( int output = 0; output < outputCount; ++output ) {
			for( int filter = 0; filter < filterCount; ++filter ) {
				vaddps( acc[output][filter], acc[output][filter], biasYmm[filter] );
			}
		}
	}

	L( skipBias );

	for( int output = 0; output < outputCount; ++output ) {
		for( int filter = 0; filter < filterCount; ++filter ) {
			vmovups( ptr[outputRegs[filter / 2] + ( filter % 2 ) * regOutputStride + output * SizeOfYmm],
				acc[output][filter] );
		}
	}

	add( regOutput, outputCount * 8 * sizeof( float ) );
}

void CBlockedConvGen::genPaddingProcessing( int filterCount )
{
	Label processNextOutput;

	L( processNextOutput );
	genComputeBlocks( filterCount, 1 );
	add( regInput, regStrideWidth );
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
	// Initially subtract 3 for jumping via comparison with 0 (jz, jb, jae, etc.)
	sub( regRemOutputCount, 3 );
	jb( processRemainingOutputs, T_NEAR );

	L( processThreeOutputs );
	genComputeBlocks( filterCount, 3 );
	const reg64_t regTemp = rax;
	lea( regTemp, ptr[regStrideWidth + 2 * regStrideWidth] );
	add( regInput, regTemp );
	sub( regRemOutputCount, 3 );
	jae( processThreeOutputs, T_NEAR );

	L( processRemainingOutputs );
	add( regRemOutputCount, 3 );
	jz( processRemainingWithRightPad, T_NEAR );
	cmp( regRemOutputCount, 2 );
	jb( processRemainingWithRightPad, T_NEAR );
	genComputeBlocks( filterCount, 2 );
	lea( regInput, ptr[regInput + 2 * regStrideWidth] );
	sub( regRemOutputCount, 2 );

	Label skipRightPad;
	L( processRemainingWithRightPad );
	add( regRemOutputCount, ptr[rsp + offsetof( CParams, OutputCountRightPad )] );
	jz( skipRightPad, T_NEAR );
	genPaddingProcessing( filterCount );
	L( skipRightPad );
}

void CBlockedConvGen::genConvKernel()
{
	const reg64_t regFilterCount = r11;
	mov( regFilterCount, ptr[rsp + offsetof( CParams, FilterCount )] );

	Label processThreeFilters;
	Label processLessThanThreeFilters;
	Label processOneFilter;
	Label convKernelEnd;

	cmp( regFilterCount, 3 );
	je( processThreeFilters, T_NEAR );
	jb( processLessThanThreeFilters, T_NEAR );
	genProcessFilterCount( 4 );
	jmp( convKernelEnd, T_NEAR );

	L( processThreeFilters );
	genProcessFilterCount( 3 );
	jmp( convKernelEnd, T_NEAR );

	L( processLessThanThreeFilters );
	cmp( regFilterCount, 2 );
	jb( processOneFilter, T_NEAR );
	genProcessFilterCount( 2 );
	jmp( convKernelEnd, T_NEAR );

	L( processOneFilter );
	genProcessFilterCount( 1 );
	L( convKernelEnd );
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

static CBlockedConvGen blockedConvGen;

// Calculates the number of output elements affected by front or back padding
static void calcOutputPad( int inputSize, int filterSize, int outputSize, int stride, int padding, int dilation,
	size_t& outputFrontPad, size_t& outputNoPad, size_t& outputBackPad )
{
	const int filterCoverage = ( filterSize - 1 ) * dilation + 1;
	const int unaffectedByBackPad = ( inputSize + padding >= filterCoverage )
		? ( inputSize + padding - filterCoverage ) / stride + 1 : 0;
	outputFrontPad = static_cast<size_t>( min( unaffectedByBackPad, ( padding + stride - 1 ) / stride ) );
	outputNoPad = static_cast<size_t>( unaffectedByBackPad ) - outputFrontPad;
	outputBackPad = static_cast<size_t>( outputSize - unaffectedByBackPad );
}

void CAvxMathEngine::BlockedConvolution( const CConvolutionDesc& convDesc, const float* packedSource,
	const float* packedFilter, const float* freeTerm, float* packedResult ) const
{
	const CAvxBlockedConvDesc& desc = static_cast<const CAvxBlockedConvDesc&>( convDesc );
	
	const int inputHeight = desc.Source.Height();
	const int inputWidth = desc.Source.Width();
	const int inputChannels = desc.Source.Depth() * desc.Source.Channels();
	const int outputHeight = desc.Result.Height();
	const int outputWidth = desc.Result.Width();
	const int filterCount = desc.Filter.ObjectCount();
	const int filterHeight = desc.Filter.Height();
	const int filterWidth = desc.Filter.Width();
	const int dilationHeight = desc.DilationHeight;

	size_t outputColLeftPad = 0, outputColNoPad = 0, outputColRightPad = 0;
	calcOutputPad( inputWidth, filterWidth, outputWidth, desc.StrideWidth, desc.PaddingWidth, desc.DilationWidth,
		outputColLeftPad, outputColNoPad, outputColRightPad );

	// AVX allows to process 8 floats via single instruction
	const int blockSize = 8;
	// JIT function may process up to 4 filter blocks at one time
	const int filterSetBlocks = 4;
	const int filterSetSize = filterSetBlocks * blockSize;

	const int filterSetCount = ( filterCount + filterSetSize - 1 ) / filterSetSize;
	const int totalWork = desc.Source.ObjectCount() * filterSetCount * outputHeight;
	const int curThreadCount = IsOmpRelevant( totalWork,
		static_cast<int64_t>( desc.Result.BlobSize() ) * desc.Filter.ObjectSize() ) ? threadCount : 1;

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int workIndex, workRem;
		if( OmpGetTaskIndexAndCount( totalWork, workIndex, workRem ) ) {
			int outputRowIndex = workIndex % outputHeight;
			int filterSetIndex = ( workIndex / outputHeight ) % filterSetCount;
			int batchIndex = ( workIndex / outputHeight ) / filterSetCount;

			const float* inputImage = packedSource + batchIndex * desc.Source.ObjectSize();
			const float* filter = packedFilter + filterSetIndex * filterSetSize * desc.Filter.ObjectSize();
			float* output = packedResult + batchIndex * desc.Result.ObjectSize()
				+ filterSetIndex * filterSetSize * outputHeight * outputWidth;

			CBlockedConvGen::CParams callParams;

			callParams.Bias = freeTerm != nullptr ? freeTerm + filterSetIndex * filterSetSize : freeTerm;
			callParams.StrideWidthBytes = sizeof( float ) * blockSize * desc.StrideWidth;
			callParams.DilationWidthBytes = sizeof( float ) * blockSize * desc.DilationWidth;
			callParams.FilterStrideBytes = sizeof( float ) * blockSize * desc.Filter.ObjectSize();
			callParams.OutputStrideBytes = sizeof( float ) * blockSize * outputHeight * outputWidth;
			callParams.InputWidthBytes = sizeof( float ) * blockSize * inputWidth;
			callParams.DilatedInputWidthBytes = sizeof( float ) * blockSize * dilationHeight * inputWidth;
			callParams.InputStrideBytes = callParams.DilatedInputWidthBytes - filterWidth * callParams.DilationWidthBytes;
			callParams.FilterWidth = filterWidth;
			callParams.OutputCountLeftPad = outputColLeftPad;
			callParams.OutputCount = outputColNoPad;
			callParams.OutputCountRightPad = outputColRightPad;
			callParams.FilterCount = min( filterSetBlocks, ( filterCount / blockSize ) - filterSetIndex * filterSetBlocks );

			const int blockedOutputWidth = blockSize * outputWidth;

			while( workRem > 0 ) {
				const int workThisIter = min( workRem, outputHeight - outputRowIndex );
				for( int ic = 0; ic < inputChannels; ic += blockSize ) {
					callParams.Flags = ic == 0 ? 0 : ACCUMULATE_OUTPUT;

					if( ic + blockSize == inputChannels && callParams.Bias != nullptr ) {
						callParams.Flags |= ADD_BIAS;
					}

					const float* input = inputImage + ic * inputHeight * inputWidth;
					callParams.Output = output + outputRowIndex * blockedOutputWidth;

					for( int work = 0; work < workThisIter; ++work ) {
						callParams.Filter = filter + blockSize * ic * filterHeight * filterWidth;
						callParams.FilterHeight = static_cast< size_t >( filterHeight );

						// Calculate filter intersection with top and bottom padding
						const int firstInputRowIndex = ( outputRowIndex + work ) * desc.StrideHeight - desc.PaddingHeight;
						const int topPaddingFilterRows = min( filterHeight, firstInputRowIndex < 0
							? ( -firstInputRowIndex + dilationHeight - 1 ) / dilationHeight : 0 );
						const int lastInputRowIndex = firstInputRowIndex + ( filterHeight - 1 ) * dilationHeight;
						const int bottomPaddingFilterRows = min( filterHeight, lastInputRowIndex >= inputHeight
							? ( lastInputRowIndex - inputHeight ) / dilationHeight + 1 : 0 );

						// Don't process paddings
						callParams.FilterHeight -= topPaddingFilterRows + bottomPaddingFilterRows;
						// Skip filter rows which are covering top padding 
						callParams.Filter += topPaddingFilterRows * blockSize * blockSize * filterWidth;

						// Calculate input pointers accordingly (skip top padding)
						const int inputRowOffset = firstInputRowIndex + topPaddingFilterRows * dilationHeight;
						callParams.Input = input + blockSize * ( inputRowOffset * inputWidth - desc.PaddingWidth );
						callParams.InputAfterLeftPad = input + blockSize * ( inputRowOffset * inputWidth );

						blockedConvGen.Run( callParams );

						callParams.Output += blockedOutputWidth;
					}
				}

				workRem -= workThisIter;
				if( outputRowIndex + workThisIter == outputHeight ) {
					// All of the rows are processed. Switch to next image
					size_t blockedFilterCount = blockSize * callParams.FilterCount;
					output += blockedFilterCount * outputHeight * outputWidth;
					filter += blockedFilterCount * inputChannels * filterHeight * filterWidth;
					if( callParams.Bias != nullptr ) {
						callParams.Bias += blockedFilterCount;
					}

					if( ++filterSetIndex == filterSetCount ) {
						inputImage += desc.Source.ObjectSize();
						filter = packedFilter;
						callParams.Bias = freeTerm;
						filterSetIndex = 0;
					}

					callParams.FilterCount = min( filterSetBlocks, ( filterCount / blockSize ) - filterSetIndex * filterSetBlocks );
					outputRowIndex = 0;
				}
			}

			_mm256_zeroupper();
		}
	}
}

} // namespace NeoML
