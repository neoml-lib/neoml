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
// This algo requires data to be packed from
// N H W C_orig -> N C H W c
// where c == BlockSize and C == C_Orig / BlockSize
// This applies to both input and output
// As a result this algo requires both C_Input and C_Output to be a multiple of BlockSize

// AVX works with 256-bit registers (8 floats)
static const int BlockSize = 8;

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
	if( filterCount % BlockSize != 0 || inputChannels % BlockSize != 0 ) {
		// Algorithmic restrictions
		return nullptr;
	}

	if( filter.Height() == 1 && filter.Width() == 1 ) {
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
	}

	return new CAvxBlockedConvDesc( source, result, filter, strideHeight, strideWidth, paddingHeight, paddingWidth,
		dilationHeight, dilationWidth );
}

// Packs NHWC data from source into NCHWc result
// The desc must contain data dims in NHWC format
void CAvxMathEngine::PackBlockedData( const CBlobDesc& desc, const float* source, float* result ) const
{
	const int channels = desc.Depth() * desc.Channels();
	assert( channels % BlockSize == 0 );

	const int height = desc.Height();
	const int width = desc.Width();
	const int geomSize = height * width;
	const int batch = desc.ObjectCount();

	for( int b = 0; b < batch; ++b ) {
		for( int hw = 0; hw < geomSize; ++hw ) {
			float* channelResult = result + hw * BlockSize;
			for( int C = 0; C < channels / BlockSize; ++C ) {
				_mm256_storeu_ps( channelResult, _mm256_loadu_ps( source ) );
				source += BlockSize;
				channelResult += geomSize * BlockSize;
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
	assert( channels % BlockSize == 0 );

	const int height = desc.Height();
	const int width = desc.Width();
	const int geomSize = height * width;
	const int batch = desc.ObjectCount();

	for( int b = 0; b < batch; ++b ) {
		for( int C = 0; C < channels / BlockSize; ++C ) {
			float* geomResult = result + C * BlockSize;
			for( int hw = 0; hw < geomSize; ++hw ) {
				_mm256_storeu_ps( geomResult, _mm256_loadu_ps( source ) );
				source += BlockSize;
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
	assert( filterCount % BlockSize == 0 );
	assert( channels % BlockSize == 0 );

	for( int O = 0; O < filterCount / BlockSize; ++O ) {
		for( int o = 0; o < BlockSize; ++o ) {
			for( int h = 0; h < height; ++h ) {
				for( int w = 0; w < width; ++w ) {
					for( int I = 0; I < channels / BlockSize; ++I ) {
#ifdef _MSC_VER 
						__m256 simdSource = _mm256_loadu_ps( source );
						for( int i = 0; i < BlockSize; ++i ) {
							const int idx = o + BlockSize * ( i + BlockSize * ( w + width * ( h + height * ( I + channels / BlockSize * O ) ) ) );
							result[idx] = simdSource.m256_f32[i];
						}
#else
						for( int i = 0; i < BlockSize; ++i ) {
							const int idx = o + BlockSize * ( i + BlockSize * ( w + width * ( h + height * ( I + channels / BlockSize * O ) ) ) );
							result[idx] = source[i];
						}
#endif
						source += BlockSize;
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

// Algorithm splits filter blocks (groups of 8 filters) into sets
// One filter sets contains up to 4 filter blocks
// This restriction is caused by the number of Ymm register
static const int MaxFilterSetSize = 4;

// One jit call calculates convolution of one row of output over BlockSize of input channels
// for one filter set (up to MaxFilterSetSize * BlockSize filters)

// The idea is to call it over all blocks of input channels for each filter set
// for all of the output rows

class CBlockedConvGen : public Xbyak::CodeGenerator {
public:
	// Flags which can be added to Flags parameter in order to
	// control convolution behavior

	// Add values from memory to the result
	// Used when memory contains previous calculations
	// (e.g. processied input channel block isn't first)
	static const size_t AccumulateOutputFlag = 1;
	// Add values from free terms to result
	// Used when processing last input channel block
	static const size_t AddFreeTermFlag = 2;

	// JIT call parameters and their order
	struct CParams {
		// Current position in input
		// May be outside of physical data (when pointing at padding)
		const float* Input;
		// First position in input which point to physical data
		// Used for padding detection
		const float* InputAfterLeftPad;
		// Distance between elements in the same column of input image
		size_t InputWidthBytes;
		// Distance to the input element covered by the next filter column
		size_t DilationWidthBytes;
		// Distance to the input element covered by the next filter row
		size_t DilatedInputWidthBytes;
		// Distance to the next input element covered by the same filter
		size_t StrideWidthBytes;
		// Distance between the input element covered by the last element of this filter row
		// and the first element covered by the next filter row
		size_t InputStrideBytes;
		// Filter pointer
		const float* Filter;
		// Number of blocks in the current set
		size_t FilterSetSize;
		// Number of filter rows to be processed (!rows in padding are excluded!)
		size_t FilterHeight;
		// Number of filter columns to be processed
		size_t FilterWidth;
		// Distance to the next filter block
		size_t FilterStrideBytes;
		// Free term pointer
		const float* FreeTerm;
		// Output pointer
		float* Output;
		// Distance between different channel blocks belonging to the same "original" output channel
		size_t OutputStrideBytes;
		// Number of output elements in row affected by left padding
		size_t OutputColCountLeftPad;
		// Number of output elements not affected by any padding
		size_t OutputColCountNoPad;
		// Number of output elements in row affected by right padding
		size_t OutputColCountRightPad;
		// Special flags for the correct processing of first/last input channel block
		size_t Flags;
		// Special field used to store stack pointer during JIT call
		float* PrevRsp;
	};

	CBlockedConvGen();

	// Runs generated code with the given parameters
	void Run( CParams& params );

private:
	// Prologue
	void genPrologue();

	void genComputeOutputColumns( int filterSetSize, int outputColCount );
	void genComputeOutputColumnsPrologue( int outputColCount );
	void genClearAccumulators( int filterSetSize, int outputColCount );
	void genFilterGeometryLoops( int filterSetSize, int outputColCount );
	void genProcessSingleInputChannel( int filterSetSize, int outputColCount, int inBlockPos );
	void genComputeOutputColumnsEpilogue( int filterSetSize, int outputColCount );
	void genPaddingProcessing( int filterSetSize );
	void genProcessFilterSet( int filterSetSize );
	void genConvKernel();

	// Epilogue
	void genEpilogue();

	const reg64_t regInput = rdi;
	const reg64_t regRemOutputColCount = r10;

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

	const Ymm acc[3][MaxFilterSetSize] = {
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

	// Store all the YMM registers
	sub( rsp, static_cast<uint32_t>( 16 * SizeOfYmm ) );
	for( int i = 0; i < 16; i++ ) {
		vmovdqu( ptr[rsp + i * SizeOfYmm], Ymm( i ) );
	}

	// Store all the general purpose registers
	reg64Vec_t gprs = { rax, rbx, rbp, rcx, rdx, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15 };
	for( int i = 0; i < gprs.size(); i++ ) {
		push( gprs[i] );
	}

	// At this moment Param1 register contains a pointer to the CBlockedConvGen::CParams
	// The idea is to
	//    1. Write current stack pointer (rsp) to a special field (CParams::PrevRsp)
	//    2. Move stack pointer to the beginning of the CParams
	//    3. Exract params via rsp + offsetof(CParams, ParamName)
	//    4. Later (in epilogue) restore original rsp from this special field (CParams::PrevRsp)
	mov( ptr[Param1 + offsetof( CParams, PrevRsp )], rsp );
	mov( rsp, Param1 );

	mov( regInput, ptr[rsp + offsetof( CParams, Input )] );
	mov( regFilterStride, ptr[rsp + offsetof( CParams, FilterStrideBytes )] );
	mov( regDilationWidth, ptr[rsp + offsetof( CParams, DilationWidthBytes )] );
	mov( regOutput, ptr[rsp + offsetof( CParams, Output )] );
	mov( regStrideWidth, ptr[rsp + offsetof( CParams, StrideWidthBytes )] );
	mov( regInputStride, ptr[rsp + offsetof( CParams, InputStrideBytes )] );
}

// Generates code which computes outputColCount neighboring output columns
// for the given filter set over current input channels block
void CBlockedConvGen::genComputeOutputColumns( int filterSetSize, int outputColCount )
{
	genComputeOutputColumnsPrologue( outputColCount );
	genClearAccumulators( filterSetSize, outputColCount );
	genFilterGeometryLoops( filterSetSize, outputColCount );
	genComputeOutputColumnsEpilogue( filterSetSize, outputColCount );
}

// Generates code which fills registers used during computing output columns
void CBlockedConvGen::genComputeOutputColumnsPrologue( int outputColCount )
{
	mov( regBlockInput, regInput );
	mov( regFilter, ptr[rsp + offsetof( CParams, Filter )] );
	mov( regFilterHeight, ptr[rsp + offsetof( CParams, FilterHeight )] );
	mov( regFilterWidth, ptr[rsp + offsetof( CParams, FilterWidth )] );
	if( outputColCount == 1 ) {
		mov( regNegInputAfterLeftPad, ptr[rsp + offsetof( CParams, InputAfterLeftPad )] );
		neg( regNegInputAfterLeftPad );
		mov( regInputWidth, ptr[rsp + offsetof( CParams, InputWidthBytes )] );
	}
}

 // Generates code which zeroes ymm accumulators used during computing output columns
void CBlockedConvGen::genClearAccumulators( int filterSetSize, int outputColCount )
{
	for( int filterBlockIdx = 0; filterBlockIdx < filterSetSize; ++filterBlockIdx ) {
		for( int outputColIdx = 0; outputColIdx < outputColCount; ++outputColIdx ) {
			vxorps( acc[outputColIdx][filterBlockIdx], acc[outputColIdx][filterBlockIdx] );
		}
	}
}

// Generates loops processing whole filter geometry (filterHeight x filterWidth) in current call
void CBlockedConvGen::genFilterGeometryLoops( int filterSetSize, int outputColCount )
{
	// Cycle over filter rows
	//   for( regFilterHeight = filterHeight; regFilterHeight > 0; --regFilterHeight )
	mov( regFilterHeight, ptr[rsp + offsetof( CParams, FilterHeight )] );

	// Corner case: whole filter is in padding
	Label skilpWholeFilterInPadding;
	test( regFilterHeight, regFilterHeight );
	jz( skilpWholeFilterInPadding, T_NEAR );

	Label rowCycleStart;
	L( rowCycleStart );

	// Inner cycle over filter columns
	//    for( regRemCols = filterWidth; regRemCols > 0; --regRemCols )
	const reg64_t regRemCols = rax;
	mov( regRemCols, regFilterWidth );

	Label colCycleStart;
	L( colCycleStart );

	// The padding check is performed only while process 1 output column at a time
	Label skipPadding;
	if( outputColCount == 1 ) {
		// regNegInputAfterLeftPad contains negative value of the pointer to first element after padding
		// regBlockInput contains current pointer
		// Due to the fact that used types are unsigned we can check both left and right padding by
		// checking the value of (currInput - inputAfterLeftPad)
		// In case of right padding the value will be >= inputWidth
		// In case of left padding the value will be very large (overflow of unsigned type)
		lea( rbx, ptr[regBlockInput + regNegInputAfterLeftPad] );
		cmp( rbx, regInputWidth );
		jae( skipPadding, CodeGenerator::T_NEAR );
	}

	if( outputColCount >= 3 ) {
		lea( regShiftedInput, ptr[regBlockInput + 2 * regStrideWidth] );
	}

	if( filterSetSize >= 3 ) {
		lea( regShiftedFilter, ptr[regFilter + 2 * regFilterStride] );
	}

	// (Reminder: this code is inside cycles over filterHeight and filterWidth)
	// Generate code for processing each input channel in this block
	for( int inBlockPos = 0; inBlockPos < BlockSize; ++inBlockPos ) {
		genProcessSingleInputChannel( filterSetSize, outputColCount, inBlockPos );
	}

	L( skipPadding );
	add( regBlockInput, regDilationWidth );
	// Move filter pointer to the next columns
	// Due to the OIHWio packing it moves the filter pointer to the beginning of the next row
	// when current filter column is the last one
	add( regFilter, BlockSize * BlockSize * sizeof( float ) );
	dec( regRemCols );
	jnz( colCycleStart, CodeGenerator::T_NEAR );
	// Cycle over filter columns ends

	add( regBlockInput, regInputStride );
	if( outputColCount == 1 ) {
		// Move pointer used for padding detection to the next input line covered by this filter set
		sub( regNegInputAfterLeftPad, ptr[rsp + offsetof( CParams, DilatedInputWidthBytes )] );
	}

	dec( regFilterHeight );
	jnz( rowCycleStart, CodeGenerator::T_NEAR );

	L( skilpWholeFilterInPadding );
}

// Generates code which mutilplies one column of one filter set by outputColCount input elements
// inBlockPos is an index of the used input channel inside of a block
//
// It takes outputColCount input elements belonging to the inBlockPos'th input channel inside the block of input channels
// and affected by the same filter elements when filter moves along image width
// The results of applying current filter set to this channel are added to neighboring output columns
void CBlockedConvGen::genProcessSingleInputChannel( int filterSetSize, int outputColCount, int inBlockPos )
{
	const int inBlockInputOffset = inBlockPos * sizeof( float );
	const int inBlockFilerOffset = inBlockPos * BlockSize * sizeof( float );

	Ymm inputYmm[3] = { Ymm( 13 ), Ymm( 14 ), Ymm( 15 ) };
	// Calculate address of input elements for current filter position,
	// and for (outputColCount - 1) neighboring filter position
	Address inputAddr[3] = { ptr[regBlockInput + inBlockInputOffset],
		ptr[regBlockInput + regStrideWidth + inBlockInputOffset], ptr[regShiftedInput + inBlockInputOffset] };

	// Fill input ymms with the values from input from the current channel
	// corresponding to given output columns
	for( int outputColIdx = 0; outputColIdx < outputColCount; ++outputColIdx ) {
		vbroadcastss( inputYmm[outputColIdx], inputAddr[outputColIdx] );
	}

	Address filterAddr[MaxFilterSetSize] = { ptr[regFilter + inBlockFilerOffset],
		ptr[regFilter + regFilterStride + inBlockFilerOffset], ptr[regShiftedFilter + inBlockFilerOffset],
		ptr[regShiftedFilter + regFilterStride + inBlockFilerOffset] };

	if( outputColCount == 1 ) {
		for( int filterBlockIdx = 0; filterBlockIdx < filterSetSize; ++filterBlockIdx ) {
			vfmadd231ps( acc[0][filterBlockIdx], inputYmm[0], filterAddr[filterBlockIdx] );
		}
	} else {
		for( int filterBlockIdx = 0; filterBlockIdx < filterSetSize; ++filterBlockIdx ) {
			Ymm filterYmm( 12 );
			vmovups( filterYmm, filterAddr[filterBlockIdx] );
			for( int outputColIdx = 0; outputColIdx < outputColCount; ++outputColIdx ) {
				vfmadd231ps( acc[outputColIdx][filterBlockIdx], inputYmm[outputColIdx], filterYmm);
			}
		}
	}
}

void CBlockedConvGen::genComputeOutputColumnsEpilogue( int filterSetSize, int outputColCount )
{
	const reg64_t regOutputStride = rax;
	mov( regOutputStride, ptr[rsp + offsetof( CParams, OutputStrideBytes )] );
	const reg64_t regShiftedOutput = rbx;
	if( filterSetSize >= 3 ) {
		lea( regShiftedOutput, ptr[regOutput + 2 * regOutputStride] );
	}
	const reg64_t regFlags = rdx;
	mov( regFlags, ptr[rsp + offsetof( CParams, Flags )] );

	Label skipAccumulateOutput;
	test( regFlags, AccumulateOutputFlag );
	jz( skipAccumulateOutput, CodeGenerator::T_NEAR );

	RegExp outputRegExp[4] = { regOutput, regOutput + regOutputStride,
		regShiftedOutput, regShiftedOutput + regOutputStride };
	for( int filterBlockIdx = 0; filterBlockIdx < filterSetSize; ++filterBlockIdx ) {
		for( int outputColIdx = 0; outputColIdx < outputColCount; ++outputColIdx ) {
			vaddps( acc[outputColIdx][filterBlockIdx], acc[outputColIdx][filterBlockIdx],
				ptr[outputRegExp[filterBlockIdx] + outputColIdx * SizeOfYmm] );
		}
	}

	L( skipAccumulateOutput );

	Label skipFreeTerm;
	test( regFlags, AddFreeTermFlag );
	jz( skipFreeTerm, CodeGenerator::T_NEAR );

	const reg64_t regFreeTerm = rcx;
	mov( regFreeTerm, ptr[rsp + offsetof( CParams, FreeTerm )] );

	if( outputColCount == 1 ) {
		for( int filterBlockIdx = 0; filterBlockIdx < filterSetSize; ++filterBlockIdx ) {
			vaddps( acc[0][filterBlockIdx], acc[0][filterBlockIdx], ptr[regFreeTerm + filterBlockIdx * SizeOfYmm]);
		}
	} else {
		const Ymm freeTermYmm[4] = { Ymm( 12 ), Ymm( 13 ), Ymm( 14 ), Ymm( 15 ) };
		for( int filterBlockIdx = 0; filterBlockIdx < filterSetSize; ++filterBlockIdx ) {
			vmovups( freeTermYmm[filterBlockIdx], ptr[regFreeTerm + filterBlockIdx * SizeOfYmm]);
		}
		for( int outputColIdx = 0; outputColIdx < outputColCount; ++outputColIdx ) {
			for( int filterBlockIdx = 0; filterBlockIdx < filterSetSize; ++filterBlockIdx ) {
				vaddps( acc[outputColIdx][filterBlockIdx], acc[outputColIdx][filterBlockIdx], freeTermYmm[filterBlockIdx] );
			}
		}
	}

	L( skipFreeTerm );

	for( int filterBlockIdx = 0; filterBlockIdx < filterSetSize; ++filterBlockIdx ) {
		for( int outputColIdx = 0; outputColIdx < outputColCount; ++outputColIdx ) {
			vmovups( ptr[outputRegExp[filterBlockIdx] + outputColIdx * SizeOfYmm],
				acc[outputColIdx][filterBlockIdx] );
		}
	}

	add( regOutput, outputColCount * BlockSize * sizeof( float ) );
}

// Generates code which processes output columns affected by padding
void CBlockedConvGen::genPaddingProcessing( int filterSetSize )
{
	Label processNextOutput;

	L( processNextOutput );
	// Current algo generates padding processing only when 1 output column is processed
	genComputeOutputColumns( filterSetSize, 1 );
	add( regInput, regStrideWidth );
	dec( regRemOutputColCount );
	jnz( processNextOutput, T_NEAR );
}

// Generate processing of filter set of given size
void CBlockedConvGen::genProcessFilterSet( int filterSetSize )
{
	assert( filterSetSize >= 1 && filterSetSize <= MaxFilterSetSize );
	Label skipLeftPad;
	mov( regRemOutputColCount, ptr[rsp + offsetof( CParams, OutputColCountLeftPad )] );
	// Process left padding (if needed)
	test( regRemOutputColCount, regRemOutputColCount );
	jz( skipLeftPad, T_NEAR );
	genPaddingProcessing( filterSetSize );
	L( skipLeftPad );

	Label processThreeOutputs;
	Label processRemainingOutputs;
	Label processRemainingWithRightPad;

	mov( regRemOutputColCount, ptr[rsp + offsetof( CParams, OutputColCountNoPad )] );
	// Initially subtract 3 for jumping via comparison with 0 (jz, jb, jae, etc.)
	sub( regRemOutputColCount, 3 );
	jb( processRemainingOutputs, T_NEAR );

	L( processThreeOutputs );
	genComputeOutputColumns( filterSetSize, 3 );
	const reg64_t regTemp = rax;
	lea( regTemp, ptr[regStrideWidth + 2 * regStrideWidth] );
	add( regInput, regTemp );
	sub( regRemOutputColCount, 3 );
	jae( processThreeOutputs, T_NEAR );

	L( processRemainingOutputs );
	// Compensate subtraction above
	add( regRemOutputColCount, 3 );
	jz( processRemainingWithRightPad, T_NEAR );
	cmp( regRemOutputColCount, 2 );
	jb( processRemainingWithRightPad, T_NEAR );
	genComputeOutputColumns( filterSetSize, 2 );
	lea( regInput, ptr[regInput + 2 * regStrideWidth] );
	sub( regRemOutputColCount, 2 );

	// Process irhgt padding (if needed)
	Label skipRightPad;
	L( processRemainingWithRightPad );
	add( regRemOutputColCount, ptr[rsp + offsetof( CParams, OutputColCountRightPad )] );
	jz( skipRightPad, T_NEAR );
	genPaddingProcessing( filterSetSize );
	L( skipRightPad );
}

void CBlockedConvGen::genConvKernel()
{
	const reg64_t regFilterSetSize = r11;
	mov( regFilterSetSize, ptr[rsp + offsetof( CParams, FilterSetSize )] );

	Label processThreeFilters;
	Label processLessThanThreeFilters;
	Label processOneFilter;
	Label convKernelEnd;

	cmp( regFilterSetSize, 3 );
	je( processThreeFilters, T_NEAR );
	jb( processLessThanThreeFilters, T_NEAR );
	genProcessFilterSet( 4 );
	jmp( convKernelEnd, T_NEAR );

	L( processThreeFilters );
	genProcessFilterSet( 3 );
	jmp( convKernelEnd, T_NEAR );

	L( processLessThanThreeFilters );
	cmp( regFilterSetSize, 2 );
	jb( processOneFilter, T_NEAR );
	genProcessFilterSet( 2 );
	jmp( convKernelEnd, T_NEAR );

	L( processOneFilter );
	genProcessFilterSet( 1 );
	L( convKernelEnd );
}

void CBlockedConvGen::genEpilogue()
{
	// Restore stack pointer from the special field
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

	const int filterNumInSet = MaxFilterSetSize * BlockSize;
	const int filterSetCount = ( filterCount + filterNumInSet - 1 ) / filterNumInSet;
	const int totalWork = desc.Source.ObjectCount() * filterSetCount * outputHeight;
	const int curThreadCount = IsOmpRelevant( totalWork,
		static_cast<int64_t>( desc.Result.BlobSize() ) * desc.Filter.ObjectSize() ) ? threadCount : 1;

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int workIndex, workRem;
		if( OmpGetTaskIndexAndCount( totalWork, workIndex, workRem ) ) {
			int outputRowIndex = workIndex % outputHeight;
			int filterSetIndex = ( workIndex / outputHeight ) % filterSetCount;
			const int batchIndex = ( workIndex / outputHeight ) / filterSetCount;

			const float* inputImage = packedSource + batchIndex * desc.Source.ObjectSize();
			const float* filter = packedFilter + filterSetIndex * filterNumInSet * desc.Filter.ObjectSize();
			float* output = packedResult + batchIndex * desc.Result.ObjectSize()
				+ filterSetIndex * filterNumInSet * outputHeight * outputWidth;

			CBlockedConvGen::CParams callParams;

			callParams.FreeTerm = freeTerm != nullptr ? freeTerm + filterSetIndex * filterNumInSet : freeTerm;
			callParams.StrideWidthBytes = sizeof( float ) * BlockSize * desc.StrideWidth;
			callParams.DilationWidthBytes = sizeof( float ) * BlockSize * desc.DilationWidth;
			callParams.FilterStrideBytes = sizeof( float ) * BlockSize * desc.Filter.ObjectSize();
			callParams.OutputStrideBytes = sizeof( float ) * BlockSize * outputHeight * outputWidth;
			callParams.InputWidthBytes = sizeof( float ) * BlockSize * inputWidth;
			callParams.DilatedInputWidthBytes = sizeof( float ) * BlockSize * dilationHeight * inputWidth;
			callParams.InputStrideBytes = callParams.DilatedInputWidthBytes - filterWidth * callParams.DilationWidthBytes;
			callParams.FilterWidth = filterWidth;
			callParams.OutputColCountLeftPad = outputColLeftPad;
			callParams.OutputColCountNoPad = outputColNoPad;
			callParams.OutputColCountRightPad = outputColRightPad;
			callParams.FilterSetSize = min( MaxFilterSetSize, ( filterCount / BlockSize ) - filterSetIndex * MaxFilterSetSize );

			const int blockedOutputWidth = BlockSize * outputWidth;

			while( workRem > 0 ) {
				const int workThisIter = min( workRem, outputHeight - outputRowIndex );
				for( int ic = 0; ic < inputChannels; ic += BlockSize ) {
					callParams.Flags = ic == 0 ? 0 : CBlockedConvGen::AccumulateOutputFlag;

					if( ic + BlockSize == inputChannels && callParams.FreeTerm != nullptr ) {
						callParams.Flags |= CBlockedConvGen::AddFreeTermFlag;
					}

					const float* input = inputImage + ic * inputHeight * inputWidth;
					callParams.Output = output + outputRowIndex * blockedOutputWidth;

					for( int work = 0; work < workThisIter; ++work ) {
						callParams.Filter = filter + BlockSize * ic * filterHeight * filterWidth;
						callParams.FilterHeight = static_cast<size_t>( filterHeight );

						// Calculate filter intersection with top and bottom padding
						const int firstInputRowIndex = ( outputRowIndex + work ) * desc.StrideHeight - desc.PaddingHeight;
						const int topPaddingFilterRows = min( filterHeight, firstInputRowIndex < 0
							? ( -firstInputRowIndex + dilationHeight - 1 ) / dilationHeight : 0 );
						const int lastInputRowIndex = firstInputRowIndex + ( filterHeight - 1 ) * dilationHeight;
						const int bottomPaddingFilterRows = min( filterHeight, lastInputRowIndex >= inputHeight
							? ( lastInputRowIndex - inputHeight ) / dilationHeight + 1 : 0 );

						// Skip filter rows which are covering top padding 
						callParams.Filter += topPaddingFilterRows * BlockSize * BlockSize * filterWidth;
						// Don't process paddings
						callParams.FilterHeight -= topPaddingFilterRows + bottomPaddingFilterRows;

						// Calculate input pointers accordingly (skip top padding)
						const int inputRowOffset = firstInputRowIndex + topPaddingFilterRows * dilationHeight;
						callParams.Input = input + BlockSize * ( inputRowOffset * inputWidth - desc.PaddingWidth );
						callParams.InputAfterLeftPad = input + BlockSize * ( inputRowOffset * inputWidth );

						blockedConvGen.Run( callParams );

						callParams.Output += blockedOutputWidth;
					}
				}

				workRem -= workThisIter;
				if( outputRowIndex + workThisIter == outputHeight ) {
					size_t blockedFilterCount = BlockSize * callParams.FilterSetSize;
					output += blockedFilterCount * outputHeight * outputWidth;
					filter += blockedFilterCount * inputChannels * filterHeight * filterWidth;
					if( callParams.FreeTerm != nullptr ) {
						callParams.FreeTerm += blockedFilterCount;
					}

					if( ++filterSetIndex == filterSetCount ) {
						inputImage += desc.Source.ObjectSize();
						filter = packedFilter;
						callParams.FreeTerm = freeTerm;
						filterSetIndex = 0;
					}

					callParams.FilterSetSize = min( MaxFilterSetSize, ( filterCount / BlockSize ) - filterSetIndex * MaxFilterSetSize );
					outputRowIndex = 0;
				}
			}

			_mm256_zeroupper();
		}
	}
}

} // namespace NeoML
