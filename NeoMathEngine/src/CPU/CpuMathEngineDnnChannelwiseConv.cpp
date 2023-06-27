/* Copyright © 2017-2023 ABBYY

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

#include <algorithm>

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnConv.h>
#include <CpuMathEnginePrivate.h>
#include <CpuMathEngineDnnRowwise.h>

namespace NeoML {

static constexpr int RowwiseCacheSize = 32 * 1024;

static inline void fillResultRow( const CCommonChannelwiseConvolutionDesc& desc, const float* freeTerm, float* result )
{
	const int channels = desc.Result.Channels();
	int count = desc.Result.Width();
	while( count > 0 ) {
		NeoML::dataCopy( result, freeTerm, channels );
		result += channels;
		count--;
	}
}

static inline void process3x3Row( const CCommonChannelwiseConvolutionDesc& desc, const float* filter, const float* source, float* result )
{
	PRESUME_EXPR( desc.PaddingWidth == 1 );
	PRESUME_EXPR( desc.Filter.Width() == 3 );

	const int resultWidth = desc.Result.Width();
	int width = resultWidth - 1;
	if( desc.StrideWidth == 1 || desc.Source.Width() % 2 == 1 ) {
		--width;
	}
	int channels = desc.Result.Channels();
	const float* filter1 = filter + channels;
	const float* filter2 = filter1 + channels;

	NeoML::vectorEltwiseMultiplyAdd( filter1, source, result, channels );
	if( desc.Source.Width() > 1 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter2, source + channels, result, channels );
	}

	const float* sourcePos = source + ( desc.StrideWidth - 1 ) * channels;
	float* resultPos = result + channels;
	if( desc.StrideWidth == 1 && channels % 4 == 0 ) {
		while( width >= 2 ) {
			channelwise1x3( sourcePos, filter, filter1, filter2, resultPos, channels );
			resultPos += 2 * channels;
			sourcePos += 2 * channels;
			width -= 2;
		}
	}

	while( width > 0 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter1, sourcePos + channels, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter2, sourcePos + 2 * channels, resultPos, channels );
		resultPos += channels;
		sourcePos += desc.StrideWidth * channels;
		width--;
	}

	if( ( desc.StrideWidth == 1 || desc.Source.Width() % 2 == 1 ) && resultWidth > 1 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter1, sourcePos + channels, resultPos, channels );
	}
}

static inline void process5x5RowStride1( const CCommonChannelwiseConvolutionDesc& desc, const float* filter, const float* source, float* result )
{
	PRESUME_EXPR( desc.PaddingWidth == 2 );
	PRESUME_EXPR( desc.Filter.Width() == 5 );

	const int resultWidth = desc.Result.Width();
	int width = resultWidth - 4;
	int channels = desc.Result.Channels();
	const float* filter1 = filter + channels;
	const float* filter2 = filter1 + channels;
	const float* filter3 = filter2 + channels;
	const float* filter4 = filter3 + channels;

	NeoML::vectorEltwiseMultiplyAdd( filter2, source, result, channels );
	if( resultWidth > 1 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter1, source, result + channels, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter3, source + channels, result, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter2, source + channels, result + channels, channels );
		if( resultWidth > 2 ) {
			NeoML::vectorEltwiseMultiplyAdd( filter4, source + 2 * channels, result, channels );
			NeoML::vectorEltwiseMultiplyAdd( filter3, source + 2 * channels, result + channels, channels );
			if( resultWidth > 3 ) {
				NeoML::vectorEltwiseMultiplyAdd( filter4, source + 3 * channels, result + channels, channels );
			}
		}
	}

	const float* sourcePos = source;
	float* resultPos = result + 2 * channels;
#ifdef NEOML_USE_SSE
	if( channels % 4 == 0 ) {
		while( width >= 2 ) {
			channelwise1x5( sourcePos, filter, filter1, filter2, filter3, filter4, resultPos, channels );

			resultPos += 2 * channels;
			sourcePos += 2 * channels;
			width -= 2;
		}
	}
#endif

	while( width > 0 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter + channels, sourcePos + channels, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter + 2 * channels, sourcePos + 2 * channels, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter + 3 * channels, sourcePos + 3 * channels, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter + 4 * channels, sourcePos + 4 * channels, resultPos, channels );
		resultPos += channels;
		sourcePos += channels;
		width--;
	}

	if( resultWidth > 2 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter + channels, sourcePos + channels, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter + 2 * channels, sourcePos + 2 * channels, resultPos, channels );
		if( resultWidth > 3 ) {
			NeoML::vectorEltwiseMultiplyAdd( filter + 3 * channels, sourcePos + 3 * channels, resultPos, channels );
			sourcePos += channels;
			resultPos += channels;
			NeoML::vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
			NeoML::vectorEltwiseMultiplyAdd( filter + channels, sourcePos + channels, resultPos, channels );
			NeoML::vectorEltwiseMultiplyAdd( filter + 2 * channels, sourcePos + 2 * channels, resultPos, channels );
		}
	}
}

static inline void process5x5RowStride2( const CCommonChannelwiseConvolutionDesc& desc, const float* filter, const float* source, float* result )
{
	PRESUME_EXPR( desc.PaddingWidth == 2 );
	PRESUME_EXPR( desc.Filter.Width() == 5 );

	const int resultWidth = desc.Result.Width();
	int width = resultWidth - 2;
	int channels = desc.Result.Channels();
	const float* filter1 = filter + channels;
	const float* filter2 = filter1 + channels;
	const float* filter3 = filter2 + channels;
	const float* filter4 = filter3 + channels;

	NeoML::vectorEltwiseMultiplyAdd( filter2, source, result, channels );
	if( desc.Source.Width() > 1 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter3, source + channels, result, channels );
		if( desc.Source.Width() > 2 ) {
			NeoML::vectorEltwiseMultiplyAdd( filter4, source + 2 * channels, result, channels );
		}
	}

	result += channels;

	while( width > 0 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter, source, result, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter1, source + channels, result, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter2, source + 2 * channels, result, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter3, source + 3 * channels, result, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter4, source + 4 * channels, result, channels );
		result += channels;
		source += 2 * channels;
		width--;
	}

	if( resultWidth > 1 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter, source, result, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter1, source + channels, result, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter2, source + 2 * channels, result, channels );
		if( desc.Source.Width() % 2 == 0 ) {
			NeoML::vectorEltwiseMultiplyAdd( filter3, source + 3 * channels, result, channels );
		}
	}
}

// Calculates `outputRowsToProcess` rows of output of channelwise conv 3x3 with pad 1 and stride 1 or 2
// currInput points to currInputRowIndex'th row of input image
// currOutput points to currOutputRowIndex'th row of output image
static inline void processChannelwise3x3( const CCommonChannelwiseConvolutionDesc& desc, int outputRowsToProcess,
	const float* currInput, int currInputRowIndex, const float* filter, const float* freeTerm, float* currOutput, int currOutputRowIndex )
{
	const int inputHeight = desc.Source.Height();
	const int inputRowSize = desc.Filter.Channels() * desc.Source.Width();
	const int outputRowSize = desc.Filter.Channels() * desc.Result.Width();
	const int filterRowSize = desc.Filter.Channels() * desc.Filter.Width();
	const int stride = desc.StrideHeight;

	float* outputRow = currOutput;
	int remOutputRowsThisStep = outputRowsToProcess;
	const bool processBottomPadding = ( stride == 1 || inputHeight % 2 == 1 )
		&& currOutputRowIndex + outputRowsToProcess == desc.Result.Height()
		&& desc.Result.Height() > 1;
	if( processBottomPadding ) {
		--remOutputRowsThisStep;
	}

	const float* inputRow = currInput + ( currOutputRowIndex * stride - 1 - currInputRowIndex ) * inputRowSize;
	if( currOutputRowIndex == 0 ) {
		// Process top padding
		if( freeTerm == nullptr ) {
			vectorFill0( outputRow, outputRowSize );
		} else {
			fillResultRow( desc, freeTerm, outputRow );
		}

		// Omit first filter row (it's in padding)
		// Processing second and third rows
		process3x3Row( desc, filter + filterRowSize, inputRow + inputRowSize, outputRow );
		// Check corner case: 1x1 input to 3x3 conv with 1x1 padding
		if( inputHeight > 1 ) {
			process3x3Row( desc, filter + 2 * filterRowSize, inputRow + 2 * inputRowSize, outputRow );
		}
		--remOutputRowsThisStep;
		inputRow += stride * inputRowSize;
		outputRow += outputRowSize;
	}

	while( remOutputRowsThisStep > 0 ) {
		if( freeTerm == nullptr ) {
			vectorFill0( outputRow, outputRowSize );
		} else {
			fillResultRow( desc, freeTerm, outputRow );
		}
		// Process all 3 rows without padding checks
		process3x3Row( desc, filter, inputRow, outputRow );
		process3x3Row( desc, filter + filterRowSize, inputRow + inputRowSize, outputRow );
		process3x3Row( desc, filter + 2 * filterRowSize, inputRow + 2 * inputRowSize, outputRow );
		--remOutputRowsThisStep;
		inputRow += stride * inputRowSize;
		outputRow += outputRowSize;
	}

	if( processBottomPadding ) {
		// Process bottom padding
		if( freeTerm == nullptr ) {
			vectorFill0( outputRow, outputRowSize );
		} else {
			fillResultRow( desc, freeTerm, outputRow );
		}
		process3x3Row( desc, filter, inputRow, outputRow );
		process3x3Row( desc, filter + filterRowSize, inputRow + inputRowSize, outputRow );
		// Omit last filter row (it's in bottom padding)
	}
}

// Calculates `outputRowsToProcess` rows of output of channelwise conv 5x5 with pad 2 and stride 1
// currInput points to currInputRowIndex'th row of input image
// currOutput points to currOutputRowIndex'th row of output image
static inline void processChannelwise5x5Stride1( const CCommonChannelwiseConvolutionDesc& desc, int outputRowsToProcess,
	const float* currInput, int currInputRowIndex, const float* filter, const float* freeTerm, float* currOutput, int currOutputRowIndex )
{
	const int inputHeight = desc.Source.Height();
	const int outputHeight = desc.Result.Height();
	const int inputRowSize = desc.Source.Width() * desc.Source.Channels();
	const int filterRowSize = desc.Filter.Width() * desc.Filter.Channels();
	const int outputRowSize = desc.Result.Width() * desc.Result.Channels();

	auto initOutputRow = [&] () {
		if( freeTerm != nullptr ) {
			fillResultRow( desc, freeTerm, currOutput );
		} else {
			NeoML::vectorFill( currOutput, 0, outputRowSize );
		}
	};
	auto switchToNextOutputRow = [&] () {
		currOutput += outputRowSize;
		currOutputRowIndex++;
		outputRowsToProcess--;
	};

	if( currOutputRowIndex == 0 ) {
		PRESUME_EXPR( currInputRowIndex == 0 );
		initOutputRow();
		process5x5RowStride1( desc, filter + 2 * filterRowSize, currInput, currOutput );
		if( inputHeight > 1 ) {
			process5x5RowStride1( desc, filter + 3 * filterRowSize, currInput + inputRowSize, currOutput );
			if( inputHeight > 2 ) {
				process5x5RowStride1( desc, filter + 4 * filterRowSize, currInput + 2 * inputRowSize, currOutput );
			}
		}
		switchToNextOutputRow();
	}

	if( currOutputRowIndex == 1 && outputRowsToProcess > 0 ) {
		PRESUME_EXPR( currInputRowIndex == 0 );
		initOutputRow();
		process5x5RowStride1( desc, filter + filterRowSize, currInput, currOutput );
		process5x5RowStride1( desc, filter + 2 * filterRowSize, currInput + inputRowSize, currOutput );
		if( inputHeight > 2 ) {
			process5x5RowStride1( desc, filter + 3 * filterRowSize, currInput + 2 * inputRowSize, currOutput );
			if( inputHeight > 3 ) {
				process5x5RowStride1( desc, filter + 4 * filterRowSize, currInput + 3 * inputRowSize, currOutput );
			}
		}
		switchToNextOutputRow();
	}

	auto hasRowsWithoutPadding = [&] { return outputRowsToProcess > 0 && currOutputRowIndex < outputHeight - 2; };
	for( const float* firstInputUnderFilter = currInput + ( currOutputRowIndex - 2 - currInputRowIndex ) * inputRowSize;
		hasRowsWithoutPadding();
		firstInputUnderFilter += inputRowSize )
	{
		initOutputRow();
		process5x5RowStride1( desc, filter + 0 * filterRowSize, firstInputUnderFilter + 0 * inputRowSize, currOutput );
		process5x5RowStride1( desc, filter + 1 * filterRowSize, firstInputUnderFilter + 1 * inputRowSize, currOutput );
		process5x5RowStride1( desc, filter + 2 * filterRowSize, firstInputUnderFilter + 2 * inputRowSize, currOutput );
		process5x5RowStride1( desc, filter + 3 * filterRowSize, firstInputUnderFilter + 3 * inputRowSize, currOutput );
		process5x5RowStride1( desc, filter + 4 * filterRowSize, firstInputUnderFilter + 4 * inputRowSize, currOutput );
		switchToNextOutputRow();
	}

	if( outputRowsToProcess > 0 && inputHeight > 3 && currOutputRowIndex == outputHeight - 2 ) {
		initOutputRow();
		const float* firstInputUnderFilter = currInput + ( currOutputRowIndex - 2 - currInputRowIndex ) * inputRowSize;
		process5x5RowStride1( desc, filter + 0 * filterRowSize, firstInputUnderFilter + 0 * inputRowSize, currOutput );
		process5x5RowStride1( desc, filter + 1 * filterRowSize, firstInputUnderFilter + 1 * inputRowSize, currOutput );
		process5x5RowStride1( desc, filter + 2 * filterRowSize, firstInputUnderFilter + 2 * inputRowSize, currOutput );
		process5x5RowStride1( desc, filter + 3 * filterRowSize, firstInputUnderFilter + 3 * inputRowSize, currOutput );
		switchToNextOutputRow();
	}

	if( outputRowsToProcess > 0 && inputHeight > 2 && currOutputRowIndex == outputHeight - 1 ) {
		initOutputRow();
		const float* firstInputUnderFilter = currInput + ( currOutputRowIndex - 2 - currInputRowIndex ) * inputRowSize;
		process5x5RowStride1( desc, filter + 0 * filterRowSize, firstInputUnderFilter + 0 * inputRowSize, currOutput );
		process5x5RowStride1( desc, filter + 1 * filterRowSize, firstInputUnderFilter + 1 * inputRowSize, currOutput );
		process5x5RowStride1( desc, filter + 2 * filterRowSize, firstInputUnderFilter + 2 * inputRowSize, currOutput );
	}
}

// Calculates `outputRowsToProcess` rows of output of channelwise conv 5x5 with pad 2 and stride 2
// currInput points to currInputRowIndex'th row of input image
// currOutput points to currOutputRowIndex'th row of output image
static inline void processChannelwise5x5Stride2( const CCommonChannelwiseConvolutionDesc& desc, int outputRowsToProcess,
	const float* currInput, int currInputRowIndex, const float* filter, const float* freeTerm, float* currOutput, int currOutputRowIndex )
{
	const int inputHeight = desc.Source.Height();
	const int outputHeight = desc.Result.Height();
	const int inputRowSize = desc.Source.Width() * desc.Source.Channels();
	const int filterRowSize = desc.Filter.Width() * desc.Filter.Channels();
	const int outputRowSize = desc.Result.Width() * desc.Result.Channels();

	auto initOutputRow = [&] () {
		if( freeTerm != nullptr ) {
			fillResultRow( desc, freeTerm, currOutput );
		} else {
			NeoML::vectorFill( currOutput, 0, outputRowSize );
		}
	};
	auto switchToNextOutputRow = [&] () {
		currOutput += outputRowSize;
		currOutputRowIndex++;
		outputRowsToProcess--;
	};

	if( currOutputRowIndex == 0 ) {
		PRESUME_EXPR( currInputRowIndex == 0 );
		initOutputRow();
		process5x5RowStride2( desc, filter + 2 * filterRowSize, currInput, currOutput );
		if( inputHeight > 1 ) {
			process5x5RowStride2( desc, filter + 3 * filterRowSize, currInput + inputRowSize, currOutput );
			if( inputHeight > 2 ) {
				process5x5RowStride2( desc, filter + 4 * filterRowSize, currInput + 2 * inputRowSize, currOutput );
			}
		}
		switchToNextOutputRow();
	}

	auto hasRowsWithoutPadding = [&] { return outputRowsToProcess > 0 && currOutputRowIndex < outputHeight - 1; };
	for( const float* firstInputUnderFilter = currInput + ( currOutputRowIndex * 2 - 2 - currInputRowIndex ) * inputRowSize;
		hasRowsWithoutPadding();
		firstInputUnderFilter += 2 * inputRowSize )
	{
		initOutputRow();
		process5x5RowStride2( desc, filter + 0 * filterRowSize, firstInputUnderFilter + 0 * inputRowSize, currOutput );
		process5x5RowStride2( desc, filter + 1 * filterRowSize, firstInputUnderFilter + 1 * inputRowSize, currOutput );
		process5x5RowStride2( desc, filter + 2 * filterRowSize, firstInputUnderFilter + 2 * inputRowSize, currOutput );
		process5x5RowStride2( desc, filter + 3 * filterRowSize, firstInputUnderFilter + 3 * inputRowSize, currOutput );
		process5x5RowStride2( desc, filter + 4 * filterRowSize, firstInputUnderFilter + 4 * inputRowSize, currOutput );
		switchToNextOutputRow();
	}

	if( outputRowsToProcess > 0 && currOutputRowIndex == outputHeight - 1 ) {
		initOutputRow();
		const float* firstInputUnderFilter = currInput + ( currOutputRowIndex * 2 - 2 - currInputRowIndex ) * inputRowSize;
		process5x5RowStride2( desc, filter + 0 * filterRowSize, firstInputUnderFilter + 0 * inputRowSize, currOutput );
		process5x5RowStride2( desc, filter + 1 * filterRowSize, firstInputUnderFilter + 1 * inputRowSize, currOutput );
		process5x5RowStride2( desc, filter + 2 * filterRowSize, firstInputUnderFilter + 2 * inputRowSize, currOutput );
		if( inputHeight % 2 == 0 ) {
			process5x5RowStride2( desc, filter + 3 * filterRowSize, firstInputUnderFilter + 3 * inputRowSize, currOutput );
		}
	}
}

typedef void (*TChannelwiseProcessFunction)( const CCommonChannelwiseConvolutionDesc&, int, const float*, int,
	const float*, const float*, float*, int );

void CCpuMathEngine::BlobChannelwiseConvolution( const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	CCpuExecutionScope scope;
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );

	const float* source = GetRaw( sourceData );
	const float* filter = GetRaw( filterData );
	const float* freeTerm = freeTermData != 0 ? GetRaw( *freeTermData ) : 0;
	float* result = GetRaw( resultData );

	// Specify process function if it's a special case
	TChannelwiseProcessFunction process = nullptr;
	if( desc.Filter.Height() == desc.Filter.Width() && desc.PaddingHeight == desc.PaddingWidth
		&& desc.StrideHeight == desc.StrideWidth )
	{
		if( desc.Filter.Width() == 3 && desc.PaddingWidth == 1 && desc.StrideWidth <= 2 ) {
			process = processChannelwise3x3;
		} else if( desc.Filter.Width() == 5 && desc.PaddingWidth == 2 ) {
			if( desc.StrideWidth == 1 ) {
				process = processChannelwise5x5Stride1;
			} else if( desc.StrideWidth == 2 ) {
				process = processChannelwise5x5Stride2;
			}
		}
	}

	const CBlobDesc& sourceDesc = desc.Source;
	const CBlobDesc& filterDesc = desc.Filter;
	const CBlobDesc& resultDesc = desc.Result;

	const int curThreadCount = IsOmpRelevant( sourceDesc.ObjectCount() * resultDesc.Height(),
		static_cast<int64_t>( sourceDesc.BlobSize() ) * filterDesc.BlobSize() ) ? threadCount : 1;

	const int channels = sourceDesc.Channels() * sourceDesc.Depth();

	const int inputRowSize = sourceDesc.Width() * channels;
	const int outputRowSize = resultDesc.Width() * channels;
	const int filterRowSize = filterDesc.Width() * channels;

	const int inputObjectSize = inputRowSize * sourceDesc.Height();
	const int outputObjectSize = outputRowSize * resultDesc.Height();

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int batchStart;
		int batchCount;
		int resultStart;
		int resultCount;
		if( OmpGetTaskIndexAndCount2D( sourceDesc.ObjectCount(), resultDesc.Height(), batchStart, batchCount, resultStart, resultCount ) ) {
			const float* src = source + batchStart * inputObjectSize;
			const float* srcEnd = src + batchCount * inputObjectSize;
			float* res = result + batchStart * outputObjectSize + resultStart * resultDesc.Width() * channels;

			for( ; src < srcEnd; src += inputObjectSize, res += outputObjectSize ) {
				if( process != nullptr ) {
					process( desc, resultCount, src, 0, filter, freeTerm, res, resultStart );
					continue;
				}

				float* resultRow = res;
				float* resultRowEnd = resultRow + resultCount * outputRowSize;
				int firstFilteredRow = resultStart * desc.StrideHeight - desc.PaddingHeight;
				for( ; resultRow < resultRowEnd; resultRow += outputRowSize, firstFilteredRow += desc.StrideHeight ) {
					if( freeTerm != 0 ) {
						float* rowStart = resultRow;
						float* rowEnd = rowStart + channels * resultDesc.Width();
						for( ; rowStart < rowEnd; rowStart += channels ) {
							NeoML::dataCopy(rowStart, freeTerm, channels);
						}
					} else {
						NeoML::vectorFill(resultRow, 0, resultDesc.Width() * channels);
					}

					const int filterFirstRow = std::max( 0, -firstFilteredRow );
					const int filterLastRow = std::min( filterDesc.Height(), sourceDesc.Height() - firstFilteredRow );
					const float* filterRow = filter + filterFirstRow * filterRowSize;
					const float* filterRowEnd = filter + filterLastRow * filterRowSize;
					const float* srcRow = src + (firstFilteredRow + filterFirstRow) * inputRowSize;

					for( ; filterRow < filterRowEnd; filterRow += filterRowSize, srcRow += inputRowSize ) {
						int firstFilteredCol = -desc.PaddingWidth;
						float* resultPos = resultRow;
						float* resultPosEnd = resultPos + channels * resultDesc.Width();
						for( ; resultPos < resultPosEnd; resultPos += channels, firstFilteredCol += desc.StrideWidth ) {
							const int filterFirstCol = std::max( 0, -firstFilteredCol );
							const int filterLastCol = std::min( filterDesc.Width(), sourceDesc.Width() - firstFilteredCol );
							const float* filterPos = filterRow + filterFirstCol * channels;
							const float* filterPosEnd = filterRow + filterLastCol * channels;
							const float* srcPos = srcRow + (firstFilteredCol + filterFirstCol) * channels;
							for( ; filterPos < filterPosEnd; filterPos += channels, srcPos += channels ) {
								NeoML::vectorEltwiseMultiplyAdd(filterPos, srcPos, resultPos, channels);
							}
						}
					}
				}
			}
		}
	}
}

//=====================================================================================================================

void CCpuMathEngine::multiplyMatrixByTransposedWithFreeTerm( const float* first, int firstHeight,
	int firstWidth, const float* second, int secondHeight, const float* freeTerm, float* result )
{
	multiplyMatrixByTransposedMatrix( first, firstHeight, firstWidth, firstWidth, second,
		secondHeight, firstWidth, result, secondHeight );
	if( freeTerm != nullptr ) {
		addVectorToMatrixRows( result, result, firstHeight, secondHeight, secondHeight,
			secondHeight, freeTerm );
	}
}

void CCpuMathEngine::MobileNetV3PreSEBlock( const CBlobDesc& inputDesc, const CBlobDesc& outputDesc,
	const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& inputHandle,
	const CConstFloatHandle& expandFilterData, const CConstFloatHandle* expandFreeTermData,
	TActivationFunction expandActivation, float expandActivationParam, const CConstFloatHandle& channelwiseFilterData,
	const CConstFloatHandle* channelwiseFreeTermData, TActivationFunction channelwiseActivation,
	float channelwiseActivationParam, const CFloatHandle& outputHandle )
{
	CCpuExecutionScope scope;
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );

	const int inputChannels = inputDesc.Channels();
	const int outputChannels = outputDesc.Channels();
	const int inputHeight = desc.Source.Height();
	const int inputWidth = desc.Source.Width();
	const int chInputRowSize = outputChannels * inputWidth;
	const int inputRowSize = inputChannels * inputWidth;
	const int outputHeight = desc.Result.Height();
	const int outputRowSize = outputChannels * desc.Result.Width();
	const int padding = desc.PaddingHeight;
	const int filterSize = desc.Filter.Width();
	const int stride = desc.StrideHeight;

	const float* inputObject = GetRaw( inputHandle );
	const float* expandFilter = GetRaw( expandFilterData );
	const float* expandFreeTerm = expandFreeTermData == nullptr ? nullptr : GetRaw( *expandFreeTermData );
	CRowwiseActivation expandActivationImpl( expandActivation, expandActivationParam, 0.f );
	expandActivationImpl.Reshape( desc.Source );

	TChannelwiseProcessFunction channelwise = filterSize == 3 ? processChannelwise3x3
		: ( stride == 1 ? processChannelwise5x5Stride1 : processChannelwise5x5Stride2 );
	const float* channelwiseFilter = GetRaw( channelwiseFilterData );
	const float* channelwiseFreeTerm = channelwiseFreeTermData == nullptr ? nullptr : GetRaw( *channelwiseFreeTermData );
	CRowwiseActivation channelwiseActivationImpl( channelwiseActivation, channelwiseActivationParam, 0.f );
	channelwiseActivationImpl.Reshape( desc.Result );

	const int maxInputRowsPerStep = std::max<int>( 1,
		( RowwiseCacheSize / ( std::max<int>( inputChannels, outputChannels ) * inputWidth ) ) );
	const int maxOutputRowsPerStep = std::max<int>( 1, ( RowwiseCacheSize / ( outputChannels * desc.Result.Width() ) ) );

	// Buffer for the input rows of channelwise convolution
	CFloatHandleStackVar chInputBuffVar( *this,
		std::min<int>( inputHeight, maxInputRowsPerStep + filterSize - 1 ) * chInputRowSize );

	float* chInputBuff = GetRaw( chInputBuffVar.GetHandle() );
	float* outputObject = GetRaw( outputHandle );

	for( int b = 0; b < inputDesc.ObjectCount(); ++b ) {
		const float* input = inputObject;
		float* output = outputObject;

		int inputRowsProcessed = 0;
		int outputRowsProcessed = 0;
		// The channelwise input row buffer can't hold the full image
		// That's why the buffer on each step contains [firstInputRowInBuffer, inputRowsProcessed) rows
		int firstInputRowInBuffer = 0;

		while( inputRowsProcessed < inputHeight ) {
			// Process a bunch of rows of input image (till channelwise convolution: expandConv + expandReLU)
			const int inputRowsThisStep = std::min<int>( maxInputRowsPerStep, inputHeight - inputRowsProcessed );
			float* chInput = chInputBuff + ( inputRowsProcessed - firstInputRowInBuffer ) * chInputRowSize;

			// Apply expand convolution
			multiplyMatrixByTransposedWithFreeTerm( input, inputRowsThisStep * inputWidth, inputChannels,
				expandFilter, outputChannels, expandFreeTerm, chInput );
			expandActivationImpl.Process( chInput, inputRowsProcessed, inputRowsThisStep,
				chInput, inputRowsProcessed, inputRowsThisStep, nullptr );
			inputRowsProcessed += inputRowsThisStep;

			// Calculate how many output rows we can calculate with the processed input rows
			const int outputRowsCanBeProcesed = inputRowsProcessed == inputHeight ? outputHeight
				: ( inputRowsProcessed < ( filterSize - padding ) ? 0 : ( inputRowsProcessed - filterSize + padding ) / stride + 1 );

			while( outputRowsProcessed < outputRowsCanBeProcesed ) {
				// Process channelwise output rows (while there are any)
				const int outputRowsThisStep = std::min<int>( maxOutputRowsPerStep,
					outputRowsCanBeProcesed - outputRowsProcessed );

				// Channelwise conv
				channelwise( desc, outputRowsThisStep, chInputBuff, firstInputRowInBuffer,
					channelwiseFilter, channelwiseFreeTerm, output, outputRowsProcessed );
				channelwiseActivationImpl.Process( output, outputRowsProcessed, outputRowsThisStep,
					output, outputRowsProcessed, outputRowsThisStep, nullptr );

				output += outputRowsThisStep * outputRowSize;
				outputRowsProcessed += outputRowsThisStep;
			}

			input += inputRowsThisStep * inputRowSize;

			if( outputRowsProcessed < outputHeight ) {
				const int firstInputRowNeeded = outputRowsProcessed * stride - padding;
				if( firstInputRowNeeded > firstInputRowInBuffer ) {
					// Buffer for channelwise input contains rows that won't be used in future
					const int rowsToDelete = firstInputRowNeeded - firstInputRowInBuffer;
					const int rowsToMove = inputRowsProcessed - firstInputRowNeeded;
					if( rowsToMove > 0 ) {
						// There are rows which should be saved between iteration over input
						// Move them to the beginning of the buffer
						dataCopy( chInputBuff, chInputBuff + rowsToDelete * chInputRowSize,
							rowsToMove * chInputRowSize );
					}
					// Mark that channelwise input buffer now starts with a new row
					firstInputRowInBuffer = firstInputRowNeeded;
				}
			}
		}

		inputObject += inputDesc.ObjectSize();
		outputObject += outputDesc.ObjectSize();
	}
}

void CCpuMathEngine::MobileNetV3PostSEBlock( const CBlobDesc& channelwiseOutputDesc, int outputChannels,
	const CConstFloatHandle& channelwiseOutputHandle, const CConstFloatHandle& squeezeAndExciteHandle,
	const CConstFloatHandle* residualHandle, TActivationFunction activation, float activationParam,
	const CConstFloatHandle& downFilterHandle, const CConstFloatHandle* downFreeTermHandle,
	const CFloatHandle& outputHandle )
{
	CCpuExecutionScope scope;
	const int inputChannels = channelwiseOutputDesc.Channels();
	const int width = channelwiseOutputDesc.Width();
	const int rowCount = channelwiseOutputDesc.Height();
	const int inputRowSize = inputChannels * width;
	const int outputRowSize = outputChannels * width;
	const int outputObjectSize = outputRowSize * rowCount;
	CRowwiseActivation activationImpl( activation, activationParam, 0.f );
	activationImpl.Reshape( channelwiseOutputDesc );

	const int maxRowsPerStep = std::max( 1, ( RowwiseCacheSize / ( std::max( inputChannels, outputChannels ) * width ) ) );

	CFloatHandleStackVar buffVar( *this, std::min( rowCount, maxRowsPerStep ) * inputRowSize );
	const float* inputObject = GetRaw( channelwiseOutputHandle );
	const float* squeezeVector = GetRaw( squeezeAndExciteHandle );
	const float* residualObject = residualHandle != nullptr ? GetRaw( *residualHandle ) : nullptr;
	const float* downFilter = GetRaw( downFilterHandle );
	const float* downFreeTerm = downFreeTermHandle != nullptr ? GetRaw( *downFreeTermHandle ) : nullptr;
	float* squeezed = GetRaw( buffVar.GetHandle() );
	float* outputObject = GetRaw( outputHandle );

	for( int b = 0; b < channelwiseOutputDesc.ObjectCount(); ++b ) {
		int rowsProcessed = 0;
		const float* input = inputObject;
		float* output = outputObject;
		const float* residual = residualObject;
		while( rowsProcessed < rowCount ) {
			const int rowsThisStep = std::min( rowCount - rowsProcessed, maxRowsPerStep );
			// Multiply by vector from Squeeze-and-Excite
			multiplyMatrixByDiagMatrix( input, rowsThisStep * width, inputChannels,
				squeezeVector, squeezed );
			// Activation (if present, not present means trivial linear)
			activationImpl.Process( squeezed, rowsProcessed, rowsThisStep,
				squeezed, rowsProcessed, rowsThisStep, nullptr );
			// Down-convolution (1x1)
			multiplyMatrixByTransposedWithFreeTerm( squeezed, rowsThisStep * width, inputChannels,
				downFilter, outputChannels, downFreeTerm, output );
			// Residual connection (if present)
			if( residual != nullptr ) {
				vectorAdd( output, residual, output, rowsThisStep * width * outputChannels );
				residual += rowsThisStep * outputRowSize;
			}

			rowsProcessed += rowsThisStep;
			input += rowsThisStep * inputRowSize;
			output += rowsThisStep * outputRowSize;
		}

		inputObject += inputRowSize * rowCount;
		squeezeVector += inputChannels;
		if( residualObject != nullptr ) {
			residualObject += outputObjectSize;
		}
		outputObject += outputObjectSize;
	}
}

//=====================================================================================================================

class CCpuMathEngine::CRowwiseChannelwiseWith1x1 : public IRowwiseCpuImpl, public CRowwiseOperationDesc {
public:
	CRowwiseChannelwiseWith1x1( CCpuMathEngine& mathEngine, int stride, const float* chFilter, const float* chFreeTerm,
			TActivationFunction activation, float activationParam, const float* convFilter,
			const float* convFreeTerm, int outputChannels, bool residual ) :
		mathEngine( mathEngine ),
		chFilter( chFilter ),
		chFreeTerm( chFreeTerm ),
		activationImpl( activation, activationParam, 0.f ),
		convFilter( convFilter ),
		convFreeTerm( convFreeTerm ),
		residual( residual ),
		outputChannels( outputChannels ),
		desc( 1, 1, stride, stride, CBlobDesc(), CBlobDesc(), CBlobDesc() )
	{
	}

	int MinInputRowCount() const { return 3; }
	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int InOperationBufferSize() const override
		{ return desc.Result.Channels() * desc.Result.Width() * maxOutputRowsPerStep(); }
	int OutputRowCount() const override { return desc.Result.ObjectCount() * desc.Result.Height(); }
	int OutputRowSize() const override { return desc.Result.Width() * outputChannels; }
	bool IsInPlace() const override { return false; }
	CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const override;

private:
	CCpuMathEngine& mathEngine;
	const float* chFilter;
	const float* chFreeTerm;
	CRowwiseActivation activationImpl;
	const float* convFilter;
	const float* convFreeTerm;
	int outputChannels;
	bool residual;
	CCommonChannelwiseConvolutionDesc desc;

	int maxOutputRowsPerStep() const;
};

CBlobDesc CCpuMathEngine::CRowwiseChannelwiseWith1x1::Reshape( const CBlobDesc& inputSize )
{
	CBlobDesc outputSize = inputSize;
	outputSize.SetDimSize( BD_Height, 1 + ( outputSize.Height() - 1 ) / desc.StrideHeight );
	outputSize.SetDimSize( BD_Width, 1 + ( outputSize.Width() - 1 ) / desc.StrideWidth );
	CBlobDesc filterSize( CT_Float );
	filterSize.SetDimSize( BD_Height, 3 );
	filterSize.SetDimSize( BD_Width, 3 );
	filterSize.SetDimSize( BD_Channels, inputSize.Channels() );
	desc = CCommonChannelwiseConvolutionDesc( desc.PaddingHeight, desc.PaddingWidth, desc.StrideHeight,
		desc.StrideWidth, inputSize, filterSize, outputSize );
	activationImpl.Reshape( outputSize );
	outputSize.SetDimSize( BD_Channels, outputChannels );
	return outputSize;
}

int CCpuMathEngine::CRowwiseChannelwiseWith1x1::maxOutputRowsPerStep() const
{
	const int maxRowSize = std::max( desc.Result.Channels(), desc.Source.Channels() ) * desc.Result.Width();
	return std::min( std::max( RowwiseCacheSize / maxRowSize, 1 ), desc.Result.Height() );
}

IRowwiseCpuImpl::CProcessingReport CCpuMathEngine::CRowwiseChannelwiseWith1x1::Process( const float* input,
	int inputRowIndex, int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable,
	float* buffer ) const
{
	PRESUME_EXPR( !residual || inputRowIndex <= outputRowIndex );
	CProcessingReport report = RowwiseConvProcessingReport( inputRowIndex, inputRowsAvailable, outputRowIndex,
		outputRowsAvailable, desc.Source.Height(), desc.Result.Height(), desc.Filter.Height(), desc.PaddingHeight,
		desc.StrideHeight, 1 );
	if( report.OutputRowsCalculated == 0 ) {
		PRESUME_EXPR( report.InputRowsMayBeRemoved == 0 );
		return report;
	}

	const int outputRowsThisCall = outputRowIndex + report.OutputRowsCalculated;
	const int maxRowsPerStep = maxOutputRowsPerStep();
	const int chOutputRowSize = desc.Result.Channels() * desc.Result.Width();
	const int outputWidth = desc.Result.Width();
	const int inputChannels = desc.Source.Channels();

	const float* residualInput = input + ( outputRowIndex - inputRowIndex ) * desc.Source.Width() * desc.Source.Channels();
	while( outputRowIndex < outputRowsThisCall ) {
		// Process channelwise output rows (while there are any)
		const int outputImageRowIndex = outputRowIndex % desc.Result.Height();
		const int outputRowsThisStep = std::min( maxRowsPerStep,
			std::min( desc.Result.Height() - outputImageRowIndex, outputRowsThisCall - outputRowIndex ) );

		processChannelwise3x3( desc, outputRowsThisStep, input, inputRowIndex % desc.Source.Height(),
			chFilter, chFreeTerm, buffer, outputImageRowIndex );
		activationImpl.Process( buffer, outputRowIndex, outputRowsThisStep,
			buffer, outputRowIndex, outputRowsThisStep, nullptr );
		mathEngine.multiplyMatrixByTransposedWithFreeTerm( buffer, outputRowsThisStep * outputWidth, inputChannels,
			convFilter, outputChannels, convFreeTerm, output );
		if( residual ) {
			vectorAdd( output, residualInput, output, outputRowsThisStep * outputWidth * outputChannels );
			residualInput += outputRowsThisStep * desc.Source.Width() * inputChannels;
		}

		output += outputRowsThisStep * outputChannels * outputWidth;
		outputRowIndex += outputRowsThisStep;

		if( outputRowIndex % desc.Result.Height() == 0 && outputRowIndex < outputRowsThisCall ) {
			// Switch to the next image in batch
			const int imageIndex = outputRowIndex / desc.Result.Height();
			const int diff = imageIndex * desc.Source.Height() - inputRowIndex;
			PRESUME_EXPR( diff >= 0 );
			input += diff * desc.Source.Width() * inputChannels;
			inputRowIndex += diff;
			inputRowsAvailable -= diff;
		}
	}

	return report;
}

void CCpuMathEngine::ChannelwiseWith1x1( const CBlobDesc& inputDesc, const CBlobDesc& outputDesc,
	const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& inputHandle,
	const CConstFloatHandle& channelwiseFilterData, const CConstFloatHandle* channelwiseFreeTermData,
	TActivationFunction activation, float activationParam, const CConstFloatHandle& convFilterData,
	const CConstFloatHandle* convFreeTermData, bool residual, const CFloatHandle& outputHandle )
{
	CCpuExecutionScope scope;
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );

	CRowwiseChannelwiseWith1x1 impl( *this, desc.StrideHeight, GetRaw( channelwiseFilterData ),
		channelwiseFreeTermData == nullptr ? nullptr : GetRaw( *channelwiseFreeTermData ),
		activation, activationParam, GetRaw( convFilterData ),
		convFreeTermData == nullptr ? nullptr : GetRaw( *convFreeTermData ),
		outputDesc.Channels(), residual );
	(void)impl.Reshape( inputDesc );

	// Buffer for the output rows of channelwise convolution
	CFloatHandleStackVar bufferVar( *this, static_cast<size_t>( impl.InOperationBufferSize() ) );

	float* buffer = GetRaw( bufferVar.GetHandle() );
	const float* input = GetRaw( inputHandle );
	float* output = GetRaw( outputHandle );

	const int inputRowCount = desc.Source.ObjectCount() * desc.Source.Height();
	const int outputRowCount = desc.Result.ObjectCount() * desc.Result.Height();
	IRowwiseCpuImpl::CProcessingReport report = impl.Process( input, 0, inputRowCount,
		output, 0, outputRowCount, buffer );
	( void ) report; // Avoid compiler warning in release configuration
	PRESUME_EXPR( report.InputRowsMayBeRemoved == inputRowCount );
	PRESUME_EXPR( report.OutputRowsCalculated == outputRowCount );
}

CRowwiseOperationDesc* CCpuMathEngine::InitChannelwiseWith1x1Rowwise( int stride, const CConstFloatHandle& channelwiseFilter,
	const CConstFloatHandle* channelwiseFreeTerm, TActivationFunction activation, float activationParam,
	const CConstFloatHandle& convFilter, const CConstFloatHandle* convFreeTerm, int outputChannels, bool residual )
{
	return new CRowwiseChannelwiseWith1x1( *this, stride, GetRaw( channelwiseFilter ),
		channelwiseFreeTerm == nullptr ? nullptr : GetRaw( *channelwiseFreeTerm ),
		activation, activationParam, GetRaw( convFilter ),
		convFreeTerm == nullptr ? nullptr : GetRaw( *convFreeTerm ),
		outputChannels, residual );
}

//=====================================================================================================================

class CCpuMathEngine::CRowwiseMobileNetV2 : public IRowwiseCpuImpl, public CRowwiseOperationDesc {
public:
	CRowwiseMobileNetV2( CCpuMathEngine& mathEngine, int inputChannels,
			const float* expandFilter, const float* expandFreeTerm, int expandedChannels,
			TActivationFunction expandActivation, float expandActivationParam,
			const float* channelwiseFilter, const float* channelwiseFreeTerm, int stride,
			TActivationFunction channelwiseActivation, float channelwiseActivationParam,
			const float* downFilter, const float* downFreeTerm, int outputChannels, bool residual ) :
		mathEngine( mathEngine ),
		inputChannels( inputChannels ),
		expandFilter( expandFilter ),
		expandFreeTerm( expandFreeTerm ),
		expandedChannels( expandedChannels ),
		expandActivationImpl( expandActivation, expandActivationParam, 0.f ),
		desc( 1, 1, stride, stride, CBlobDesc(), CBlobDesc( { 3, 3, 1, expandedChannels } ), CBlobDesc() ),
		channelwiseFilter( channelwiseFilter ),
		channelwiseFreeTerm( channelwiseFreeTerm ),
		channelwiseActivationImpl( channelwiseActivation, channelwiseActivationParam, 0.f ),
		downFilter( downFilter ),
		downFreeTerm( downFreeTerm ),
		outputChannels( outputChannels ),
		residual( residual )
	{
	}

	int MinInputRowCount() const override { return 3; }

	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int InOperationBufferSize() const override { return 0; }
	int OutputRowCount() const override { return desc.Result.Height(); }
	int OutputRowSize() const override { return desc.Result.Width() * outputChannels; }
	bool IsInPlace() const override { return false; }
	CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const override;

private:
	CCpuMathEngine& mathEngine;
	int inputChannels;
	const float* expandFilter;
	const float* expandFreeTerm;
	int expandedChannels;
	CRowwiseActivation expandActivationImpl;
	CCommonChannelwiseConvolutionDesc desc;
	const float* channelwiseFilter;
	const float* channelwiseFreeTerm;
	CRowwiseActivation channelwiseActivationImpl;
	const float* downFilter;
	const float* downFreeTerm;
	int outputChannels;
	bool residual;
	mutable std::unique_ptr<CRowwiseBuffer> chInput;
	mutable std::unique_ptr<CRowwiseBuffer> chOutput;

	int getMaxInputRowsPerStep() const { return std::max<int>( 1,
		( RowwiseCacheSize / ( std::max<int>( inputChannels, expandedChannels ) * desc.Source.Width() ) ) ); }
	int getMaxOutputRowsPerStep() const { return std::max<int>( 1,
		( RowwiseCacheSize / ( std::max<int>( outputChannels, expandedChannels ) * desc.Result.Width() ) ) ); }
};

CBlobDesc CCpuMathEngine::CRowwiseMobileNetV2::Reshape( const CBlobDesc& inputSize )
{
	CBlobDesc chInputSize = inputSize;
	chInputSize.SetDimSize( BD_Channels, expandedChannels );
	expandActivationImpl.Reshape( chInputSize );
	CBlobDesc outputSize = chInputSize;
	if( desc.StrideHeight == 2 ) {
		outputSize.SetDimSize( BD_Height, ( outputSize.Height() + 1 ) / 2 );
		outputSize.SetDimSize( BD_Width, ( outputSize.Width() + 1 ) / 2 );
	}
	desc = CCommonChannelwiseConvolutionDesc( desc.PaddingHeight, desc.PaddingWidth, desc.StrideHeight, desc.StrideWidth,
		chInputSize, desc.Filter, outputSize );
	channelwiseActivationImpl.Reshape( outputSize );
	outputSize.SetDimSize( BD_Channels, outputChannels );
	return outputSize;
}

IRowwiseCpuImpl::CProcessingReport CCpuMathEngine::CRowwiseMobileNetV2::Process( const float* input, int inputRowIndex,
	int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable, float* ) const
{
	CProcessingReport report = RowwiseConvProcessingReport( inputRowIndex, inputRowsAvailable, outputRowIndex,
		outputRowsAvailable, desc.Source.Height(), desc.Result.Height(), desc.Filter.Height(), desc.PaddingHeight,
		desc.StrideHeight, 1 );
	if( report.OutputRowsCalculated == 0 ) {
		PRESUME_EXPR( report.InputRowsMayBeRemoved == 0 );
		return report;
	}

	if( chInput == nullptr ) {
		chInput.reset( new CRowwiseBuffer( mathEngine,
			std::min( desc.Source.Height(), getMaxInputRowsPerStep() + 2 ),
			desc.Source.Width() * expandedChannels, desc.Source.Height() * desc.Source.ObjectCount() ) );
		chOutput.reset( new CRowwiseBuffer( mathEngine, getMaxOutputRowsPerStep(),
			desc.Result.Width() * expandedChannels, desc.Result.Height() * desc.Result.ObjectCount() ) );
	}

	PRESUME_EXPR( chOutput->DataRowProcessed() == outputRowIndex );
	PRESUME_EXPR( chInput->DataRowProcessed() >= inputRowIndex );
	PRESUME_EXPR( chInput->DataRowProcessed() <= inputRowIndex + inputRowsAvailable );
	PRESUME_EXPR( chInput->DataRowCount() <= 3 - desc.StrideHeight );
	PRESUME_EXPR( chInput->DataRowIndex() == getFirstRequiredInputRow( outputRowIndex, desc ) );

	const int inputWidth = desc.Source.Width();
	const int outputWidth = desc.Result.Width();
	const int inputRowSize = inputWidth * inputChannels;
	const int chInputRowSize = inputWidth * expandedChannels;
	const int chOutputRowSize = outputWidth * expandedChannels;

	const float* residualInput = input + ( outputRowIndex - inputRowIndex ) * inputRowSize;
	const bool isInPlace = ( residualInput == output );

	const int outputRowsThisCall = outputRowIndex + report.OutputRowsCalculated;
	// Total number of input rows used during this call
	const int inputRowsUsedThisCall = std::min( desc.Source.ObjectCount() * desc.Source.Height(),
		( outputRowsThisCall - 1 ) * desc.StrideHeight + 2 );

	while( chOutput->DataRowProcessed() < outputRowsThisCall ) {
		// Process a bunch of rows of input image (till channelwise convolution: expandConv + expandReLU)
		const int imageIndex = chOutput->DataRowProcessed() / desc.Result.Height();
		const int inputRowsInBuffer = std::min( desc.Source.Height(),
			chInput->DataRowProcessed() - imageIndex * desc.Source.Height() );
		const int inputRowsThisStep = std::min( getMaxInputRowsPerStep(),
			std::min( inputRowsUsedThisCall - chInput->DataRowProcessed(), desc.Source.Height() - inputRowsInBuffer ) );
		PRESUME_EXPR( inputRowsThisStep >= 0 && inputRowsThisStep <= chInput->EmptyRowCount() );

		if( inputRowsThisStep > 0 ) {
			const float* expandConvInput = input + ( chInput->DataRowProcessed() - inputRowIndex ) * inputRowSize;
			// Apply expand convolution with activation
			mathEngine.multiplyMatrixByTransposedWithFreeTerm( expandConvInput, inputRowsThisStep * inputWidth, inputChannels,
				expandFilter, expandedChannels, expandFreeTerm, chInput->EmptyRows() );
			expandActivationImpl.Process( chInput->EmptyRows(), 0, inputRowsThisStep,
				chInput->EmptyRows(), 0, inputRowsThisStep, nullptr );
			chInput->AddRows( inputRowsThisStep );
		}

		// Calculate how many output rows we can calculate with the processed input rows
		const int inputImageRowsInBuffer = chInput->DataRowProcessed() - imageIndex * desc.Source.Height();
		const int outputImageRowsCanBeProcessed = std::min( outputRowsThisCall - imageIndex * desc.Result.Height(),
			inputImageRowsInBuffer >= desc.Source.Height() ? desc.Result.Height()
				: ( inputImageRowsInBuffer < 2 ? 0 : 1 + ( inputImageRowsInBuffer - 2 ) / desc.StrideHeight ) );
		const int outputRowsCanBeProcessed = imageIndex * desc.Result.Height() + outputImageRowsCanBeProcessed;

		while( chOutput->DataRowProcessed() < outputRowsCanBeProcessed ) {
			// Process channelwise output rows (while there are any)
			const int outputRowsThisStep = std::min<int>( getMaxOutputRowsPerStep(),
				outputRowsCanBeProcessed - chOutput->DataRowProcessed() );
			PRESUME_EXPR( outputRowsThisStep <= chOutput->EmptyRowCount() );

			processChannelwise3x3( desc, outputRowsThisStep, chInput->DataRows(), chInput->DataRowIndex() % desc.Source.Height(),
				channelwiseFilter, channelwiseFreeTerm, chOutput->EmptyRows(), chOutput->DataRowProcessed() % desc.Result.Height() );
			channelwiseActivationImpl.Process( chOutput->EmptyRows(), 0, outputRowsThisStep,
				chOutput->EmptyRows(), 0, outputRowsThisStep, nullptr );
			chOutput->AddRows( outputRowsThisStep );

			if( residual && isInPlace ) {
				// Block input and output are located in the same memory
				// It's possible to simultaneously calculate down conv output and add the residual connection
				mathEngine.multiplyMatrixByTransposedMatrixAndAdd( chOutput->DataRows(), outputRowsThisStep * outputWidth,
					expandedChannels, expandedChannels, downFilter, outputChannels, expandedChannels, output,
					outputChannels );
			} else {
				mathEngine.multiplyMatrixByTransposedMatrix( chOutput->DataRows(), outputRowsThisStep * outputWidth,
					expandedChannels, expandedChannels, downFilter, outputChannels, expandedChannels,
					output, outputChannels );
			}

			if( downFreeTerm != nullptr ) {
				mathEngine.addVectorToMatrixRows( output, output, outputRowsThisStep * outputWidth, outputChannels,
					outputChannels, outputChannels, downFreeTerm );
			}

			if( residual && !isInPlace ) {
				// Input and output are located in different memory regions
				// Add residual connection
				vectorAdd( output, residualInput, output, outputRowsThisStep * outputWidth * outputChannels );
				residualInput += outputRowsThisStep * inputWidth * inputChannels;
			}

			output += outputRowsThisStep * outputChannels * outputWidth;
			chOutput->RemoveRows( outputRowsThisStep );
		}

		if( chOutput->DataRowProcessed() < desc.Result.ObjectCount() * desc.Result.Height() ) {
			const int firstInputRowNeeded = RowwiseConvFirstInputRow( chOutput->DataRowProcessed(),
				desc.Source.Height(), desc.Result.Height(), desc.StrideHeight, desc.PaddingHeight );
			if( firstInputRowNeeded > chInput->DataRowIndex() ) {
				chInput->RemoveRows( firstInputRowNeeded - chInput->DataRowIndex() );
			}
		}
	}

	if( chOutput->DataRowProcessed() == desc.Result.ObjectCount() * desc.Result.Height() ) {
		chInput.reset( nullptr );
		chOutput.reset( nullptr );
	}

	return report;
}

void CCpuMathEngine::MobileNetV2Block( const CBlobDesc& inputDesc, const CBlobDesc& outputDesc,
	const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& inputHandle,
	const CConstFloatHandle& expandFilterData, const CConstFloatHandle* expandFreeTermData,
	TActivationFunction expandActivation, float expandActivationParam, const CConstFloatHandle& channelwiseFilterData,
	const CConstFloatHandle* channelwiseFreeTermData, TActivationFunction channelwiseActivation,
	float channelwiseActivationParam, const CConstFloatHandle& downFilterData, const CConstFloatHandle* downFreeTermData,
	bool residual, const CFloatHandle& outputHandle )
{
	CCpuExecutionScope scope;
	const CCommonChannelwiseConvolutionDesc& desc = static_cast< const CCommonChannelwiseConvolutionDesc& >( convDesc );

	CRowwiseMobileNetV2 blockImpl( *this, inputDesc.Channels(), GetRaw( expandFilterData ),
		expandFreeTermData == nullptr ? nullptr : GetRaw( *expandFreeTermData ),
		desc.Source.Channels(), expandActivation, expandActivationParam, GetRaw( channelwiseFilterData ),
		channelwiseFreeTermData == nullptr ? nullptr : GetRaw( *channelwiseFreeTermData ),
		desc.StrideHeight, channelwiseActivation, channelwiseActivationParam, GetRaw( downFilterData ),
		downFreeTermData == nullptr ? nullptr : GetRaw( *downFreeTermData ),
		outputDesc.Channels(), residual );
	const CBlobDesc reshapeResult = blockImpl.Reshape( inputDesc );
	( void ) reshapeResult;
	PRESUME_EXPR( blockImpl.InOperationBufferSize() == 0 );
	PRESUME_EXPR( reshapeResult.HasEqualDimensions( outputDesc ) );
	const IRowwiseCpuImpl::CProcessingReport report = blockImpl.Process( GetRaw( inputHandle ), 0,
		inputDesc.ObjectCount() * inputDesc.Height(), GetRaw( outputHandle ), 0,
		outputDesc.ObjectCount() * outputDesc.Height(), nullptr );
	( void ) report;
	PRESUME_EXPR( report.InputRowsMayBeRemoved == inputDesc.ObjectCount() * inputDesc.Height() );
	PRESUME_EXPR( report.OutputRowsCalculated == outputDesc.ObjectCount() * outputDesc.Height() );
}

CRowwiseOperationDesc* CCpuMathEngine::InitMobileNetV2Rowwise( int inputChannels,
	const CConstFloatHandle& expandFilter, const CConstFloatHandle* expandFreeTerm, int expandedChannels,
	TActivationFunction expandActivation, float expandActivationParam,
	const CConstFloatHandle& channelwiseFilter, const CConstFloatHandle* channelwiseFreeTerm, int stride,
	TActivationFunction channelwiseActivation, float channelwiseActivationParam,
	const CConstFloatHandle& downFilter, const CConstFloatHandle* downFreeTerm, int outputChannels, bool residual )
{
	return new CRowwiseMobileNetV2( *this, inputChannels, GetRaw( expandFilter ),
		expandFreeTerm == nullptr ? nullptr : GetRaw( *expandFreeTerm ),
		expandedChannels, expandActivation, expandActivationParam, GetRaw( channelwiseFilter ),
		channelwiseFreeTerm == nullptr ? nullptr : GetRaw( *channelwiseFreeTerm ),
		stride, channelwiseActivation, channelwiseActivationParam, GetRaw( downFilter ),
		downFreeTerm == nullptr ? nullptr : GetRaw( *downFreeTerm ),
		outputChannels, residual );
}

} // namespace NeoML
