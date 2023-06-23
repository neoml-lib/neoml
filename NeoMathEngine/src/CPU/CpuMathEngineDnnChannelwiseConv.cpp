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

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnConv.h>
#include <CpuMathEnginePrivate.h>

namespace NeoML {

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
	const bool processBottomPadding = currOutputRowIndex + outputRowsToProcess == desc.Result.Height()
		&& ( stride == 1 || inputHeight % 2 == 1 ) && desc.Result.Height() > 1;
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

void CCpuMathEngine::MobileNetV2Block( const CBlobDesc& inputDesc, const CBlobDesc& outputDesc,
	const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& inputHandle,
	const CConstFloatHandle& expandFilterData, const CConstFloatHandle* expandFreeTermData,
	TActivationFunction expandActivation, float expandActivationParam, const CConstFloatHandle& channelwiseFilterData,
	const CConstFloatHandle* channelwiseFreeTermData, TActivationFunction channelwiseActivation,
	float channelwiseActivationParam, const CConstFloatHandle& downFilterData, const CConstFloatHandle* downFreeTermData,
	bool residual, const CFloatHandle& outputHandle )
{
	CCpuExecutionScope scope;
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );

	const int cacheSize = 32 * 1024;
	const bool isInPlace = inputHandle == outputHandle;
	const int inputChannels = inputDesc.Channels();
	const int expandedChannels = desc.Filter.Channels();
	const int outputChannels = outputDesc.Channels();
	const int inputHeight = desc.Source.Height();
	const int inputWidth = desc.Source.Width();
	const int chInputRowSize = expandedChannels * inputWidth;
	const int outputWidth = desc.Result.Width();
	const int chOutputRowSize = expandedChannels * outputWidth;

	const float* inputObject = GetRaw( inputHandle );
	const float* expandFilter = GetRaw( expandFilterData );
	const float* expandFreeTerm = expandFreeTermData == nullptr ? nullptr : GetRaw( *expandFreeTermData );
	const float* channelwiseFilter = GetRaw( channelwiseFilterData );
	const float* channelwiseFreeTerm = channelwiseFreeTermData == nullptr ? nullptr : GetRaw( *channelwiseFreeTermData );
	const float* downFilter = GetRaw( downFilterData );
	const float* downFreeTerm = downFreeTermData == nullptr ? nullptr : GetRaw( *downFreeTermData );

	const int maxInputRowsPerStep = std::max<int>( 1,
		( cacheSize / ( std::max<int>( inputChannels, expandedChannels ) * inputWidth ) ) );
	const int maxOutputRowsPerStep = std::max<int>( 1,
		( cacheSize / ( std::max<int>( outputChannels, expandedChannels ) * outputWidth ) ) );

	// Buffer for the input rows of channelwise convolution
	CFloatHandleStackVar chInputBuffVar( *this,
		std::min<int>( inputHeight, maxInputRowsPerStep + 2 ) * chInputRowSize );
	// Buffer for the output rows of channelwise convolution
	CFloatHandleStackVar chOutputBuffVar( *this, maxOutputRowsPerStep * chOutputRowSize );

	float* chInputBuff = GetRaw( chInputBuffVar.GetHandle() );
	float* chOutputBuff = GetRaw( chOutputBuffVar.GetHandle() );
	float* outputObject = GetRaw( outputHandle );

	for( int b = 0; b < inputDesc.ObjectCount(); ++b ) {
		const float* input = inputObject;
		const float* residualInput = input;
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
			multiplyMatrixByTransposedMatrix( input, inputRowsThisStep * inputWidth, inputChannels, inputChannels,
				expandFilter, expandedChannels, inputChannels, chInput, expandedChannels );
			if( expandFreeTerm != nullptr ) {
				addVectorToMatrixRows( chInput, chInput, inputRowsThisStep * inputWidth, expandedChannels, expandedChannels,
					expandedChannels, expandFreeTerm );
			}

			// Apply expand activation
			if( expandActivation == AF_HSwish ) {
				vectorHSwish( chInput, chInput, inputRowsThisStep * chInputRowSize );
			} else if( expandActivation == AF_ReLU ) {
				if( expandActivationParam > 0 ) {
					vectorReLU( chInput, chInput, inputRowsThisStep * chInputRowSize, expandActivationParam );
				} else {
					vectorReLU( chInput, chInput, inputRowsThisStep * chInputRowSize );
				}
			}
			inputRowsProcessed += inputRowsThisStep;

			// Calculate how many output rows we can calculate with the processed input rows
			const int outputRowsCanBeProcesed = inputRowsProcessed == inputHeight ? desc.Result.Height()
				: ( inputRowsProcessed < 2 ? 0 : ( inputRowsProcessed - 2 ) / desc.StrideHeight + 1 );

			while( outputRowsProcessed < outputRowsCanBeProcesed ) {
				// Process channelwise output rows (while there are any)
				const int outputRowsThisStep = std::min<int>( maxOutputRowsPerStep,
					outputRowsCanBeProcesed - outputRowsProcessed );

				processChannelwise3x3( desc, outputRowsThisStep, chInputBuff, firstInputRowInBuffer,
					channelwiseFilter, channelwiseFreeTerm, chOutputBuff, outputRowsProcessed );

				if( channelwiseActivation == AF_HSwish ) {
					vectorHSwish( chOutputBuff, chOutputBuff, outputRowsThisStep * chOutputRowSize );
				} else if( channelwiseActivation == AF_ReLU ) {
					if( channelwiseActivationParam > 0 ) {
						vectorReLU( chOutputBuff, chOutputBuff, outputRowsThisStep * chOutputRowSize,
							channelwiseActivationParam );
					} else {
						vectorReLU( chOutputBuff, chOutputBuff, outputRowsThisStep * chOutputRowSize );
					}
				}

				if( residual && isInPlace ) {
					// Block input and output are located in the same memory
					// It's possible to simultaneously calculate down conv output and add the residual connection
					multiplyMatrixByTransposedMatrixAndAdd( chOutputBuff, outputRowsThisStep * outputWidth,
						expandedChannels, expandedChannels, downFilter, outputChannels, expandedChannels, output,
						outputChannels );
				} else {
					multiplyMatrixByTransposedMatrix( chOutputBuff, outputRowsThisStep * outputWidth, expandedChannels,
						expandedChannels, downFilter, outputChannels, expandedChannels, output, outputChannels );
				}

				if( downFreeTerm != nullptr ) {
					addVectorToMatrixRows( output, output, outputRowsThisStep * outputWidth, outputChannels,
						outputChannels, outputChannels, downFreeTerm );
				}

				if( residual && !isInPlace ) {
					// Input and output are located in different memory regions
					// Add residual connection
					vectorAdd( output, residualInput, output, outputRowsThisStep * outputWidth * outputChannels );
					residualInput += outputRowsThisStep * inputWidth * inputChannels;
				}

				output += outputRowsThisStep * outputChannels * outputWidth;
				outputRowsProcessed += outputRowsThisStep;
			}

			input += inputRowsThisStep * inputChannels * inputWidth;

			if( outputRowsProcessed < desc.Result.Height() ) {
				const int firstInputRowNeeded = outputRowsProcessed * desc.StrideHeight - 1;
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

void CCpuMathEngine::MobileNetV3PreSEBlock( const CBlobDesc& inputDesc, const CBlobDesc& outputDesc,
	const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& inputHandle,
	const CConstFloatHandle& expandFilterData, const CConstFloatHandle* expandFreeTermData,
	TActivationFunction expandActivation, float expandActivationParam, const CConstFloatHandle& channelwiseFilterData,
	const CConstFloatHandle* channelwiseFreeTermData, TActivationFunction channelwiseActivation,
	float channelwiseActivationParam, const CFloatHandle& outputHandle )
{
	CCpuExecutionScope scope;
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );

	const int cacheSize = 32 * 1024;
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
	const float* channelwiseFilter = GetRaw( channelwiseFilterData );
	const float* channelwiseFreeTerm = channelwiseFreeTermData == nullptr ? nullptr : GetRaw( *channelwiseFreeTermData );

	const int maxInputRowsPerStep = std::max<int>( 1,
		( cacheSize / ( std::max<int>( inputChannels, outputChannels ) * inputWidth ) ) );
	const int maxOutputRowsPerStep = std::max<int>( 1, ( cacheSize / ( outputChannels * desc.Result.Width() ) ) );

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
			multiplyMatrixByTransposedMatrix( input, inputRowsThisStep * inputWidth, inputChannels, inputChannels,
				expandFilter, outputChannels, inputChannels, chInput, outputChannels );
			if( expandFreeTerm != nullptr ) {
				addVectorToMatrixRows( chInput, chInput, inputRowsThisStep * inputWidth, outputChannels,
					outputChannels, outputChannels, expandFreeTerm );
			}

			// Apply expand activation
			if( expandActivation == AF_HSwish ) {
				vectorHSwish( chInput, chInput, inputRowsThisStep * chInputRowSize );
			} else if( expandActivation == AF_ReLU ) {
				if( expandActivationParam > 0 ) {
					vectorReLU( chInput, chInput, inputRowsThisStep * chInputRowSize, expandActivationParam );
				} else {
					vectorReLU( chInput, chInput, inputRowsThisStep * chInputRowSize );
				}
			}
			inputRowsProcessed += inputRowsThisStep;

			// Calculate how many output rows we can calculate with the processed input rows
			const int outputRowsCanBeProcesed = inputRowsProcessed == inputHeight ? outputHeight
				: ( inputRowsProcessed < ( filterSize - padding ) ? 0 : ( inputRowsProcessed - filterSize + padding ) / stride + 1 );

			while( outputRowsProcessed < outputRowsCanBeProcesed ) {
				// Process channelwise output rows (while there are any)
				const int outputRowsThisStep = std::min<int>( maxOutputRowsPerStep,
					outputRowsCanBeProcesed - outputRowsProcessed );

				// Channelwise conv
				if( filterSize == 3 ) {
					processChannelwise3x3( desc, outputRowsThisStep, chInputBuff, firstInputRowInBuffer,
						channelwiseFilter, channelwiseFreeTerm, output, outputRowsProcessed );
				} else if( stride == 1 ) {
					processChannelwise5x5Stride1( desc, outputRowsThisStep, chInputBuff, firstInputRowInBuffer,
						channelwiseFilter, channelwiseFreeTerm, output, outputRowsProcessed );
				} else {
					processChannelwise5x5Stride2( desc, outputRowsThisStep, chInputBuff, firstInputRowInBuffer,
						channelwiseFilter, channelwiseFreeTerm, output, outputRowsProcessed );
				}

				// Apply expand activation
				if( channelwiseActivation == AF_HSwish ) {
					vectorHSwish( output, output, outputRowsThisStep * outputRowSize );
				} else if( channelwiseActivation == AF_ReLU ) {
					if( channelwiseActivationParam > 0 ) {
						vectorReLU( output, output, outputRowsThisStep * outputRowSize, channelwiseActivationParam );
					} else {
						vectorReLU( output, output, outputRowsThisStep * outputRowSize );
					}
				}

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

	const int cacheSize = 32 * 1024;
	const int maxRowsPerStep = std::max( 1, ( cacheSize / ( std::max( inputChannels, outputChannels ) * width ) ) );

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

			multiplyMatrixByDiagMatrix( input, rowsThisStep * width, inputChannels,
				squeezeVector, squeezed );

			if( activation == AF_HSwish ) {
				vectorHSwish( squeezed, squeezed, rowsThisStep * inputRowSize );
			} else if( activation == AF_ReLU ) {
				if( activationParam > 0 ) {
					vectorReLU( squeezed, squeezed, rowsThisStep * inputRowSize, activationParam );
				} else {
					vectorReLU( squeezed, squeezed, rowsThisStep * inputRowSize );
				}
			}

			multiplyMatrixByTransposedMatrix( squeezed, rowsThisStep * width, inputChannels,
				inputChannels, downFilter, outputChannels, inputChannels, output, outputChannels );

			if( downFreeTerm != nullptr ) {
				addVectorToMatrixRows( output, output, rowsThisStep * width, outputChannels,
					outputChannels, outputChannels, downFreeTerm );
			}

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

} // namespace NeoML
