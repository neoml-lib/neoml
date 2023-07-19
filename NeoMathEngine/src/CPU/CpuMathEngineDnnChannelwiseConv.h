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

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <NeoMathEngine/NeoMathEngineException.h>
#include <MathEngineDnnConv.h>
#include <CpuMathEnginePrivate.h>

namespace NeoML {

inline void FillResultRow( const CCommonChannelwiseConvolutionDesc& desc, const float* freeTerm, float* result )
{
	const int channels = desc.Result.Channels();
	int count = desc.Result.Width();
	while( count > 0 ) {
		dataCopy( result, freeTerm, channels );
		result += channels;
		count--;
	}
}

inline void Process3x3Row( const CCommonChannelwiseConvolutionDesc& desc, const float* filter, const float* source, float* result )
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

	vectorEltwiseMultiplyAdd( filter1, source, result, channels );
	if( desc.Source.Width() > 1 ) {
		vectorEltwiseMultiplyAdd( filter2, source + channels, result, channels );
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
		vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
		vectorEltwiseMultiplyAdd( filter1, sourcePos + channels, resultPos, channels );
		vectorEltwiseMultiplyAdd( filter2, sourcePos + 2 * channels, resultPos, channels );
		resultPos += channels;
		sourcePos += desc.StrideWidth * channels;
		width--;
	}

	if( ( desc.StrideWidth == 1 || desc.Source.Width() % 2 == 1 ) && resultWidth > 1 ) {
		vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
		vectorEltwiseMultiplyAdd( filter1, sourcePos + channels, resultPos, channels );
	}
}

inline void Process5x5RowStride1( const CCommonChannelwiseConvolutionDesc& desc, const float* filter, const float* source, float* result )
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

	vectorEltwiseMultiplyAdd( filter2, source, result, channels );
	if( resultWidth > 1 ) {
		vectorEltwiseMultiplyAdd( filter1, source, result + channels, channels );
		vectorEltwiseMultiplyAdd( filter3, source + channels, result, channels );
		vectorEltwiseMultiplyAdd( filter2, source + channels, result + channels, channels );
		if( resultWidth > 2 ) {
			vectorEltwiseMultiplyAdd( filter4, source + 2 * channels, result, channels );
			vectorEltwiseMultiplyAdd( filter3, source + 2 * channels, result + channels, channels );
			if( resultWidth > 3 ) {
				vectorEltwiseMultiplyAdd( filter4, source + 3 * channels, result + channels, channels );
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
		vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
		vectorEltwiseMultiplyAdd( filter + channels, sourcePos + channels, resultPos, channels );
		vectorEltwiseMultiplyAdd( filter + 2 * channels, sourcePos + 2 * channels, resultPos, channels );
		vectorEltwiseMultiplyAdd( filter + 3 * channels, sourcePos + 3 * channels, resultPos, channels );
		vectorEltwiseMultiplyAdd( filter + 4 * channels, sourcePos + 4 * channels, resultPos, channels );
		resultPos += channels;
		sourcePos += channels;
		width--;
	}

	if( resultWidth > 2 ) {
		vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
		vectorEltwiseMultiplyAdd( filter + channels, sourcePos + channels, resultPos, channels );
		vectorEltwiseMultiplyAdd( filter + 2 * channels, sourcePos + 2 * channels, resultPos, channels );
		if( resultWidth > 3 ) {
			vectorEltwiseMultiplyAdd( filter + 3 * channels, sourcePos + 3 * channels, resultPos, channels );
			sourcePos += channels;
			resultPos += channels;
			vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
			vectorEltwiseMultiplyAdd( filter + channels, sourcePos + channels, resultPos, channels );
			vectorEltwiseMultiplyAdd( filter + 2 * channels, sourcePos + 2 * channels, resultPos, channels );
		}
	}
}

inline void Process5x5RowStride2( const CCommonChannelwiseConvolutionDesc& desc, const float* filter, const float* source, float* result )
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

	vectorEltwiseMultiplyAdd( filter2, source, result, channels );
	if( desc.Source.Width() > 1 ) {
		vectorEltwiseMultiplyAdd( filter3, source + channels, result, channels );
		if( desc.Source.Width() > 2 ) {
			vectorEltwiseMultiplyAdd( filter4, source + 2 * channels, result, channels );
		}
	}

	result += channels;

	while( width > 0 ) {
		vectorEltwiseMultiplyAdd( filter, source, result, channels );
		vectorEltwiseMultiplyAdd( filter1, source + channels, result, channels );
		vectorEltwiseMultiplyAdd( filter2, source + 2 * channels, result, channels );
		vectorEltwiseMultiplyAdd( filter3, source + 3 * channels, result, channels );
		vectorEltwiseMultiplyAdd( filter4, source + 4 * channels, result, channels );
		result += channels;
		source += 2 * channels;
		width--;
	}

	if( resultWidth > 1 ) {
		vectorEltwiseMultiplyAdd( filter, source, result, channels );
		vectorEltwiseMultiplyAdd( filter1, source + channels, result, channels );
		vectorEltwiseMultiplyAdd( filter2, source + 2 * channels, result, channels );
		if( desc.Source.Width() % 2 == 0 ) {
			vectorEltwiseMultiplyAdd( filter3, source + 3 * channels, result, channels );
		}
	}
}

// Calculates `outputRowsToProcess` rows of output of channelwise conv 3x3 with pad 1 and stride 1 or 2
// currInput points to currInputRowIndex'th row of input image
// currOutput points to currOutputRowIndex'th row of output image
inline void ProcessChannelwise3x3( const CCommonChannelwiseConvolutionDesc& desc, int outputRowsToProcess,
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
			FillResultRow( desc, freeTerm, outputRow );
		}

		// Omit first filter row (it's in padding)
		// Processing second and third rows
		Process3x3Row( desc, filter + filterRowSize, inputRow + inputRowSize, outputRow );
		// Check corner case: 1x1 input to 3x3 conv with 1x1 padding
		if( inputHeight > 1 ) {
			Process3x3Row( desc, filter + 2 * filterRowSize, inputRow + 2 * inputRowSize, outputRow );
		}
		--remOutputRowsThisStep;
		inputRow += stride * inputRowSize;
		outputRow += outputRowSize;
	}

	while( remOutputRowsThisStep > 0 ) {
		if( freeTerm == nullptr ) {
			vectorFill0( outputRow, outputRowSize );
		} else {
			FillResultRow( desc, freeTerm, outputRow );
		}
		// Process all 3 rows without padding checks
		Process3x3Row( desc, filter, inputRow, outputRow );
		Process3x3Row( desc, filter + filterRowSize, inputRow + inputRowSize, outputRow );
		Process3x3Row( desc, filter + 2 * filterRowSize, inputRow + 2 * inputRowSize, outputRow );
		--remOutputRowsThisStep;
		inputRow += stride * inputRowSize;
		outputRow += outputRowSize;
	}

	if( processBottomPadding ) {
		// Process bottom padding
		if( freeTerm == nullptr ) {
			vectorFill0( outputRow, outputRowSize );
		} else {
			FillResultRow( desc, freeTerm, outputRow );
		}
		Process3x3Row( desc, filter, inputRow, outputRow );
		Process3x3Row( desc, filter + filterRowSize, inputRow + inputRowSize, outputRow );
		// Omit last filter row (it's in bottom padding)
	}
}

// Calculates `outputRowsToProcess` rows of output of channelwise conv 5x5 with pad 2 and stride 1
// currInput points to currInputRowIndex'th row of input image
// currOutput points to currOutputRowIndex'th row of output image
inline void ProcessChannelwise5x5Stride1( const CCommonChannelwiseConvolutionDesc& desc, int outputRowsToProcess,
	const float* currInput, int currInputRowIndex, const float* filter, const float* freeTerm, float* currOutput, int currOutputRowIndex )
{
	const int inputHeight = desc.Source.Height();
	const int outputHeight = desc.Result.Height();
	const int inputRowSize = desc.Source.Width() * desc.Source.Channels();
	const int filterRowSize = desc.Filter.Width() * desc.Filter.Channels();
	const int outputRowSize = desc.Result.Width() * desc.Result.Channels();

	auto initOutputRow = [&] () {
		if( freeTerm != nullptr ) {
			FillResultRow( desc, freeTerm, currOutput );
		} else {
			vectorFill( currOutput, 0, outputRowSize );
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
		Process5x5RowStride1( desc, filter + 2 * filterRowSize, currInput, currOutput );
		if( inputHeight > 1 ) {
			Process5x5RowStride1( desc, filter + 3 * filterRowSize, currInput + inputRowSize, currOutput );
			if( inputHeight > 2 ) {
				Process5x5RowStride1( desc, filter + 4 * filterRowSize, currInput + 2 * inputRowSize, currOutput );
			}
		}
		switchToNextOutputRow();
	}

	if( currOutputRowIndex == 1 && outputRowsToProcess > 0 ) {
		PRESUME_EXPR( currInputRowIndex == 0 );
		initOutputRow();
		Process5x5RowStride1( desc, filter + filterRowSize, currInput, currOutput );
		Process5x5RowStride1( desc, filter + 2 * filterRowSize, currInput + inputRowSize, currOutput );
		if( inputHeight > 2 ) {
			Process5x5RowStride1( desc, filter + 3 * filterRowSize, currInput + 2 * inputRowSize, currOutput );
			if( inputHeight > 3 ) {
				Process5x5RowStride1( desc, filter + 4 * filterRowSize, currInput + 3 * inputRowSize, currOutput );
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
		Process5x5RowStride1( desc, filter + 0 * filterRowSize, firstInputUnderFilter + 0 * inputRowSize, currOutput );
		Process5x5RowStride1( desc, filter + 1 * filterRowSize, firstInputUnderFilter + 1 * inputRowSize, currOutput );
		Process5x5RowStride1( desc, filter + 2 * filterRowSize, firstInputUnderFilter + 2 * inputRowSize, currOutput );
		Process5x5RowStride1( desc, filter + 3 * filterRowSize, firstInputUnderFilter + 3 * inputRowSize, currOutput );
		Process5x5RowStride1( desc, filter + 4 * filterRowSize, firstInputUnderFilter + 4 * inputRowSize, currOutput );
		switchToNextOutputRow();
	}

	if( outputRowsToProcess > 0 && inputHeight > 3 && currOutputRowIndex == outputHeight - 2 ) {
		initOutputRow();
		const float* firstInputUnderFilter = currInput + ( currOutputRowIndex - 2 - currInputRowIndex ) * inputRowSize;
		Process5x5RowStride1( desc, filter + 0 * filterRowSize, firstInputUnderFilter + 0 * inputRowSize, currOutput );
		Process5x5RowStride1( desc, filter + 1 * filterRowSize, firstInputUnderFilter + 1 * inputRowSize, currOutput );
		Process5x5RowStride1( desc, filter + 2 * filterRowSize, firstInputUnderFilter + 2 * inputRowSize, currOutput );
		Process5x5RowStride1( desc, filter + 3 * filterRowSize, firstInputUnderFilter + 3 * inputRowSize, currOutput );
		switchToNextOutputRow();
	}

	if( outputRowsToProcess > 0 && inputHeight > 2 && currOutputRowIndex == outputHeight - 1 ) {
		initOutputRow();
		const float* firstInputUnderFilter = currInput + ( currOutputRowIndex - 2 - currInputRowIndex ) * inputRowSize;
		Process5x5RowStride1( desc, filter + 0 * filterRowSize, firstInputUnderFilter + 0 * inputRowSize, currOutput );
		Process5x5RowStride1( desc, filter + 1 * filterRowSize, firstInputUnderFilter + 1 * inputRowSize, currOutput );
		Process5x5RowStride1( desc, filter + 2 * filterRowSize, firstInputUnderFilter + 2 * inputRowSize, currOutput );
	}
}

// Calculates `outputRowsToProcess` rows of output of channelwise conv 5x5 with pad 2 and stride 2
// currInput points to currInputRowIndex'th row of input image
// currOutput points to currOutputRowIndex'th row of output image
inline void ProcessChannelwise5x5Stride2( const CCommonChannelwiseConvolutionDesc& desc, int outputRowsToProcess,
	const float* currInput, int currInputRowIndex, const float* filter, const float* freeTerm, float* currOutput, int currOutputRowIndex )
{
	const int inputHeight = desc.Source.Height();
	const int outputHeight = desc.Result.Height();
	const int inputRowSize = desc.Source.Width() * desc.Source.Channels();
	const int filterRowSize = desc.Filter.Width() * desc.Filter.Channels();
	const int outputRowSize = desc.Result.Width() * desc.Result.Channels();

	auto initOutputRow = [&] () {
		if( freeTerm != nullptr ) {
			FillResultRow( desc, freeTerm, currOutput );
		} else {
			vectorFill( currOutput, 0, outputRowSize );
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
		Process5x5RowStride2( desc, filter + 2 * filterRowSize, currInput, currOutput );
		if( inputHeight > 1 ) {
			Process5x5RowStride2( desc, filter + 3 * filterRowSize, currInput + inputRowSize, currOutput );
			if( inputHeight > 2 ) {
				Process5x5RowStride2( desc, filter + 4 * filterRowSize, currInput + 2 * inputRowSize, currOutput );
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
		Process5x5RowStride2( desc, filter + 0 * filterRowSize, firstInputUnderFilter + 0 * inputRowSize, currOutput );
		Process5x5RowStride2( desc, filter + 1 * filterRowSize, firstInputUnderFilter + 1 * inputRowSize, currOutput );
		Process5x5RowStride2( desc, filter + 2 * filterRowSize, firstInputUnderFilter + 2 * inputRowSize, currOutput );
		Process5x5RowStride2( desc, filter + 3 * filterRowSize, firstInputUnderFilter + 3 * inputRowSize, currOutput );
		Process5x5RowStride2( desc, filter + 4 * filterRowSize, firstInputUnderFilter + 4 * inputRowSize, currOutput );
		switchToNextOutputRow();
	}

	if( outputRowsToProcess > 0 && currOutputRowIndex == outputHeight - 1 ) {
		initOutputRow();
		const float* firstInputUnderFilter = currInput + ( currOutputRowIndex * 2 - 2 - currInputRowIndex ) * inputRowSize;
		Process5x5RowStride2( desc, filter + 0 * filterRowSize, firstInputUnderFilter + 0 * inputRowSize, currOutput );
		Process5x5RowStride2( desc, filter + 1 * filterRowSize, firstInputUnderFilter + 1 * inputRowSize, currOutput );
		Process5x5RowStride2( desc, filter + 2 * filterRowSize, firstInputUnderFilter + 2 * inputRowSize, currOutput );
		if( inputHeight % 2 == 0 ) {
			Process5x5RowStride2( desc, filter + 3 * filterRowSize, firstInputUnderFilter + 3 * inputRowSize, currOutput );
		}
	}
}

inline void ProcessChannelwiseNaive( const CCommonChannelwiseConvolutionDesc& desc, int outputRowsToProcess,
	const float* currInput, int currInputRowIndex, const float* filter, const float* freeTerm, float* currOutput,
	int currOutputRowIndex )
{
	const CBlobDesc& sourceDesc = desc.Source;
	const CBlobDesc& filterDesc = desc.Filter;
	const CBlobDesc& resultDesc = desc.Result;

	const int channels = sourceDesc.Channels() * sourceDesc.Depth();
	const int inputRowSize = sourceDesc.Width() * channels;
	const int outputRowSize = resultDesc.Width() * channels;
	const int filterRowSize = filterDesc.Width() * channels;

	float* const currOutputEnd = currOutput + outputRowsToProcess * outputRowSize;
	int firstFilteredRow = currOutputRowIndex * desc.StrideHeight - desc.PaddingHeight;
	for( ; currOutput < currOutputEnd; currOutput += outputRowSize, firstFilteredRow += desc.StrideHeight ) {
		if( freeTerm != 0 ) {
			float* rowStart = currOutput;
			float* rowEnd = rowStart + channels * resultDesc.Width();
			for( ; rowStart < rowEnd; rowStart += channels ) {
				dataCopy( rowStart, freeTerm, channels );
			}
		} else {
			vectorFill( currOutput, 0, resultDesc.Width() * channels );
		}

		const int filterFirstRow = std::max( 0, -firstFilteredRow );
		const int filterLastRow = std::min( filterDesc.Height(), sourceDesc.Height() - firstFilteredRow );
		const float* filterRow = filter + filterFirstRow * filterRowSize;
		const float* filterRowEnd = filter + filterLastRow * filterRowSize;
		const float* srcRow = currInput + ( firstFilteredRow + filterFirstRow - currInputRowIndex ) * inputRowSize;

		for( ; filterRow < filterRowEnd; filterRow += filterRowSize, srcRow += inputRowSize ) {
			int firstFilteredCol = -desc.PaddingWidth;
			float* currOutputPos = currOutput;
			float* currOutputPosEnd = currOutputPos + channels * resultDesc.Width();
			for( ; currOutputPos < currOutputPosEnd; currOutputPos += channels, firstFilteredCol += desc.StrideWidth ) {
				const int filterFirstCol = std::max( 0, -firstFilteredCol );
				const int filterLastCol = std::min( filterDesc.Width(), sourceDesc.Width() - firstFilteredCol );
				const float* filterPos = filterRow + filterFirstCol * channels;
				const float* filterPosEnd = filterRow + filterLastCol * channels;
				const float* srcPos = srcRow + ( firstFilteredCol + filterFirstCol ) * channels;
				for( ; filterPos < filterPosEnd; filterPos += channels, srcPos += channels ) {
					vectorEltwiseMultiplyAdd( filterPos, srcPos, currOutputPos, channels );
				}
			}
		}
	}
}

typedef void (*TChannelwiseProcessFunction)( const CCommonChannelwiseConvolutionDesc&, int, const float*, int,
	const float*, const float*, float*, int );

inline TChannelwiseProcessFunction GetChannelwiseProcessFunction( const CCommonChannelwiseConvolutionDesc& desc )
{
	if( desc.Filter.Height() == desc.Filter.Width() && desc.PaddingHeight == desc.PaddingWidth
		&& desc.StrideHeight == desc.StrideWidth )
	{
		if( desc.Filter.Width() == 3 && desc.PaddingWidth == 1 && desc.StrideWidth <= 2 ) {
			return ProcessChannelwise3x3;
		} else if( desc.Filter.Width() == 5 && desc.PaddingWidth == 2 ) {
			if( desc.StrideWidth == 1 ) {
				return ProcessChannelwise5x5Stride1;
			} else if( desc.StrideWidth == 2 ) {
				return ProcessChannelwise5x5Stride2;
			}
		}
	}

	return ProcessChannelwiseNaive;
}

} // namespace NeoML
