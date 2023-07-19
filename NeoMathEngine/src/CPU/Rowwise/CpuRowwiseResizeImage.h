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

#include "CpuRowwiseInterface.h"
#include <CpuMathEnginePrivate.h>

namespace NeoML {

class CRowwiseImageResize : public IRowwiseCpuImpl, public CRowwiseOperationDesc {
public:
	CRowwiseImageResize( TBlobResizePadding padding, float defaultValue, int deltaLeft, int deltaRight,
			int deltaTop, int deltaBottom ) :
		padding( padding ),
		defaultValue( defaultValue ),
		deltaLeft( deltaLeft ),
		deltaRight( deltaRight ),
		deltaTop( deltaTop ),
		deltaBottom( deltaBottom )
	{
	}

	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int InputRowRequirement() const override;
	int OutputRowRequirement() const override { return 0; }
	int InOperationBufferSize() const override { return 0; }
	int OutputRowCount() const override { return to.ObjectCount() * to.Height(); }
	int OutputRowSize() const override { return to.Width() * to.Depth() * to.Channels(); }
	bool IsTrivial() const override { return false; }
	CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const override;

private:
	TBlobResizePadding padding;
	float defaultValue;
	int deltaLeft;
	int deltaRight;
	int deltaTop;
	int deltaBottom;
	CBlobDesc from;
	CBlobDesc to;

	IRowwiseCpuImpl::CProcessingReport processTrivialCase( const float* input, int inputRowIndex,
		int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable ) const;
	int inputCoord( int unnormalizedInputCoord, int dimSize ) const;
	int requiredInputRow( int outputRowIndex ) const;
	float* processHorizontalPadding( const float* input, float* output,
		int delta, int firstColIndex, int totalChannels ) const;
};

inline int CRowwiseImageResize::InputRowRequirement() const
{
	if( padding != TBlobResizePadding::Reflect ) {
		// Constant padding doesn't rely on input data
		// Reflect padding relies only on 1 line (top or bottom)
		return 1;
	}

	// Guarantee to store all the lines required for reflection padding
	return std::min( from.Height(), std::max( { 1, 1 + deltaTop, 1 + deltaBottom } ) );
}

inline CBlobDesc CRowwiseImageResize::Reshape( const CBlobDesc& inputSize )
{
	from = inputSize;
	to = from;
	to.SetDimSize( BD_Width, to.Width() + deltaLeft + deltaRight );
	to.SetDimSize( BD_Height, to.Height() + deltaTop + deltaBottom );
	return to;
}

// Processes trivial case: when operation doesn't resize anything (all deltas are zeros)
inline IRowwiseCpuImpl::CProcessingReport CRowwiseImageResize::processTrivialCase( const float* input,
	int inputRowIndex, int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable ) const
{
	PRESUME_EXPR( inputRowIndex <= outputRowIndex );
	PRESUME_EXPR( deltaLeft == 0 && deltaRight == 0 && deltaTop == 0 && deltaBottom == 0 );

	CProcessingReport report;

	const int inputRowSize = from.Width() * from.Depth() * from.Channels();
	const int rowsAfterThisCall = std::min( inputRowIndex + inputRowsAvailable,
		outputRowIndex + outputRowsAvailable );
	report.OutputRowsCalculated = rowsAfterThisCall - outputRowIndex;
	report.InputRowsMayBeRemoved = rowsAfterThisCall - inputRowIndex;
	dataCopy( output, input + ( outputRowIndex - inputRowIndex ) * inputRowSize,
		report.OutputRowsCalculated * inputRowSize );
	return report;
}

inline IRowwiseCpuImpl::CProcessingReport CRowwiseImageResize::Process( const float* input, int inputRowIndex,
	int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable, float* ) const
{
	// If the size hasn't changed, copy the image
	if( deltaLeft == 0 && deltaRight == 0 && deltaTop == 0 && deltaBottom == 0 ) {
		return processTrivialCase( input, inputRowIndex, inputRowsAvailable,
			output, outputRowIndex, outputRowsAvailable );
	}

	const int maxRowsCalculated = outputRowIndex + outputRowsAvailable;
	const int firstBatchIndex = outputRowIndex / to.Height();
	const int lastBatchIndex = ( maxRowsCalculated - 1 ) / to.Height();
	const int inputImageSize = from.ObjectSize();
	const int totalChannels = from.Depth() * from.Channels();
	const int inputRowSize = from.Width() * totalChannels;
	const int outputRowSize = to.Width() * totalChannels;

	const int lastInputRowIndex = inputRowIndex + inputRowsAvailable - 1;
	bool hasInputData = true;

	CProcessingReport report;
	for( int batch = firstBatchIndex; batch <= lastBatchIndex && hasInputData; ++batch ) {
		const float* inputImage = input + batch * inputImageSize - inputRowIndex * inputRowSize;
		const int firstOutputRowIndex = ( batch == firstBatchIndex ? outputRowIndex % to.Height() : 0 );
		const int lastOutputRowIndex = ( batch == lastBatchIndex ? ( maxRowsCalculated - 1 ) % to.Height()
			: to.Height() - 1 );
		for( int rowIndex = firstOutputRowIndex; rowIndex <= lastOutputRowIndex && hasInputData; ++rowIndex ) {
			const int inRowIndex = inputCoord( rowIndex - deltaTop, from.Height() );
			if( inRowIndex < 0 || inRowIndex >= from.Height() ) {
				PRESUME_EXPR( padding == TBlobResizePadding::Constant );
				vectorFill( output, defaultValue, outputRowSize );
				output += outputRowSize;
			} else {
				PRESUME_EXPR( inRowIndex + batch * from.Height() >= inputRowIndex );
				if( inRowIndex + batch * from.Height() > lastInputRowIndex ) {
					hasInputData = false; // We've calculated everything we could from current input data
					break;
				}
				const float* inputRow = inputImage + inRowIndex * inputRowSize;
				// Process left padding
				output = processHorizontalPadding( inputRow, output, deltaLeft, 0, totalChannels );
				// Copy columns which are not covered by padding
				const int intersectionWidth = from.Width() - std::max( 0, -deltaLeft ) - std::max( 0, -deltaRight );
				if( intersectionWidth > 0 ) {
					dataCopy( output, inputRow + std::max( 0, -deltaLeft ) * totalChannels, intersectionWidth * totalChannels );
					output += intersectionWidth * totalChannels;
				}
				// Process right padding
				output = processHorizontalPadding( inputRow, output, deltaRight, to.Width() - deltaRight, totalChannels );
			}
			report.OutputRowsCalculated++;
		}
	}

	report.InputRowsMayBeRemoved = std::min( inputRowsAvailable,
		requiredInputRow( outputRowIndex + report.OutputRowsCalculated ) - inputRowIndex );
	PRESUME_EXPR( report.InputRowsMayBeRemoved >= 0 );
	if( inputRowIndex + report.InputRowsMayBeRemoved == from.ObjectCount() * from.Height()
		&& outputRowIndex + report.OutputRowsCalculated < OutputRowCount() )
	{
		// Workaround for the sake of optimization
		// Technically, output rows filled with constant can be calculated without any input data
		// BUT for the sake of optimization RowwiseExecute doesn't call Process operation has unprocessed input data
		// That's why we don't report that "all input rows may be freed" untill we calculated all output rows
		--report.InputRowsMayBeRemoved;
	}
	return report;
}

// Calculates the input coordinate along dim size based on unnormalized value and dimension
// (unnormalized means that it may be outside of [0;dimSize-1])
// Returns coordinate of corresponding input data
// or smth outside of [0;dimSize-1] if it must be filled with constant values
inline int CRowwiseImageResize::inputCoord( int unnormalizedInputCoord, int dimSize ) const
{
	if( unnormalizedInputCoord >= 0 && unnormalizedInputCoord < dimSize ) {
		return unnormalizedInputCoord;
	}
	switch( padding ) {
		case TBlobResizePadding::Constant:
			return unnormalizedInputCoord;
		case TBlobResizePadding::Reflect:
			return unnormalizedInputCoord < 0 ? -( unnormalizedInputCoord % dimSize )
				: ( 2 * dimSize - 2 - ( unnormalizedInputCoord % dimSize ) ) % dimSize;
		case TBlobResizePadding::Edge:
			return unnormalizedInputCoord < 0 ? 0 : dimSize - 1;
		default:
			ASSERT_EXPR( false );
	}
	return 0;
}

// Calculates the index of first input row required to calculate output rows starting from outputRowIndex'th
inline int CRowwiseImageResize::requiredInputRow( int outputRowIndex ) const
{
	const int imageIndex = outputRowIndex / to.Height();
	const int outRowIndex = outputRowIndex % to.Height();
	int inRowIndex = inputCoord( outRowIndex - deltaTop, from.Height() );
	if( inRowIndex < 0 ) {
		return imageIndex * from.Height(); // Constant padding, top
	} else if( inRowIndex >= from.Height() ) {
		return ( imageIndex + 1 ) * from.Height(); // Constant padding, bot
	}

	if( padding != TBlobResizePadding::Reflect ) {
		return inRowIndex + imageIndex * from.Height();
	}

	// Corner-case of reflect padding: guarantee that remaining paddings of current image can be calculated
	// Check for remaining top padding
	if( outRowIndex < deltaTop ) {
		// Top padding itself doesn't need 0'th line of input
		// but it's needed to calculate deltaTop'th line of output (which maps onto 0'th line of input)
		inRowIndex = std::min( inRowIndex, 0 );
	}
	for( int i = outRowIndex; i - deltaTop < 0; ++i ) {
		inRowIndex = std::min( inRowIndex, inputCoord( i - deltaTop, from.Height() ) );
	}
	// Check for remaining bot padding
	for( int i = std::max( outRowIndex, to.Height() - deltaBottom ); i < to.Height(); ++i ) {
		inRowIndex = std::min( inRowIndex, inputCoord( i - deltaTop, from.Height() ) );
	}
	return inRowIndex + imageIndex * from.Height();
}

// Processes horizontal padding of size delta at current output
// Returns pointer to the output after this padding
inline float* CRowwiseImageResize::processHorizontalPadding( const float* inputRow, float* currOutput,
	int delta, int firstColIndex, int totalChannels ) const
{
	if( delta > 0 ) {
		if( padding == TBlobResizePadding::Constant ) {
			vectorFill( currOutput, defaultValue, delta * totalChannels );
			currOutput += delta * totalChannels;
		} else {
			for( int outColIndex = firstColIndex; outColIndex < firstColIndex + delta; ++outColIndex ) {
				const int inColIndex = inputCoord( outColIndex - deltaLeft, from.Width() );
				dataCopy( currOutput, inputRow + inColIndex * totalChannels, totalChannels );
				currOutput += totalChannels;
			}
		}
	}
	return currOutput;
}

} // namespace NeoML
