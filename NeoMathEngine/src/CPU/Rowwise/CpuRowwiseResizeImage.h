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

inline IRowwiseCpuImpl::CProcessingReport CRowwiseImageResize::Process( const float* input, int inputRowIndex,
	int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable, float* ) const
{
	CProcessingReport report;

	const int maxRowsCalculated = outputRowIndex + outputRowsAvailable;
	const int firstBatchIndex = outputRowIndex / to.Height();
	const int lastBatchIndex = ( maxRowsCalculated - 1 ) / to.Height();

	auto calcInCoord = [this] ( int inCoord, int dimSize ) -> int {
		if( inCoord >= 0 && inCoord < dimSize ) {
			return inCoord;
		}
		switch( padding ) {
			case TBlobResizePadding::Constant:
				return inCoord;
			case TBlobResizePadding::Reflect:
				return inCoord < 0 ? -( inCoord % dimSize )
					: ( 2 * dimSize - 2 - ( inCoord % dimSize ) ) % dimSize;
			case TBlobResizePadding::Edge:
				return inCoord < 0 ? 0 : dimSize - 1;
			default:
				ASSERT_EXPR( false );
		}
		return 0;
	};

	auto calcFirstInputRowNeeded = [this, &calcInCoord] ( int nextOutputRowIndex ) -> int {
		const int imageIndex = nextOutputRowIndex / to.Height();
		const int outRowIndex = nextOutputRowIndex % to.Height();
		int inRowIndex = calcInCoord( outRowIndex - deltaTop, from.Height() );
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
		for( int i = outRowIndex; i - deltaTop < 0; ++i ) {
			inRowIndex = std::min( inRowIndex, calcInCoord( i - deltaTop, from.Height() ) );
		}
		// Check for remaining bot padding
		for( int i = std::max( outRowIndex, to.Height() - deltaBottom ); i < to.Height(); ++i ) {
			inRowIndex = std::min( inRowIndex, calcInCoord( i - deltaTop, from.Height() ) );
		}
		return inRowIndex + imageIndex * from.Height();
	};

	// The image size (used to offset pointers)
	const int inputImageSize = from.ObjectSize();
	// The image rows length
	const int totalChannels = from.Depth() * from.Channels();
	const int inputRowSize = from.Width() * totalChannels;
	const int outputRowSize = to.Width() * totalChannels;

	// If the size hasn't changed, copy the image
	if( deltaLeft == 0 && deltaRight == 0 && deltaTop == 0 && deltaBottom == 0 ) {
		PRESUME_EXPR( inputRowIndex <= outputRowIndex );
		const int rowsAfterThisCall = std::min( inputRowIndex + inputRowsAvailable,
			outputRowIndex + outputRowsAvailable );
		report.OutputRowsCalculated = rowsAfterThisCall - outputRowIndex;
		report.InputRowsMayBeRemoved = rowsAfterThisCall - inputRowIndex;
		dataCopy( output, input + ( outputRowIndex - inputRowIndex ) * inputRowSize,
			report.OutputRowsCalculated * inputRowSize );
		return report;
	}

	const int lastInputRowIndex = inputRowIndex + inputRowsAvailable - 1;
	bool hasInputData = true;

	for( int batch = firstBatchIndex; batch <= lastBatchIndex && hasInputData; ++batch ) {
		const float* inputImage = input + batch * inputImageSize - inputRowIndex * inputRowSize;
		const int firstOutputRowIndex = ( batch == firstBatchIndex ? outputRowIndex % to.Height() : 0 );
		const int lastOutputRowIndex = ( batch == lastBatchIndex ? ( maxRowsCalculated - 1 ) % to.Height()
			: to.Height() - 1 );
		for( int rowIndex = firstOutputRowIndex; rowIndex <= lastOutputRowIndex && hasInputData; ++rowIndex ) {
			const int inRowIndex = calcInCoord( rowIndex - deltaTop, from.Height() );
			if( inRowIndex < 0 || inRowIndex >= from.Height() ) {
				PRESUME_EXPR( padding == TBlobResizePadding::Constant );
				vectorFill( output, defaultValue, outputRowSize );
				output += outputRowSize;
				report.OutputRowsCalculated++;
				continue;
			}

			PRESUME_EXPR( inRowIndex + batch * from.Height() >= inputRowIndex );
			if( inRowIndex + batch * from.Height() > lastInputRowIndex ) {
				// We've calculated everythin we could from current input data
				hasInputData = false;
				break;
			}

			const float* inputRow = inputImage + inRowIndex * inputRowSize;

			// Process left padding for current row
			auto processPadding = [&] ( int delta, int firstColIndex ) -> void {
				if( delta > 0 ) {
					if( padding == TBlobResizePadding::Constant ) {
						vectorFill( output, defaultValue, delta * totalChannels );
						output += delta * totalChannels;
					} else {
						for( int outColIndex = firstColIndex; outColIndex < firstColIndex + delta; ++outColIndex ) {
							const int inColIndex = calcInCoord( outColIndex - deltaLeft, from.Width() );
							dataCopy( output, inputRow + inColIndex * totalChannels, totalChannels );
							output += totalChannels;
						}
					}
				}
			};

			// Process left padding
			processPadding( deltaLeft, 0 );

			// Copy data from intersection
			const int intersectionWidth = from.Width() - std::max( 0, -deltaLeft ) - std::max( 0, -deltaRight );
			if( intersectionWidth > 0 ) {
				dataCopy( output, inputRow + std::max( 0, -deltaLeft ) * totalChannels, intersectionWidth * totalChannels );
				output += intersectionWidth * totalChannels;
			}

			// Process right padding
			processPadding( deltaRight, to.Width() - deltaRight );

			report.OutputRowsCalculated++;
		}
	}

	report.InputRowsMayBeRemoved = std::min( inputRowsAvailable,
		calcFirstInputRowNeeded( outputRowIndex + report.OutputRowsCalculated ) - inputRowIndex );
	PRESUME_EXPR( report.InputRowsMayBeRemoved >= 0 );
	if( inputRowIndex + report.InputRowsMayBeRemoved == from.ObjectCount() * from.Height()
		&& outputRowIndex + report.OutputRowsCalculated < OutputRowCount() )
	{
		--report.InputRowsMayBeRemoved;
	}
	return report;
}

} // namespace NeoML
