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

namespace NeoML {

static constexpr int RowwiseCacheSize = 32 * 1024;

static constexpr int RowwiseMatMulRequiredHeight = 64;

// Index of first input row needed to calculate outputRowIndex'th row of output
inline int RowwiseConvFirstInputRow( int outputRowIndex, int inputImageHeight, int outputImageHeight,
	int strideHeight, int paddingHeight )
{
	const int imageIndex = outputRowIndex / outputImageHeight;
	const int currOutputRowInImage = outputRowIndex % outputImageHeight;
	return imageIndex * inputImageHeight + std::max( 0,
		currOutputRowInImage * strideHeight - paddingHeight );
}

// Calculates how many output rows can be calculated with the given data
// and how many input rows can be released after that
inline IRowwiseCpuImpl::CProcessingReport RowwiseConvProcessingReport( int inputRowIndex, int inputRowsAvailable,
	int outputRowIndex, int outputRowsAvailable, int inputImageHeight, int outputImageHeight,
	int filterHeight, int paddingHeight, int strideHeight, int dilationHeight )
{
	const int inputRowCount = inputRowIndex + inputRowsAvailable;
	const int effectiveFilterSize = 1 + ( filterHeight - 1 ) * dilationHeight;

	// Number of input rows required to calculate given number of outputRows
	auto getRequiredInputRows = [&] ( int outputRowCount ) -> int {
		const int lastOutputRowIndex = outputRowCount - 1;
		const int imageIndex = lastOutputRowIndex / outputImageHeight;
		const int lastOutputRowInImage = lastOutputRowIndex % outputImageHeight;
		return imageIndex * inputImageHeight + std::min( inputImageHeight,
			lastOutputRowInImage * strideHeight - paddingHeight + effectiveFilterSize );
	};

	// Binary search for number of output rows which can be calculated during this call
	int binSearchMin = 0;
	int binSearchMax = outputRowsAvailable;
	while( binSearchMin != binSearchMax ) {
		const int binSearchMid = ( binSearchMin + binSearchMax + 1 ) / 2;
		const int inputRowsRequired = getRequiredInputRows( outputRowIndex + binSearchMid );
		if( inputRowsRequired <= inputRowCount ) {
			binSearchMin = binSearchMid; // There is enough data to process binSearchMid output rows
		} else {
			binSearchMax = binSearchMid - 1; // Not enough data
		}
	}
	IRowwiseCpuImpl::CProcessingReport result;
	result.OutputRowsCalculated = binSearchMin;

	const int firstRequiredInputRow = RowwiseConvFirstInputRow( outputRowIndex + result.OutputRowsCalculated,
		inputImageHeight, outputImageHeight, strideHeight, paddingHeight );
	result.InputRowsMayBeRemoved = std::min( inputRowCount, firstRequiredInputRow ) - inputRowIndex;

	return result;
}

} // namespace NeoML

// Using macro to guarantee inline
#define MOBILENET_ACTIVATION( type, reluParam, data, size ) \
	if( type == AF_ReLU ) { \
		if( reluParam > 0 ) { \
			vectorReLU( data, data, size, reluParam ); \
		} else { \
			vectorReLU( data, data, size ); \
		} \
	} else if( type == AF_HSwish ) vectorHSwish( data, data, size )
