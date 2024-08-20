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

#include <CpuMathEnginePrivate.h>
#include <MathEngineDnnPoolings.h>

namespace NeoML {

inline void blobMeanPooling( const CCommon2DPoolingDesc& desc, int resultRowsToProcess, const float* sourceData,
	int sourceRowIndex, float* resultData, int resultRowIndex, float* bufferPtr )
{
	auto sumMatrixRows = [] ( float* result, const float* matrix, int height, int width ) {
		dataCopy( result, matrix, width );
		matrix += width;
		for( int i = 1; i < height; ++i ) {
			vectorAdd( result, matrix, result, width );
			matrix += width;
		}
	};

	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int channels = result.Depth() * result.Channels();
	const int sourceRowSize = source.Width() * channels;
	const int resultRowSize = result.Width() * channels;
	const int windowStep = desc.StrideWidth * channels;

	const int resultRowsAfterThisCall = resultRowIndex + resultRowsToProcess;
	const int firstImageIndex = resultRowIndex / result.Height();
	const int lastImageIndex = ( resultRowsAfterThisCall - 1 ) / result.Height();

	const float* sourcePtr = sourceData + ( firstImageIndex * source.Height() - sourceRowIndex ) * sourceRowSize;
	float* resultPtr = resultData;

	for( int i = firstImageIndex; i <= lastImageIndex; ++i ) {
		const int firstRowInImage = ( i == firstImageIndex ? resultRowIndex % result.Height() : 0 );
		const int lastRowInIamge = ( i == lastImageIndex ? ( resultRowsAfterThisCall - 1 ) % result.Height()
			: result.Height() - 1 );
		for( int j = firstRowInImage; j <= lastRowInIamge; ++j ) {
			// Calculate the sum of all rows in a strip of the window height
			const float* currentStripStart = sourcePtr + sourceRowSize * desc.StrideHeight * j;
			sumMatrixRows( bufferPtr, currentStripStart, desc.FilterHeight, sourceRowSize );
			const float* currentBufferStart = bufferPtr;
			for( int k = 0; k < result.Width(); ++k ) {
				sumMatrixRows( resultPtr, currentBufferStart, desc.FilterWidth, channels );
				currentBufferStart += windowStep;
				resultPtr += channels;
			}
		}
		sourcePtr += source.ObjectSize();
	}
	// Multiply the result by the inverse of the window size
	vectorMultiply( resultData, resultData, resultRowsToProcess * resultRowSize,
		( 1.f / desc.FilterHeight / desc.FilterWidth ) );
}

} // namespace NeoML
